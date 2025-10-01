"""
Detection service for deepfake detection platform
"""

import os
import time
import asyncio
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from fastapi import BackgroundTasks
import torch
import cv2
from PIL import Image
import numpy as np

from app.core.database import get_db_session
from app.core.logging import logger
from app.core.config import settings
from app.models.database_models import DetectionResult, ModelRegistry
from app.models.ml_models import ModelRegistry as MLModelRegistry, create_model
from app.schemas.detection import (
    DetectionRequest, DetectionResponse, BatchDetectionRequest, 
    BatchDetectionResponse, VideoDetectionRequest, VideoDetectionResponse,
    DetectionHistory, DetectionHistoryList, DetectionStatistics,
    DetectionResult as DetectionResultSchema, PredictionType, FileType
)


class DetectionService:
    """Service for handling deepfake detection operations"""
    
    def __init__(self, db: Session):
        self.db = db
        self._model_cache = {}
    
    async def detect_file(
        self,
        file_path: str,
        request: DetectionRequest,
        background_tasks: BackgroundTasks
    ) -> DetectionResponse:
        """Detect deepfake in a single file"""
        start_time = time.time()
        
        try:
            # Get file info
            file_name = os.path.basename(file_path)
            file_type = self._get_file_type(file_path)
            file_size = os.path.getsize(file_path)
            
            # Load model
            model = await self._load_model(request.model_id, request.model_type)
            
            # Preprocess image
            image = self._preprocess_image(file_path)
            
            # Perform inference
            prediction, confidence = await self._inference(model, image)
            
            # Get probabilities if requested
            probabilities = None
            if request.return_probabilities:
                probabilities = await self._get_probabilities(model, image)
            
            processing_time = time.time() - start_time
            
            # Create detection result
            detection_result = DetectionResultSchema(
                prediction=prediction,
                confidence=confidence,
                probabilities=probabilities,
                processing_time=processing_time,
                model_info={
                    "model_type": model.__class__.__name__,
                    "input_size": settings.MODEL_INPUT_SIZE
                }
            )
            
            # Save to database
            db_result = await self._save_detection_result(
                file_path=file_path,
                file_name=file_name,
                file_type=file_type,
                result=detection_result,
                model_id=request.model_id
            )
            
            # Schedule cleanup in background
            background_tasks.add_task(self._cleanup_file, file_path)
            
            return DetectionResponse(
                success=True,
                file_info={
                    "name": file_name,
                    "type": file_type,
                    "size": file_size,
                    "resolution": f"{image.shape[1]}x{image.shape[0]}" if hasattr(image, 'shape') else None
                },
                result=detection_result,
                processing_time=processing_time,
                created_at=db_result.created_at
            )
            
        except Exception as e:
            logger.error("Detection failed", error=str(e), file_path=file_path)
            processing_time = time.time() - start_time
            
            return DetectionResponse(
                success=False,
                file_info={"name": os.path.basename(file_path)},
                error_message=str(e),
                processing_time=processing_time,
                created_at=time.time()
            )
    
    async def detect_batch(
        self,
        file_paths: List[str],
        request: BatchDetectionRequest,
        background_tasks: BackgroundTasks
    ) -> BatchDetectionResponse:
        """Detect deepfake in multiple files"""
        start_time = time.time()
        results = []
        processed_files = 0
        failed_files = 0
        
        # Process files in parallel if enabled
        if request.parallel_processing:
            semaphore = asyncio.Semaphore(request.max_workers)
            
            async def process_file(file_path):
                async with semaphore:
                    detection_request = DetectionRequest(
                        model_id=request.model_id,
                        model_type=request.model_type,
                        confidence_threshold=request.confidence_threshold,
                        return_probabilities=request.return_probabilities,
                        preprocess=request.preprocess
                    )
                    return await self.detect_file(file_path, detection_request, background_tasks)
            
            tasks = [process_file(path) for path in file_paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Process sequentially
            for file_path in file_paths:
                detection_request = DetectionRequest(
                    model_id=request.model_id,
                    model_type=request.model_type,
                    confidence_threshold=request.confidence_threshold,
                    return_probabilities=request.return_probabilities,
                    preprocess=request.preprocess
                )
                result = await self.detect_file(file_path, detection_request, background_tasks)
                results.append(result)
        
        # Count results
        for result in results:
            if isinstance(result, Exception):
                failed_files += 1
            elif result.success:
                processed_files += 1
            else:
                failed_files += 1
        
        processing_time = time.time() - start_time
        
        # Create summary
        summary = {
            "total_files": len(file_paths),
            "processed_files": processed_files,
            "failed_files": failed_files,
            "success_rate": processed_files / len(file_paths) if file_paths else 0,
            "average_confidence": self._calculate_average_confidence(results),
            "predictions": self._count_predictions(results)
        }
        
        return BatchDetectionResponse(
            success=processed_files > 0,
            total_files=len(file_paths),
            processed_files=processed_files,
            failed_files=failed_files,
            results=results,
            summary=summary,
            processing_time=processing_time,
            created_at=time.time()
        )
    
    async def detect_video(
        self,
        video_path: str,
        request: VideoDetectionRequest,
        background_tasks: BackgroundTasks
    ) -> VideoDetectionResponse:
        """Detect deepfake in video file"""
        start_time = time.time()
        
        try:
            # Get video info
            file_name = os.path.basename(video_path)
            file_size = os.path.getsize(video_path)
            
            # Extract frames
            frames = await self._extract_frames(
                video_path,
                request.frame_extraction_interval,
                request.max_frames
            )
            
            if not frames:
                raise ValueError("No frames could be extracted from video")
            
            # Load model
            model = await self._load_model(request.model_id, request.model_type)
            
            # Process each frame
            frame_results = []
            predictions = []
            confidences = []
            
            for i, (frame, timestamp) in enumerate(frames):
                try:
                    # Preprocess frame
                    processed_frame = self._preprocess_frame(frame)
                    
                    # Perform inference
                    prediction, confidence = await self._inference(model, processed_frame)
                    
                    predictions.append(prediction)
                    confidences.append(confidence)
                    
                    frame_result = {
                        "frame_number": i + 1,
                        "timestamp": timestamp,
                        "prediction": prediction,
                        "confidence": confidence
                    }
                    frame_results.append(frame_result)
                    
                except Exception as e:
                    logger.warning("Frame processing failed", 
                                 frame_number=i+1, 
                                 error=str(e))
                    continue
            
            if not frame_results:
                raise ValueError("No frames could be processed successfully")
            
            # Aggregate results
            aggregated_prediction, aggregated_confidence = self._aggregate_results(
                predictions, confidences, request.confidence_threshold
            )
            
            processing_time = time.time() - start_time
            
            # Create response
            video_info = {
                "name": file_name,
                "size": file_size,
                "total_frames": len(frames),
                "processed_frames": len(frame_results),
                "duration": frames[-1][1] if frames else 0
            }
            
            summary = {
                "total_frames": len(frames),
                "processed_frames": len(frame_results),
                "success_rate": len(frame_results) / len(frames) if frames else 0,
                "average_confidence": np.mean(confidences) if confidences else 0,
                "prediction_distribution": self._count_predictions_list(predictions)
            }
            
            return VideoDetectionResponse(
                success=True,
                video_info=video_info,
                aggregated_result=DetectionResultSchema(
                    prediction=aggregated_prediction,
                    confidence=aggregated_confidence,
                    processing_time=processing_time
                ) if request.aggregate_results else None,
                frame_results=frame_results if request.return_frame_results else None,
                summary=summary,
                processing_time=processing_time,
                created_at=time.time()
            )
            
        except Exception as e:
            logger.error("Video detection failed", error=str(e), video_path=video_path)
            processing_time = time.time() - start_time
            
            return VideoDetectionResponse(
                success=False,
                video_info={"name": os.path.basename(video_path)},
                summary={},
                processing_time=processing_time,
                created_at=time.time()
            )
        finally:
            # Schedule cleanup
            background_tasks.add_task(self._cleanup_file, video_path)
    
    async def get_history(
        self,
        skip: int = 0,
        limit: int = 100,
        prediction: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> DetectionHistoryList:
        """Get detection history"""
        try:
            query = self.db.query(DetectionResult).filter(DetectionResult.del_flag == 0)
            
            if prediction:
                query = query.filter(DetectionResult.prediction == prediction)
            
            if model_type:
                query = query.join(ModelRegistry).filter(ModelRegistry.model_type == model_type)
            
            total = query.count()
            results = query.order_by(DetectionResult.created_at.desc()).offset(skip).limit(limit).all()
            
            detections = []
            for result in results:
                detection = DetectionHistory(
                    id=result.id,
                    file_name=result.file_name,
                    file_type=FileType(result.file_type),
                    prediction=PredictionType(result.prediction),
                    confidence=result.confidence,
                    processing_time=result.processing_time,
                    model_name=result.model.name if result.model else "Unknown",
                    created_at=result.created_at
                )
                detections.append(detection)
            
            return DetectionHistoryList(
                detections=detections,
                total=total,
                page=skip // limit + 1,
                size=limit,
                pages=(total + limit - 1) // limit
            )
            
        except Exception as e:
            logger.error("Failed to get detection history", error=str(e))
            raise
    
    async def get_statistics(self) -> DetectionStatistics:
        """Get detection statistics"""
        try:
            # Get total detections
            total_detections = self.db.query(DetectionResult).filter(
                DetectionResult.del_flag == 0
            ).count()
            
            # Get predictions count
            real_detections = self.db.query(DetectionResult).filter(
                DetectionResult.del_flag == 0,
                DetectionResult.prediction == "real"
            ).count()
            
            fake_detections = self.db.query(DetectionResult).filter(
                DetectionResult.del_flag == 0,
                DetectionResult.prediction == "fake"
            ).count()
            
            # Calculate averages
            avg_confidence = self.db.query(DetectionResult.confidence).filter(
                DetectionResult.del_flag == 0
            ).all()
            average_confidence = np.mean([c[0] for c in avg_confidence]) if avg_confidence else 0
            
            avg_processing_time = self.db.query(DetectionResult.processing_time).filter(
                DetectionResult.del_flag == 0
            ).all()
            average_processing_time = np.mean([t[0] for t in avg_processing_time]) if avg_processing_time else 0
            
            # Get detections by model
            detections_by_model = {}
            models = self.db.query(ModelRegistry).all()
            for model in models:
                count = self.db.query(DetectionResult).filter(
                    DetectionResult.del_flag == 0,
                    DetectionResult.model_id == model.id
                ).count()
                if count > 0:
                    detections_by_model[model.model_type] = count
            
            # Get detections by file type
            detections_by_file_type = {}
            for file_type in ["image", "video"]:
                count = self.db.query(DetectionResult).filter(
                    DetectionResult.del_flag == 0,
                    DetectionResult.file_type == file_type
                ).count()
                if count > 0:
                    detections_by_file_type[file_type] = count
            
            # Confidence distribution
            confidence_ranges = {
                "0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0,
                "0.6-0.8": 0, "0.8-1.0": 0
            }
            
            confidences = self.db.query(DetectionResult.confidence).filter(
                DetectionResult.del_flag == 0
            ).all()
            
            for conf in confidences:
                value = conf[0]
                if value < 0.2:
                    confidence_ranges["0.0-0.2"] += 1
                elif value < 0.4:
                    confidence_ranges["0.2-0.4"] += 1
                elif value < 0.6:
                    confidence_ranges["0.4-0.6"] += 1
                elif value < 0.8:
                    confidence_ranges["0.6-0.8"] += 1
                else:
                    confidence_ranges["0.8-1.0"] += 1
            
            return DetectionStatistics(
                total_detections=total_detections,
                real_detections=real_detections,
                fake_detections=fake_detections,
                average_confidence=average_confidence,
                average_processing_time=average_processing_time,
                detections_by_model=detections_by_model,
                detections_by_file_type=detections_by_file_type,
                confidence_distribution=confidence_ranges,
                daily_detections={}  # TODO: Implement daily statistics
            )
            
        except Exception as e:
            logger.error("Failed to get detection statistics", error=str(e))
            raise
    
    async def delete_detection_record(self, detection_id: int) -> bool:
        """Delete detection record"""
        try:
            result = self.db.query(DetectionResult).filter(
                DetectionResult.id == detection_id,
                DetectionResult.del_flag == 0
            ).first()
            
            if not result:
                return False
            
            result.del_flag = 1
            self.db.commit()
            
            logger.info("Detection record deleted", detection_id=detection_id)
            return True
            
        except Exception as e:
            logger.error("Failed to delete detection record", error=str(e), detection_id=detection_id)
            self.db.rollback()
            raise
    
    # Private helper methods
    
    def _get_file_type(self, file_path: str) -> str:
        """Determine file type from extension"""
        ext = os.path.splitext(file_path)[1].lower()
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        
        if ext in image_extensions:
            return "image"
        elif ext in video_extensions:
            return "video"
        else:
            return "unknown"
    
    async def _load_model(self, model_id: Optional[int], model_type: Optional[str]):
        """Load model for inference"""
        cache_key = f"{model_id}_{model_type}"
        
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        if model_id:
            # Load specific model from database
            model_record = self.db.query(ModelRegistry).filter(
                ModelRegistry.id == model_id,
                ModelRegistry.del_flag == 0
            ).first()
            
            if not model_record:
                raise ValueError(f"Model with ID {model_id} not found")
            
            model_type = model_record.model_type
            # TODO: Load actual model from file
            model = create_model(model_type)
        else:
            # Load default model type
            model_type = model_type or settings.DEFAULT_MODEL_TYPE
            model = create_model(model_type)
        
        # Set to evaluation mode
        model.eval()
        
        # Cache the model
        self._model_cache[cache_key] = model
        
        return model
    
    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for inference"""
        image = Image.open(image_path).convert('RGB')
        image = image.resize((settings.MODEL_INPUT_SIZE, settings.MODEL_INPUT_SIZE))
        image = np.array(image) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        image = torch.FloatTensor(image).unsqueeze(0)  # Add batch dimension
        return image
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess video frame for inference"""
        image = Image.fromarray(frame).convert('RGB')
        image = image.resize((settings.MODEL_INPUT_SIZE, settings.MODEL_INPUT_SIZE))
        image = np.array(image) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        image = torch.FloatTensor(image).unsqueeze(0)  # Add batch dimension
        return image
    
    async def _inference(self, model, input_tensor: torch.Tensor) -> tuple:
        """Perform model inference"""
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            prediction = "fake" if predicted.item() == 0 else "real"
            confidence = confidence.item()
            
            return prediction, confidence
    
    async def _get_probabilities(self, model, input_tensor: torch.Tensor) -> Dict[str, float]:
        """Get class probabilities"""
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            return {
                "fake": probabilities[0][0].item(),
                "real": probabilities[0][1].item()
            }
    
    async def _extract_frames(self, video_path: str, interval: int, max_frames: int) -> List[tuple]:
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        extracted_count = 0
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % interval == 0:
                    timestamp = frame_count / fps if fps > 0 else 0
                    frames.append((frame, timestamp))
                    extracted_count += 1
                    
                    if extracted_count >= max_frames:
                        break
                
                frame_count += 1
            
            return frames
            
        finally:
            cap.release()
    
    def _aggregate_results(self, predictions: List[str], confidences: List[float], threshold: float) -> tuple:
        """Aggregate frame results for video"""
        if not predictions:
            return "unknown", 0.0
        
        # Count predictions
        fake_count = predictions.count("fake")
        real_count = predictions.count("real")
        
        # Use majority vote
        if fake_count > real_count:
            prediction = "fake"
        elif real_count > fake_count:
            prediction = "real"
        else:
            # Tie breaker: use average confidence
            avg_fake_conf = np.mean([c for p, c in zip(predictions, confidences) if p == "fake"])
            avg_real_conf = np.mean([c for p, c in zip(predictions, confidences) if p == "real"])
            prediction = "fake" if avg_fake_conf > avg_real_conf else "real"
        
        # Calculate average confidence for the predicted class
        prediction_confidences = [c for p, c in zip(predictions, confidences) if p == prediction]
        confidence = np.mean(prediction_confidences) if prediction_confidences else 0.0
        
        return prediction, confidence
    
    async def _save_detection_result(
        self,
        file_path: str,
        file_name: str,
        file_type: str,
        result: DetectionResultSchema,
        model_id: Optional[int]
    ) -> DetectionResult:
        """Save detection result to database"""
        db_result = DetectionResult(
            file_path=file_path,
            file_name=file_name,
            file_type=file_type,
            prediction=result.prediction,
            confidence=result.confidence,
            processing_time=result.processing_time,
            model_id=model_id
        )
        
        self.db.add(db_result)
        self.db.commit()
        self.db.refresh(db_result)
        
        return db_result
    
    async def _cleanup_file(self, file_path: str):
        """Clean up temporary file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info("Temporary file cleaned up", file_path=file_path)
        except Exception as e:
            logger.warning("Failed to cleanup file", file_path=file_path, error=str(e))
    
    def _calculate_average_confidence(self, results: List) -> float:
        """Calculate average confidence from results"""
        confidences = []
        for result in results:
            if isinstance(result, DetectionResponse) and result.result:
                confidences.append(result.result.confidence)
        
        return np.mean(confidences) if confidences else 0.0
    
    def _count_predictions(self, results: List) -> Dict[str, int]:
        """Count predictions from results"""
        counts = {"fake": 0, "real": 0}
        
        for result in results:
            if isinstance(result, DetectionResponse) and result.result:
                prediction = result.result.prediction
                if prediction in counts:
                    counts[prediction] += 1
        
        return counts
    
    def _count_predictions_list(self, predictions: List[str]) -> Dict[str, int]:
        """Count predictions from list"""
        counts = {"fake": 0, "real": 0}
        
        for prediction in predictions:
            if prediction in counts:
                counts[prediction] += 1
        
        return counts
