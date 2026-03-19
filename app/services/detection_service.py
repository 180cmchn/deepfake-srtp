"""
Detection service for deepfake detection platform
"""

import os
import time
import asyncio
from pathlib import Path
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
    DetectionRequest,
    DetectionResponse,
    BatchDetectionRequest,
    BatchDetectionResponse,
    VideoDetectionRequest,
    VideoDetectionResponse,
    DetectionHistory,
    DetectionHistoryList,
    DetectionStatistics,
    DetectionResult as DetectionResultSchema,
    PredictionType,
    FileType,
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
        background_tasks: BackgroundTasks,
    ) -> DetectionResponse:
        """Detect deepfake in a single file"""
        start_time = time.time()

        try:
            # Get file info
            file_name = os.path.basename(file_path)
            file_type = self._get_file_type(file_path)
            file_size = os.path.getsize(file_path)

            # Load model
            loaded_model = await self._load_model(request.model_id, request.model_type)
            model = loaded_model["model"]

            if loaded_model["model_type"] == "lrcn":
                raise ValueError("LRCN models only support video detection")

            # Preprocess image
            image = self._preprocess_image(
                file_path,
                input_size=loaded_model.get("input_size", settings.MODEL_INPUT_SIZE),
            )

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
                    "model_id": loaded_model.get("model_id"),
                    "model_name": loaded_model.get("model_name"),
                    "model_type": loaded_model["model_type"],
                    "input_size": loaded_model.get(
                        "input_size", settings.MODEL_INPUT_SIZE
                    ),
                    "source": loaded_model.get("source"),
                },
            )

            # Save to database
            db_result = await self._save_detection_result(
                file_path=file_path,
                file_name=file_name,
                file_type=file_type,
                result=detection_result,
                model_id=request.model_id,
            )

            # Schedule cleanup in background
            background_tasks.add_task(self._cleanup_file, file_path)

            return DetectionResponse(
                success=True,
                file_info={
                    "name": file_name,
                    "type": file_type,
                    "size": file_size,
                    "resolution": f"{image.shape[1]}x{image.shape[0]}"
                    if hasattr(image, "shape")
                    else None,
                },
                result=detection_result,
                processing_time=processing_time,
                created_at=db_result.created_at,
            )

        except Exception as e:
            logger.error("Detection failed", error=str(e), file_path=file_path)
            processing_time = time.time() - start_time

            return DetectionResponse(
                success=False,
                file_info={"name": os.path.basename(file_path)},
                error_message=str(e),
                processing_time=processing_time,
                created_at=time.time(),
            )

    async def detect_batch(
        self,
        file_paths: List[str],
        request: BatchDetectionRequest,
        background_tasks: BackgroundTasks,
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
                        preprocess=request.preprocess,
                    )
                    return await self.detect_file(
                        file_path, detection_request, background_tasks
                    )

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
                    preprocess=request.preprocess,
                )
                result = await self.detect_file(
                    file_path, detection_request, background_tasks
                )
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
            "predictions": self._count_predictions(results),
        }

        return BatchDetectionResponse(
            success=processed_files > 0,
            total_files=len(file_paths),
            processed_files=processed_files,
            failed_files=failed_files,
            results=results,
            summary=summary,
            processing_time=processing_time,
            created_at=time.time(),
        )

    async def detect_video(
        self,
        video_path: str,
        request: VideoDetectionRequest,
        background_tasks: BackgroundTasks,
    ) -> VideoDetectionResponse:
        """Detect deepfake in video file"""
        start_time = time.time()

        try:
            # Get video info
            file_name = os.path.basename(video_path)
            file_size = os.path.getsize(video_path)

            # Extract frames
            frames = await self._extract_frames(
                video_path, request.frame_extraction_interval, request.max_frames
            )

            if not frames:
                raise ValueError("No frames could be extracted from video")

            # Load model
            loaded_model = await self._load_model(request.model_id, request.model_type)
            model = loaded_model["model"]

            frame_results = []
            predictions = []
            confidences = []

            if loaded_model["model_type"] == "lrcn":
                processed_clip = self._preprocess_video_clip(
                    frames,
                    sequence_length=loaded_model.get("sequence_length", 16),
                    input_size=loaded_model.get(
                        "input_size", settings.MODEL_INPUT_SIZE
                    ),
                )
                aggregated_prediction, aggregated_confidence = await self._inference(
                    model, processed_clip
                )
                predictions.append(aggregated_prediction)
                confidences.append(aggregated_confidence)
            else:
                for i, (frame, timestamp) in enumerate(frames):
                    try:
                        processed_frame = self._preprocess_frame(
                            frame,
                            input_size=loaded_model.get(
                                "input_size", settings.MODEL_INPUT_SIZE
                            ),
                        )
                        prediction, confidence = await self._inference(
                            model, processed_frame
                        )

                        predictions.append(prediction)
                        confidences.append(confidence)
                        frame_results.append(
                            {
                                "frame_number": i + 1,
                                "timestamp": timestamp,
                                "result": {
                                    "prediction": prediction,
                                    "confidence": confidence,
                                    "processing_time": 0.0,
                                    "model_info": {
                                        "model_id": loaded_model.get("model_id"),
                                        "model_name": loaded_model.get("model_name"),
                                        "model_type": loaded_model["model_type"],
                                    },
                                },
                            }
                        )
                    except Exception as e:
                        logger.warning(
                            "Frame processing failed", frame_number=i + 1, error=str(e)
                        )
                        continue

                if not frame_results:
                    raise ValueError("No frames could be processed successfully")

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
                "duration": frames[-1][1] if frames else 0,
            }

            summary = {
                "total_frames": len(frames),
                "processed_frames": len(frame_results),
                "success_rate": len(frame_results) / len(frames) if frames else 0,
                "average_confidence": np.mean(confidences) if confidences else 0,
                "prediction_distribution": self._count_predictions_list(predictions),
            }

            return VideoDetectionResponse(
                success=True,
                video_info=video_info,
                aggregated_result=DetectionResultSchema(
                    prediction=aggregated_prediction,
                    confidence=aggregated_confidence,
                    processing_time=processing_time,
                    model_info={
                        "model_id": loaded_model.get("model_id"),
                        "model_name": loaded_model.get("model_name"),
                        "model_type": loaded_model["model_type"],
                        "input_size": loaded_model.get(
                            "input_size", settings.MODEL_INPUT_SIZE
                        ),
                        "source": loaded_model.get("source"),
                    },
                )
                if request.aggregate_results
                else None,
                frame_results=frame_results
                if request.return_frame_results and frame_results
                else None,
                summary=summary,
                processing_time=processing_time,
                created_at=time.time(),
            )

        except Exception as e:
            logger.error("Video detection failed", error=str(e), video_path=video_path)
            processing_time = time.time() - start_time

            return VideoDetectionResponse(
                success=False,
                video_info={"name": os.path.basename(video_path)},
                summary={},
                processing_time=processing_time,
                created_at=time.time(),
            )
        finally:
            # Schedule cleanup
            background_tasks.add_task(self._cleanup_file, video_path)

    async def get_history(
        self,
        skip: int = 0,
        limit: int = 100,
        prediction: Optional[str] = None,
        model_type: Optional[str] = None,
        user_id: Optional[str] = None,
        search: Optional[str] = None,
        order_by: str = "created_at",
        order_desc: bool = True,
    ) -> DetectionHistoryList:
        """Get detection history with filtering, searching, and pagination"""
        try:
            query = self.db.query(DetectionResult).filter(DetectionResult.del_flag == 0)

            # Apply filters
            if prediction:
                query = query.filter(DetectionResult.prediction == prediction)

            if model_type:
                query = query.join(ModelRegistry).filter(
                    ModelRegistry.model_type == model_type
                )

            # TODO: Add user_id filtering when DetectionResult model has user_id field
            # if user_id:
            #     query = query.filter(DetectionResult.user_id == user_id)

            # Apply search
            if search:
                search_term = f"%{search}%"
                query = query.filter(DetectionResult.file_name.ilike(search_term))

            # Apply ordering
            if hasattr(DetectionResult, order_by):
                order_column = getattr(DetectionResult, order_by)
                if order_desc:
                    query = query.order_by(order_column.desc())
                else:
                    query = query.order_by(order_column.asc())
            else:
                # Default ordering
                query = query.order_by(DetectionResult.created_at.desc())

            # Get total count
            total = query.count()

            # Apply pagination
            results = query.offset(skip).limit(limit).all()

            detections = []
            for result in results:
                try:
                    file_type_enum = FileType(result.file_type)
                except Exception:
                    file_type_enum = FileType.IMAGE

                try:
                    prediction_enum = PredictionType(result.prediction)
                except Exception:
                    prediction_enum = PredictionType.FAKE

                detection = DetectionHistory(
                    id=result.id,
                    file_name=result.file_name,
                    file_type=file_type_enum,
                    prediction=prediction_enum,
                    confidence=result.confidence
                    if result.confidence is not None
                    else 0.0,
                    processing_time=result.processing_time
                    if result.processing_time is not None
                    else 0.0,
                    model_name=result.model.name if result.model else "Unknown",
                    created_at=result.created_at,
                )
                detections.append(detection)

            return DetectionHistoryList(
                detections=detections,
                total=total,
                page=skip // limit + 1,
                size=limit,
                pages=(total + limit - 1) // limit,
            )

        except Exception as e:
            logger.error("Failed to get detection history", error=str(e))
            raise

    async def get_statistics(self) -> DetectionStatistics:
        """Get detection statistics"""
        try:
            # Get total detections
            total_detections = (
                self.db.query(DetectionResult)
                .filter(DetectionResult.del_flag == 0)
                .count()
            )

            # Get predictions count
            real_detections = (
                self.db.query(DetectionResult)
                .filter(
                    DetectionResult.del_flag == 0, DetectionResult.prediction == "real"
                )
                .count()
            )

            fake_detections = (
                self.db.query(DetectionResult)
                .filter(
                    DetectionResult.del_flag == 0, DetectionResult.prediction == "fake"
                )
                .count()
            )

            # Calculate averages
            avg_confidence = (
                self.db.query(DetectionResult.confidence)
                .filter(DetectionResult.del_flag == 0)
                .all()
            )
            average_confidence = (
                np.mean([c[0] for c in avg_confidence]) if avg_confidence else 0
            )

            avg_processing_time = (
                self.db.query(DetectionResult.processing_time)
                .filter(DetectionResult.del_flag == 0)
                .all()
            )
            average_processing_time = (
                np.mean([t[0] for t in avg_processing_time])
                if avg_processing_time
                else 0
            )

            # Get detections by model
            detections_by_model = {}
            models = self.db.query(ModelRegistry).all()
            for model in models:
                count = (
                    self.db.query(DetectionResult)
                    .filter(
                        DetectionResult.del_flag == 0,
                        DetectionResult.model_id == model.id,
                    )
                    .count()
                )
                if count > 0:
                    detections_by_model[model.model_type] = count

            # Get detections by file type
            detections_by_file_type = {}
            for file_type in ["image", "video"]:
                count = (
                    self.db.query(DetectionResult)
                    .filter(
                        DetectionResult.del_flag == 0,
                        DetectionResult.file_type == file_type,
                    )
                    .count()
                )
                if count > 0:
                    detections_by_file_type[file_type] = count

            # Confidence distribution
            confidence_ranges = {
                "0.0-0.2": 0,
                "0.2-0.4": 0,
                "0.4-0.6": 0,
                "0.6-0.8": 0,
                "0.8-1.0": 0,
            }

            confidences = (
                self.db.query(DetectionResult.confidence)
                .filter(DetectionResult.del_flag == 0)
                .all()
            )

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
                daily_detections={},  # TODO: Implement daily statistics
            )

        except Exception as e:
            logger.error("Failed to get detection statistics", error=str(e))
            raise

    async def delete_detection_record(self, detection_id: int) -> bool:
        """Delete detection record"""
        try:
            result = (
                self.db.query(DetectionResult)
                .filter(
                    DetectionResult.id == detection_id, DetectionResult.del_flag == 0
                )
                .first()
            )

            if not result:
                return False

            result.del_flag = 1
            self.db.commit()

            logger.info("Detection record deleted", detection_id=detection_id)
            return True

        except Exception as e:
            logger.error(
                "Failed to delete detection record",
                error=str(e),
                detection_id=detection_id,
            )
            self.db.rollback()
            raise

    # Private helper methods

    def _get_file_type(self, file_path: str) -> str:
        """Determine file type from extension"""
        ext = os.path.splitext(file_path)[1].lower()
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]

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
            model_record = (
                self.db.query(ModelRegistry)
                .filter(ModelRegistry.id == model_id, ModelRegistry.del_flag == 0)
                .first()
            )

            if not model_record:
                raise ValueError(f"Model with ID {model_id} not found")

            model_type = model_record.model_type
            checkpoint = self._load_checkpoint(model_record.file_path)
            model = self._build_model_from_checkpoint(
                model_type, checkpoint, model_record
            )
            state_dict = checkpoint.get("model_state_dict")
            if not state_dict:
                raise ValueError("Checkpoint does not contain model_state_dict")
            model.load_state_dict(state_dict)
            loaded = {
                "model": model,
                "model_id": model_record.id,
                "model_name": model_record.name,
                "model_type": model_type,
                "input_size": checkpoint.get(
                    "input_size", model_record.input_size or settings.MODEL_INPUT_SIZE
                ),
                "sequence_length": checkpoint.get(
                    "sequence_length",
                    (model_record.parameters or {}).get("sequence_length", 16),
                ),
                "source": "registry",
            }
        else:
            model_type = model_type or settings.DEFAULT_MODEL_TYPE
            model = create_model(model_type)
            loaded = {
                "model": model,
                "model_id": None,
                "model_name": model_type,
                "model_type": model_type,
                "input_size": settings.MODEL_INPUT_SIZE,
                "sequence_length": 16,
                "source": "builtin",
            }

        loaded["model"].eval()

        self._model_cache[cache_key] = loaded

        return loaded

    def _preprocess_image(self, image_path: str, input_size: int = None) -> np.ndarray:
        """Preprocess image for inference"""
        target_size = input_size or settings.MODEL_INPUT_SIZE
        image = Image.open(image_path).convert("RGB")
        image = image.resize((target_size, target_size))
        image = np.array(image) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        image = torch.FloatTensor(image).unsqueeze(0)  # Add batch dimension
        return image

    def _preprocess_frame(
        self, frame: np.ndarray, input_size: int = None
    ) -> np.ndarray:
        """Preprocess video frame for inference"""
        target_size = input_size or settings.MODEL_INPUT_SIZE
        image = Image.fromarray(frame).convert("RGB")
        image = image.resize((target_size, target_size))
        image = np.array(image) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        image = torch.FloatTensor(image).unsqueeze(0)  # Add batch dimension
        return image

    def _preprocess_video_clip(
        self, frames: List[tuple], sequence_length: int, input_size: int = None
    ) -> torch.Tensor:
        """Preprocess extracted video frames into a fixed-length clip tensor."""
        if not frames:
            raise ValueError("No frames available for video clip preprocessing")

        target_length = max(1, sequence_length)
        selected_frames = [frame for frame, _ in frames[:target_length]]
        while len(selected_frames) < target_length:
            selected_frames.append(selected_frames[-1])

        processed_frames = [
            self._preprocess_frame(frame, input_size=input_size).squeeze(0)
            for frame in selected_frames
        ]
        return torch.stack(processed_frames, dim=0).unsqueeze(0)

    async def _inference(self, model, input_tensor: torch.Tensor) -> tuple:
        """Perform model inference"""
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            prediction = "fake" if predicted.item() == 0 else "real"
            confidence = confidence.item()

            return prediction, confidence

    async def _get_probabilities(
        self, model, input_tensor: torch.Tensor
    ) -> Dict[str, float]:
        """Get class probabilities"""
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)

            return {
                "fake": probabilities[0][0].item(),
                "real": probabilities[0][1].item(),
            }

    async def _extract_frames(
        self, video_path: str, interval: int, max_frames: int
    ) -> List[tuple]:
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

    def _aggregate_results(
        self, predictions: List[str], confidences: List[float], threshold: float
    ) -> tuple:
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
            avg_fake_conf = np.mean(
                [c for p, c in zip(predictions, confidences) if p == "fake"]
            )
            avg_real_conf = np.mean(
                [c for p, c in zip(predictions, confidences) if p == "real"]
            )
            prediction = "fake" if avg_fake_conf > avg_real_conf else "real"

        # Calculate average confidence for the predicted class
        prediction_confidences = [
            c for p, c in zip(predictions, confidences) if p == prediction
        ]
        confidence = np.mean(prediction_confidences) if prediction_confidences else 0.0

        return prediction, confidence

    async def _save_detection_result(
        self,
        file_path: str,
        file_name: str,
        file_type: str,
        result: DetectionResultSchema,
        model_id: Optional[int],
    ) -> DetectionResult:
        """Save detection result to database"""
        db_result = DetectionResult(
            file_path=file_path,
            file_name=file_name,
            file_type=file_type,
            prediction=result.prediction,
            confidence=result.confidence,
            processing_time=result.processing_time,
            model_id=model_id,
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

    def _load_checkpoint(self, file_path: str) -> Dict[str, Any]:
        """Load a retained checkpoint file from disk."""
        resolved_path = Path(file_path).expanduser()
        if not resolved_path.is_absolute():
            resolved_path = (Path.cwd() / resolved_path).resolve()
        if not resolved_path.exists() or not resolved_path.is_file():
            raise ValueError(f"Model checkpoint not found: {resolved_path}")
        return torch.load(str(resolved_path), map_location="cpu")

    def _build_model_from_checkpoint(
        self,
        model_type: str,
        checkpoint: Dict[str, Any],
        model_record: ModelRegistry,
    ):
        """Build a model instance that matches checkpoint metadata."""
        parameters = model_record.parameters or {}
        kwargs: Dict[str, Any] = {
            "num_classes": checkpoint.get("num_classes", model_record.num_classes or 2)
        }
        if model_type == "lrcn":
            kwargs.update(
                {
                    "input_size": checkpoint.get(
                        "feature_input_size",
                        parameters.get("feature_input_size", 25088),
                    ),
                    "hidden_size": checkpoint.get(
                        "hidden_size", parameters.get("hidden_size", 512)
                    ),
                    "num_layers": checkpoint.get(
                        "num_layers", parameters.get("num_layers", 2)
                    ),
                }
            )
        return create_model(model_type, **kwargs)
