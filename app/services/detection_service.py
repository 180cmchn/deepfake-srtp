"""
Detection service for deepfake detection platform
"""

import os
import time
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import or_
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


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


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

            input_size = loaded_model.get("input_size", settings.MODEL_INPUT_SIZE)
            if loaded_model.get("video_temporal_enabled"):
                image = self._preprocess_image_clip(
                    file_path,
                    sequence_length=loaded_model.get("sequence_length", 8),
                    input_size=input_size,
                )
            else:
                image = self._preprocess_image(file_path, input_size=input_size)

            inference_result = await self._predict_tensor(model, image)
            prediction = inference_result["prediction"]
            confidence = inference_result["confidence"]
            probabilities = (
                inference_result["probabilities"]
                if request.return_probabilities
                else None
            )

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
                model_name=loaded_model.get("model_name"),
                model_type=loaded_model.get("model_type"),
            )

            # Schedule cleanup in background
            background_tasks.add_task(self._cleanup_file, file_path)

            return DetectionResponse(
                success=True,
                record_id=db_result.id,
                file_info={
                    "name": file_name,
                    "type": file_type,
                    "size": file_size,
                    "resolution": f"{input_size}x{input_size}",
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
                record_id=None,
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
                    return await self._detect_batch_file(
                        file_path, request, background_tasks
                    )

            tasks = [process_file(path) for path in file_paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Process sequentially
            for file_path in file_paths:
                result = await self._detect_batch_file(
                    file_path, request, background_tasks
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

    async def _detect_batch_file(
        self,
        file_path: str,
        request: BatchDetectionRequest,
        background_tasks: BackgroundTasks,
    ) -> DetectionResponse:
        """Detect a single file inside batch mode, including videos."""
        file_type = self._get_file_type(file_path)
        if file_type == "video":
            video_request = VideoDetectionRequest(
                video_path=file_path,
                model_id=request.model_id,
                model_type=request.model_type,
                confidence_threshold=request.confidence_threshold,
                preprocess=request.preprocess,
            )
            video_response = await self.detect_video(
                file_path, video_request, background_tasks
            )
            return self._video_response_to_detection_response(video_response)

        detection_request = DetectionRequest(
            model_id=request.model_id,
            model_type=request.model_type,
            confidence_threshold=request.confidence_threshold,
            return_probabilities=request.return_probabilities,
            preprocess=request.preprocess,
        )
        return await self.detect_file(file_path, detection_request, background_tasks)

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

            if loaded_model["model_type"] == "lrcn" or loaded_model.get(
                "video_temporal_enabled"
            ):
                (
                    frame_probabilities,
                    processed_clips,
                ) = await self._predict_video_with_temporal_context(
                    model,
                    frames,
                    loaded_model,
                )
                temporal_strategy = "clip_temporal_fusion"
            else:
                raw_frame_probabilities = []
                for i, (frame, _timestamp) in enumerate(frames):
                    try:
                        prediction = await self._predict_tensor(
                            model,
                            self._preprocess_frame(
                                frame,
                                input_size=loaded_model.get(
                                    "input_size", settings.MODEL_INPUT_SIZE
                                ),
                            ),
                        )
                        raw_frame_probabilities.append(prediction["probabilities"])
                    except Exception as e:
                        logger.warning(
                            "Frame processing failed", frame_number=i + 1, error=str(e)
                        )
                        raw_frame_probabilities.append(None)

                frame_probabilities = []
                valid_probabilities = [item for item in raw_frame_probabilities if item]
                if not valid_probabilities:
                    raise ValueError("No frames could be processed successfully")

                smoothed_probabilities = self._smooth_probability_sequence(
                    valid_probabilities
                )
                _, _, fallback_probabilities = self._aggregate_probability_sequence(
                    smoothed_probabilities
                )
                smoothed_index = 0
                for probabilities in raw_frame_probabilities:
                    if probabilities:
                        frame_probabilities.append(
                            smoothed_probabilities[smoothed_index]
                        )
                        smoothed_index += 1
                    else:
                        frame_probabilities.append(fallback_probabilities)

                processed_clips = 0
                temporal_strategy = "smoothed_frame_sequence"

            if not frame_probabilities:
                raise ValueError("No frames could be processed successfully")

            frame_results = self._build_frame_results_from_probabilities(
                frames,
                frame_probabilities,
                loaded_model,
            )
            predictions = [item["result"]["prediction"] for item in frame_results]
            confidences = [item["result"]["confidence"] for item in frame_results]
            aggregated_prediction, aggregated_confidence, aggregated_probabilities = (
                self._aggregate_probability_sequence(frame_probabilities)
            )

            processing_time = time.time() - start_time

            # Create response
            video_info = {
                "name": file_name,
                "size": file_size,
                "total_frames": len(frames),
                "processed_frames": len(frame_probabilities),
                "duration": frames[-1][1] if frames else 0,
            }

            summary = {
                "total_frames": len(frames),
                "processed_frames": len(frame_probabilities),
                "processed_clips": processed_clips,
                "success_rate": len(frame_probabilities) / len(frames) if frames else 0,
                "average_confidence": np.mean(confidences) if confidences else 0,
                "prediction_distribution": self._count_predictions_list(predictions),
                "temporal_strategy": temporal_strategy,
            }

            detection_result = DetectionResultSchema(
                prediction=aggregated_prediction,
                confidence=aggregated_confidence,
                probabilities=aggregated_probabilities,
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

            db_result = await self._save_detection_result(
                file_path=video_path,
                file_name=file_name,
                file_type="video",
                result=detection_result,
                model_id=request.model_id,
                model_name=loaded_model.get("model_name"),
                model_type=loaded_model.get("model_type"),
            )

            return VideoDetectionResponse(
                success=True,
                record_id=db_result.id,
                video_info=video_info,
                aggregated_result=detection_result
                if request.aggregate_results
                else None,
                frame_results=frame_results
                if request.return_frame_results and frame_results
                else None,
                summary=summary,
                processing_time=processing_time,
                created_at=db_result.created_at,
            )

        except Exception as e:
            logger.error("Video detection failed", error=str(e), video_path=video_path)
            processing_time = time.time() - start_time

            return VideoDetectionResponse(
                success=False,
                record_id=None,
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
                query = query.outerjoin(ModelRegistry).filter(
                    or_(
                        DetectionResult.model_type == model_type,
                        ModelRegistry.model_type == model_type,
                    )
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
                    model_name=result.model_name
                    or (result.model.name if result.model else "Built-in Model"),
                    model_type=result.model_type
                    or (result.model.model_type if result.model else None),
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
            detection_model_rows = (
                self.db.query(DetectionResult)
                .filter(DetectionResult.del_flag == 0)
                .all()
            )
            for result in detection_model_rows:
                model_key = result.model_type or (
                    result.model.model_type if result.model else None
                )
                if not model_key:
                    model_key = "unknown"
                detections_by_model[model_key] = (
                    detections_by_model.get(model_key, 0) + 1
                )

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

            daily_detections: Dict[str, int] = {}
            detection_dates = (
                self.db.query(DetectionResult.created_at)
                .filter(DetectionResult.del_flag == 0)
                .all()
            )
            for (created_at,) in detection_dates:
                if not created_at:
                    continue
                day_key = created_at.strftime("%Y-%m-%d")
                daily_detections[day_key] = daily_detections.get(day_key, 0) + 1

            return DetectionStatistics(
                total_detections=total_detections,
                real_detections=real_detections,
                fake_detections=fake_detections,
                average_confidence=average_confidence,
                average_processing_time=average_processing_time,
                detections_by_model=detections_by_model,
                detections_by_file_type=detections_by_file_type,
                confidence_distribution=confidence_ranges,
                daily_detections=daily_detections,
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
            self._load_model_state(
                model,
                state_dict,
                allow_partial=bool(
                    checkpoint.get("video_temporal_enabled") or model_type == "lrcn"
                ),
            )
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
                "frame_stride": checkpoint.get(
                    "frame_stride",
                    (model_record.parameters or {}).get("frame_stride", 1),
                ),
                "video_temporal_enabled": bool(
                    checkpoint.get("video_temporal_enabled")
                ),
                "training_mode": checkpoint.get("training_mode"),
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
                "frame_stride": 1,
                "video_temporal_enabled": model_type == "lrcn",
                "training_mode": "builtin_sequence"
                if model_type == "lrcn"
                else "builtin_image",
                "source": "builtin",
            }

        loaded["model"].eval()

        self._model_cache[cache_key] = loaded

        return loaded

    def _preprocess_image(self, image_path: str, input_size: int = None) -> np.ndarray:
        """Preprocess image for inference"""
        target_size = input_size or settings.MODEL_INPUT_SIZE
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            rgb_image = rgb_image.resize((target_size, target_size))
            image_array = np.array(rgb_image, dtype=np.float32) / 255.0
        return self._normalize_image_array(image_array)

    def _preprocess_frame(
        self, frame: np.ndarray, input_size: int = None
    ) -> np.ndarray:
        """Preprocess video frame for inference"""
        target_size = input_size or settings.MODEL_INPUT_SIZE
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb).resize((target_size, target_size))
        image_array = np.array(image, dtype=np.float32) / 255.0
        return self._normalize_image_array(image_array)

    def _normalize_image_array(self, image_array: np.ndarray) -> torch.Tensor:
        """Apply the same tensor normalization used during training."""
        tensor = torch.from_numpy(np.transpose(image_array, (2, 0, 1))).float()
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        return tensor.unsqueeze(0)

    def _preprocess_video_clip(
        self,
        frames: List[tuple],
        sequence_length: int,
        input_size: int = None,
        clip_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Preprocess extracted video frames into a fixed-length clip tensor."""
        if not frames:
            raise ValueError("No frames available for video clip preprocessing")

        if clip_indices:
            selected_frames = [frames[idx][0] for idx in clip_indices]
            target_length = len(selected_frames)
        else:
            target_length = max(1, sequence_length)
            selected_frames = [frame for frame, _ in frames[:target_length]]
        while len(selected_frames) < target_length:
            selected_frames.append(selected_frames[-1])

        processed_frames = [
            self._preprocess_frame(frame, input_size=input_size).squeeze(0)
            for frame in selected_frames
        ]
        return torch.stack(processed_frames, dim=0).unsqueeze(0)

    def _preprocess_image_clip(
        self,
        image_path: str,
        sequence_length: int,
        input_size: int = None,
    ) -> torch.Tensor:
        """Repeat a single image into a fixed-length clip tensor."""
        frame_tensor = self._preprocess_image(
            image_path, input_size=input_size
        ).squeeze(0)
        clip = frame_tensor.unsqueeze(0).repeat(max(1, sequence_length), 1, 1, 1)
        return clip.unsqueeze(0)

    async def _predict_tensor(
        self, model, input_tensor: torch.Tensor
    ) -> Dict[str, Any]:
        """Run a forward pass and return prediction metadata."""
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            fake_probability = probabilities[0][0].item()
            real_probability = probabilities[0][1].item()
            prediction, confidence = self._probabilities_to_prediction(
                {"fake": fake_probability, "real": real_probability}
            )
            return {
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": {
                    "fake": fake_probability,
                    "real": real_probability,
                },
            }

    async def _inference(self, model, input_tensor: torch.Tensor) -> tuple:
        """Perform model inference"""
        result = await self._predict_tensor(model, input_tensor)
        return result["prediction"], result["confidence"]

    async def _get_probabilities(
        self, model, input_tensor: torch.Tensor
    ) -> Dict[str, float]:
        """Get class probabilities"""
        result = await self._predict_tensor(model, input_tensor)
        return result["probabilities"]

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

    def _build_inference_clip_indices(
        self,
        total_frames: int,
        sequence_length: int,
        frame_stride: int,
    ) -> List[List[int]]:
        """Build overlapping inference clips that preserve temporal continuity."""
        target_length = max(1, sequence_length)
        target_stride = max(1, frame_stride)
        clip_span = ((target_length - 1) * target_stride) + 1

        def build_clip(start_index: int) -> List[int]:
            max_index = max(total_frames - 1, 0)
            return [
                min(max_index, max(0, start_index + (idx * target_stride)))
                for idx in range(target_length)
            ]

        if total_frames <= 0:
            return []
        if total_frames <= clip_span:
            return [build_clip(0)]

        step = max(1, clip_span // 2)
        start_positions = list(range(0, total_frames, step))
        final_start = max(0, total_frames - clip_span)
        if final_start not in start_positions:
            start_positions.append(final_start)

        clip_indices: List[List[int]] = []
        seen = set()
        for start_index in start_positions:
            clip = tuple(build_clip(start_index))
            if clip in seen:
                continue
            seen.add(clip)
            clip_indices.append(list(clip))
            if start_index >= final_start:
                break
        return clip_indices

    def _clip_frame_weights(self, clip_length: int) -> List[float]:
        """Favor the middle of a clip when projecting clip scores back to frames."""
        if clip_length <= 1:
            return [1.0]
        center = (clip_length - 1) / 2.0
        weights = [1.0 / (1.0 + abs(index - center)) for index in range(clip_length)]
        total = sum(weights) or 1.0
        return [weight / total for weight in weights]

    async def _predict_video_with_temporal_context(
        self,
        model,
        frames: List[tuple],
        loaded_model: Dict[str, Any],
    ) -> tuple:
        """Infer clip-level predictions and project them back to frame timeline."""
        sequence_length = max(1, int(loaded_model.get("sequence_length", 8)))
        frame_stride = max(1, int(loaded_model.get("frame_stride", 1)))
        input_size = loaded_model.get("input_size", settings.MODEL_INPUT_SIZE)
        clip_indices_list = self._build_inference_clip_indices(
            len(frames), sequence_length, frame_stride
        )

        frame_accumulators: List[List[Dict[str, float]]] = [
            [] for _ in range(len(frames))
        ]

        for clip_indices in clip_indices_list:
            processed_clip = self._preprocess_video_clip(
                frames,
                sequence_length=sequence_length,
                input_size=input_size,
                clip_indices=clip_indices,
            )
            prediction = await self._predict_tensor(model, processed_clip)
            weights = self._clip_frame_weights(len(clip_indices))

            for weight, frame_index in zip(weights, clip_indices):
                probabilities = prediction["probabilities"]
                frame_accumulators[frame_index].append(
                    {
                        "fake": probabilities["fake"] * weight,
                        "real": probabilities["real"] * weight,
                        "weight": weight,
                    }
                )

        frame_probabilities: List[Dict[str, float]] = []
        for frame_index, (frame, _timestamp) in enumerate(frames):
            accumulators = frame_accumulators[frame_index]
            if accumulators:
                total_weight = sum(item["weight"] for item in accumulators) or 1.0
                fake_probability = (
                    sum(item["fake"] for item in accumulators) / total_weight
                )
                real_probability = (
                    sum(item["real"] for item in accumulators) / total_weight
                )
            else:
                if (
                    loaded_model.get("video_temporal_enabled")
                    or loaded_model.get("model_type") == "lrcn"
                ):
                    fallback_tensor = self._preprocess_video_clip(
                        frames,
                        sequence_length=sequence_length,
                        input_size=input_size,
                        clip_indices=[frame_index] * sequence_length,
                    )
                else:
                    fallback_tensor = self._preprocess_frame(
                        frame, input_size=input_size
                    )
                frame_prediction = await self._predict_tensor(
                    model,
                    fallback_tensor,
                )
                fake_probability = frame_prediction["probabilities"]["fake"]
                real_probability = frame_prediction["probabilities"]["real"]

            probability_total = fake_probability + real_probability
            if probability_total > 0:
                fake_probability /= probability_total
                real_probability /= probability_total
            frame_probabilities.append(
                {"fake": fake_probability, "real": real_probability}
            )

        return frame_probabilities, len(clip_indices_list)

    def _smooth_probability_sequence(
        self,
        frame_probabilities: List[Dict[str, float]],
        window_size: int = 5,
    ) -> List[Dict[str, float]]:
        """Smooth frame probabilities with a short temporal window."""
        if not frame_probabilities:
            return []

        radius = max(0, window_size // 2)
        smoothed: List[Dict[str, float]] = []
        for index, probabilities in enumerate(frame_probabilities):
            start = max(0, index - radius)
            end = min(len(frame_probabilities), index + radius + 1)
            window = frame_probabilities[start:end]
            fake_probability = sum(item["fake"] for item in window) / len(window)
            blended_fake = (fake_probability + probabilities["fake"]) / 2.0
            blended_fake = max(0.0, min(1.0, blended_fake))
            smoothed.append(
                {"fake": blended_fake, "real": max(0.0, 1.0 - blended_fake)}
            )
        return smoothed

    def _probabilities_to_prediction(self, probabilities: Dict[str, float]) -> tuple:
        """Convert class probabilities to label and confidence."""
        fake_probability = probabilities.get("fake", 0.0)
        real_probability = probabilities.get("real", 0.0)
        if fake_probability >= real_probability:
            return "fake", fake_probability
        return "real", real_probability

    def _build_frame_results_from_probabilities(
        self,
        frames: List[tuple],
        frame_probabilities: List[Dict[str, float]],
        loaded_model: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build frame result payloads from a probability timeline."""
        frame_results = []
        for index, ((_, timestamp), probabilities) in enumerate(
            zip(frames, frame_probabilities), start=1
        ):
            prediction, confidence = self._probabilities_to_prediction(probabilities)
            frame_results.append(
                {
                    "frame_number": index,
                    "timestamp": timestamp,
                    "result": {
                        "prediction": prediction,
                        "confidence": confidence,
                        "probabilities": probabilities,
                        "processing_time": 0.0,
                        "model_info": {
                            "model_id": loaded_model.get("model_id"),
                            "model_name": loaded_model.get("model_name"),
                            "model_type": loaded_model["model_type"],
                        },
                    },
                }
            )
        return frame_results

    def _aggregate_probability_sequence(
        self,
        frame_probabilities: List[Dict[str, float]],
    ) -> tuple:
        """Aggregate a probability timeline into a video-level prediction."""
        if not frame_probabilities:
            return "unknown", 0.0, {"fake": 0.0, "real": 0.0}

        fake_probability = sum(item["fake"] for item in frame_probabilities) / len(
            frame_probabilities
        )
        fake_probability = max(0.0, min(1.0, fake_probability))
        probabilities = {
            "fake": fake_probability,
            "real": max(0.0, 1.0 - fake_probability),
        }
        prediction, confidence = self._probabilities_to_prediction(probabilities)
        return prediction, confidence, probabilities

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
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
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
            model_name=model_name,
            model_type=model_type,
        )

        self.db.add(db_result)
        self.db.commit()
        self.db.refresh(db_result)

        return db_result

    def _video_response_to_detection_response(
        self, response: VideoDetectionResponse
    ) -> DetectionResponse:
        """Convert video detection responses to the generic detection shape."""
        detection_result = response.aggregated_result
        if detection_result is None and response.frame_results:
            detection_result = response.frame_results[0].result

        return DetectionResponse(
            success=response.success,
            record_id=response.record_id,
            file_info={
                "name": response.video_info.get("name"),
                "type": "video",
                "size": response.video_info.get("size"),
                "resolution": None,
                "total_frames": response.video_info.get("total_frames"),
                "processed_frames": response.video_info.get("processed_frames"),
                "duration": response.video_info.get("duration"),
            },
            result=detection_result,
            error_message=None if response.success else "Video detection failed",
            processing_time=response.processing_time,
            created_at=response.created_at,
        )

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
        if checkpoint.get("video_temporal_enabled") and model_type != "lrcn":
            kwargs.update(
                {
                    "video_temporal_enabled": True,
                    "temporal_hidden_size": checkpoint.get(
                        "temporal_hidden_size",
                        parameters.get("temporal_hidden_size", 256),
                    ),
                    "temporal_num_layers": checkpoint.get(
                        "temporal_num_layers",
                        parameters.get("temporal_num_layers", 2),
                    ),
                    "feature_projection_size": checkpoint.get(
                        "feature_projection_size",
                        parameters.get("feature_projection_size", 256),
                    ),
                }
            )
        elif model_type == "lrcn":
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
                    "frame_projection_size": checkpoint.get(
                        "frame_projection_size",
                        parameters.get("frame_projection_size", 256),
                    ),
                }
            )
        kwargs["pretrained"] = False
        return create_model(model_type, **kwargs)

    def _load_model_state(
        self,
        model,
        state_dict: Dict[str, Any],
        allow_partial: bool = False,
    ) -> None:
        """Load a checkpoint with an optional compatibility fallback."""
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            if not allow_partial:
                raise
            load_result = model.load_state_dict(state_dict, strict=False)
            logger.warning(
                "Checkpoint loaded with partial compatibility",
                missing_keys=list(load_result.missing_keys),
                unexpected_keys=list(load_result.unexpected_keys),
                error=str(exc),
            )
