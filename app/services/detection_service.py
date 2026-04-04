"""
Detection service for deepfake detection platform
"""

import os
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, or_
from fastapi import BackgroundTasks
import torch
import cv2
from PIL import Image
import numpy as np

from app.core.database import get_db_session
from app.core.logging import logger
from app.core.config import settings
from app.core.video_aggregation import aggregate_probability_sequence
from app.core.video_face_roi import SingleFaceRoiProcessor
from app.models.database_models import (
    AuditLog,
    DetectionResult,
    ModelRegistry,
    ModelStatus,
)
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
    DetectionDecisionMetrics,
    DetectionStatus,
    PredictionType,
    FileType,
)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
READY_MODEL_STATUSES = (ModelStatus.READY.value, ModelStatus.DEPLOYED.value)
MODEL_UNAVAILABLE_ERROR_PREFIX = (
    "No usable ready/deployed registry model is available for detection"
)
MODEL_UNAVAILABLE_ERROR_CODE = "model_unavailable"


class DetectionService:
    """Service for handling deepfake detection operations"""

    def __init__(self, db: Session):
        self.db = db
        self._model_cache = {}
        self._face_roi_processor = SingleFaceRoiProcessor()

    def _resolve_detection_status(
        self, status_value: Optional[str], error_message: Optional[str]
    ) -> DetectionStatus:
        try:
            return DetectionStatus(status_value)
        except Exception:
            return (
                DetectionStatus.FAILED
                if (error_message or "").strip()
                else DetectionStatus.COMPLETED
            )

    def _resolve_prediction_type(
        self, prediction_value: Optional[str]
    ) -> Optional[PredictionType]:
        try:
            return PredictionType(prediction_value)
        except Exception:
            return None

    def _resolve_error_code(self, error_message: Optional[str]) -> Optional[str]:
        if isinstance(error_message, str) and error_message.startswith(
            MODEL_UNAVAILABLE_ERROR_PREFIX
        ):
            return MODEL_UNAVAILABLE_ERROR_CODE
        return None

    def _apply_detection_result_filters(
        self,
        query,
        *,
        prediction: Optional[str] = None,
        status: Optional[str] = None,
        model_type: Optional[str] = None,
        search: Optional[str] = None,
    ):
        if prediction:
            query = query.filter(DetectionResult.prediction == prediction)

        if status:
            if status == DetectionStatus.FAILED.value:
                query = query.filter(
                    or_(
                        DetectionResult.status == status,
                        and_(
                            DetectionResult.status.is_(None),
                            DetectionResult.error_message.isnot(None),
                            func.trim(DetectionResult.error_message) != "",
                        ),
                    )
                )
            elif status == DetectionStatus.COMPLETED.value:
                query = query.filter(
                    or_(
                        DetectionResult.status == status,
                        and_(
                            DetectionResult.status.is_(None),
                            or_(
                                DetectionResult.error_message.is_(None),
                                func.trim(DetectionResult.error_message) == "",
                            ),
                        ),
                    )
                )
            else:
                query = query.filter(DetectionResult.status == status)

        if model_type:
            query = query.outerjoin(ModelRegistry).filter(
                or_(
                    DetectionResult.model_type == model_type,
                    ModelRegistry.model_type == model_type,
                )
            )

        if search:
            search_term = f"%{search}%"
            query = query.filter(DetectionResult.file_name.ilike(search_term))

        return query

    async def detect_file(
        self,
        file_path: str,
        request: DetectionRequest,
        background_tasks: BackgroundTasks,
        original_file_name: Optional[str] = None,
        audit_context: Optional[Dict[str, Any]] = None,
        audit_action: str = "detect",
    ) -> DetectionResponse:
        """Detect deepfake in a single file"""
        start_time = time.time()
        stored_file_name = os.path.basename(file_path)
        resolved_original_file_name = original_file_name or stored_file_name
        file_type = self._get_file_type(file_path)
        file_size = None
        loaded_model = None

        try:
            # Get file info
            file_size = os.path.getsize(file_path)

            # Load model
            loaded_model = await self._load_model(request.model_id, request.model_type)
            model = loaded_model["model"]
            face_roi_policy = self._resolve_face_roi_policy(loaded_model)

            if loaded_model["model_type"] == "lrcn":
                raise ValueError("LRCN models only support video detection")

            input_size = loaded_model.get("input_size", settings.MODEL_INPUT_SIZE)
            if loaded_model.get("video_temporal_enabled"):
                image = self._preprocess_image_clip(
                    file_path,
                    sequence_length=loaded_model.get("sequence_length", 8),
                    input_size=input_size,
                    face_roi_policy=face_roi_policy,
                )
            else:
                image = self._preprocess_image(
                    file_path,
                    input_size=input_size,
                    face_roi_policy=face_roi_policy,
                )

            inference_result = await self._predict_tensor(model, image)
            raw_probabilities = inference_result["probabilities"]
            prediction, confidence = self._probabilities_to_prediction(
                raw_probabilities,
                confidence_threshold=request.confidence_threshold,
            )
            decision_metrics = self._build_decision_metrics(
                raw_probabilities,
                request.confidence_threshold,
                prediction,
                confidence,
            )
            probabilities = raw_probabilities if request.return_probabilities else None

            processing_time = time.time() - start_time

            # Create detection result
            detection_result = DetectionResultSchema(
                prediction=prediction,
                confidence=confidence,
                probabilities=probabilities,
                decision_metrics=decision_metrics,
                processing_time=processing_time,
                model_info=self._build_model_info(loaded_model),
            )

            # Save to database
            db_result = await self._save_detection_result(
                file_path=file_path,
                original_file_name=resolved_original_file_name,
                file_type=file_type,
                result=detection_result,
                file_size=file_size,
                status="completed",
                model_id=loaded_model.get("model_id"),
                model_name=loaded_model.get("model_name"),
                model_type=loaded_model.get("model_type"),
            )
            self._write_detection_audit_log(
                action=audit_action,
                status="completed",
                db_result=db_result,
                file_path=file_path,
                original_file_name=resolved_original_file_name,
                file_type=file_type,
                file_size=file_size,
                model_id=loaded_model.get("model_id"),
                model_name=loaded_model.get("model_name"),
                model_type=loaded_model.get("model_type"),
                prediction=detection_result.prediction,
                confidence=detection_result.confidence,
                processing_time=processing_time,
                audit_context=audit_context,
            )

            return DetectionResponse(
                success=True,
                record_id=db_result.id,
                file_info={
                    "name": stored_file_name,
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
            error_message = str(e)
            error_code = self._resolve_error_code(error_message)
            db_result = await self._persist_failed_detection_result(
                file_path=file_path,
                original_file_name=resolved_original_file_name,
                file_type=file_type,
                file_size=file_size,
                processing_time=processing_time,
                error_message=error_message,
                loaded_model=loaded_model,
            )
            self._write_detection_audit_log(
                action=audit_action,
                status="failed",
                db_result=db_result,
                file_path=file_path,
                original_file_name=resolved_original_file_name,
                file_type=file_type,
                file_size=file_size,
                model_id=(loaded_model or {}).get("model_id"),
                model_name=(loaded_model or {}).get("model_name"),
                model_type=(loaded_model or {}).get("model_type"),
                prediction="failed",
                confidence=0.0,
                processing_time=processing_time,
                error_message=error_message,
                audit_context=audit_context,
            )

            return DetectionResponse(
                success=False,
                record_id=db_result.id if db_result else None,
                file_info={"name": stored_file_name},
                error_message=error_message,
                error_code=error_code,
                processing_time=processing_time,
                created_at=db_result.created_at if db_result else datetime.utcnow(),
            )
        finally:
            background_tasks.add_task(self._cleanup_file, file_path)

    async def detect_batch(
        self,
        file_paths: List[str],
        request: BatchDetectionRequest,
        background_tasks: BackgroundTasks,
        original_file_names: Optional[Dict[str, str]] = None,
        audit_context: Optional[Dict[str, Any]] = None,
        audit_action: str = "detect_batch",
    ) -> BatchDetectionResponse:
        """Detect deepfake in multiple files"""
        start_time = time.time()
        results = []
        processed_files = 0
        failed_files = 0
        original_file_names = original_file_names or {}

        # Process files in parallel if enabled
        if request.parallel_processing:
            semaphore = asyncio.Semaphore(request.max_workers)

            async def process_file(file_path):
                async with semaphore:
                    return await self._detect_batch_file(
                        file_path,
                        request,
                        background_tasks,
                        original_file_name=original_file_names.get(file_path),
                        audit_context=audit_context,
                        audit_action=audit_action,
                    )

            tasks = [process_file(path) for path in file_paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Process sequentially
            for file_path in file_paths:
                try:
                    result = await self._detect_batch_file(
                        file_path,
                        request,
                        background_tasks,
                        original_file_name=original_file_names.get(file_path),
                        audit_context=audit_context,
                        audit_action=audit_action,
                    )
                except Exception as exc:
                    result = self._build_batch_exception_response(
                        file_path=file_path,
                        original_file_name=original_file_names.get(file_path),
                        error=exc,
                    )
                results.append(result)

        normalized_results: List[DetectionResponse] = []
        for index, result in enumerate(results):
            if isinstance(result, Exception):
                normalized_results.append(
                    self._build_batch_exception_response(
                        file_path=file_paths[index],
                        original_file_name=original_file_names.get(file_paths[index]),
                        error=result,
                    )
                )
            else:
                normalized_results.append(result)
        results = normalized_results

        # Count results
        for result in results:
            if isinstance(result, Exception):
                failed_files += 1
            elif isinstance(result, DetectionResponse) and result.success:
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

    def _build_batch_exception_response(
        self,
        *,
        file_path: str,
        original_file_name: Optional[str],
        error: Exception,
    ) -> DetectionResponse:
        error_message = str(error)
        error_code = self._resolve_error_code(error_message)
        file_name = original_file_name or os.path.basename(file_path)
        file_type = self._get_file_type(file_path)
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else None

        return DetectionResponse(
            success=False,
            record_id=None,
            file_info={
                "name": file_name,
                "type": file_type,
                "size": file_size,
            },
            error_message=error_message,
            error_code=error_code,
            processing_time=0.0,
            created_at=datetime.utcnow(),
        )

    async def _detect_batch_file(
        self,
        file_path: str,
        request: BatchDetectionRequest,
        background_tasks: BackgroundTasks,
        original_file_name: Optional[str] = None,
        audit_context: Optional[Dict[str, Any]] = None,
        audit_action: str = "detect_batch",
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
                file_path,
                video_request,
                background_tasks,
                original_file_name=original_file_name,
                audit_context=audit_context,
                audit_action=audit_action,
            )
            return self._video_response_to_detection_response(video_response)

        detection_request = DetectionRequest(
            model_id=request.model_id,
            model_type=request.model_type,
            confidence_threshold=request.confidence_threshold,
            return_probabilities=request.return_probabilities,
            preprocess=request.preprocess,
        )
        return await self.detect_file(
            file_path,
            detection_request,
            background_tasks,
            original_file_name=original_file_name,
            audit_context=audit_context,
            audit_action=audit_action,
        )

    async def detect_video(
        self,
        video_path: str,
        request: VideoDetectionRequest,
        background_tasks: BackgroundTasks,
        original_file_name: Optional[str] = None,
        audit_context: Optional[Dict[str, Any]] = None,
        audit_action: str = "detect_video",
    ) -> VideoDetectionResponse:
        """Detect deepfake in video file"""
        start_time = time.time()
        stored_file_name = os.path.basename(video_path)
        resolved_original_file_name = original_file_name or stored_file_name
        file_size = None
        loaded_model = None
        sampled_frame_count = None
        analyzed_frame_count = None
        fallback_filled_frame_count = 0
        sampled_duration_seconds = None
        source_video_metadata = {
            "source_total_frames": None,
            "source_fps": None,
            "source_duration_seconds": None,
        }

        try:
            # Get video info
            file_size = os.path.getsize(video_path)
            source_video_metadata = self._get_video_source_metadata(video_path)

            # Extract frames
            frames = await self._extract_frames(
                video_path, request.frame_extraction_interval, request.max_frames
            )

            if not frames:
                raise ValueError("No frames could be extracted from video")

            # Load model
            loaded_model = await self._load_model(request.model_id, request.model_type)
            model = loaded_model["model"]
            face_roi_policy = self._resolve_face_roi_policy(loaded_model)
            aggregation_policy = self._resolve_video_aggregation_policy(loaded_model)

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
                    face_roi_policy,
                )
                analyzed_frame_count = len(frames)
                fallback_filled_frame_count = 0
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
                                face_roi_policy=face_roi_policy,
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

                analyzed_frame_count = len(valid_probabilities)
                fallback_filled_frame_count = max(
                    0, len(raw_frame_probabilities) - analyzed_frame_count
                )

                smoothed_probabilities = self._smooth_probability_sequence(
                    valid_probabilities
                )
                _, _, fallback_probabilities, _ = self._aggregate_probability_sequence(
                    smoothed_probabilities,
                    confidence_threshold=request.confidence_threshold,
                    aggregation_policy=aggregation_policy,
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
                request.confidence_threshold,
            )
            predictions = [item["result"]["prediction"] for item in frame_results]
            confidences = [item["result"]["confidence"] for item in frame_results]
            (
                aggregated_prediction,
                aggregated_confidence,
                aggregated_probabilities,
                aggregated_decision_metrics,
            ) = self._aggregate_probability_sequence(
                frame_probabilities,
                confidence_threshold=request.confidence_threshold,
                aggregation_policy=aggregation_policy,
            )

            processing_time = time.time() - start_time
            sampled_frame_count = len(frames)
            if analyzed_frame_count is None:
                analyzed_frame_count = len(frame_probabilities)
            sampled_duration_seconds = frames[-1][1] if frames else None

            # Create response
            video_info = {
                "name": stored_file_name,
                "size": file_size,
                "source_total_frames": source_video_metadata.get("source_total_frames"),
                "source_fps": source_video_metadata.get("source_fps"),
                "source_duration_seconds": source_video_metadata.get(
                    "source_duration_seconds"
                ),
                "sampled_frame_count": sampled_frame_count,
                "analyzed_frame_count": analyzed_frame_count,
                "fallback_filled_frame_count": fallback_filled_frame_count,
                "sampled_duration_seconds": sampled_duration_seconds,
                "total_frames": sampled_frame_count,
                "processed_frames": analyzed_frame_count,
                "duration": sampled_duration_seconds,
            }

            summary = {
                "source_total_frames": source_video_metadata.get("source_total_frames"),
                "source_fps": source_video_metadata.get("source_fps"),
                "source_duration_seconds": source_video_metadata.get(
                    "source_duration_seconds"
                ),
                "sampled_frame_count": sampled_frame_count,
                "analyzed_frame_count": analyzed_frame_count,
                "fallback_filled_frame_count": fallback_filled_frame_count,
                "sampled_duration_seconds": sampled_duration_seconds,
                "total_frames": sampled_frame_count,
                "processed_frames": analyzed_frame_count,
                "duration": sampled_duration_seconds,
                "processed_clips": processed_clips,
                "success_rate": analyzed_frame_count / sampled_frame_count
                if sampled_frame_count
                else 0,
                "average_confidence": np.mean(confidences) if confidences else 0,
                "prediction_distribution": self._count_predictions_list(predictions),
                "temporal_strategy": temporal_strategy,
                "aggregation_strategy": "weighted_topk_persistence",
                "face_roi_enabled": face_roi_policy["face_roi_enabled"],
                "face_roi_effective_enabled": face_roi_policy[
                    "face_roi_effective_enabled"
                ],
                "face_roi_policy_version": face_roi_policy["face_roi_policy_version"],
                "face_roi_selection_policy": face_roi_policy[
                    "face_roi_selection_policy"
                ],
                "video_aggregation_topk_ratio": aggregation_policy["topk_ratio"],
            }

            detection_result = DetectionResultSchema(
                prediction=aggregated_prediction,
                confidence=aggregated_confidence,
                probabilities=aggregated_probabilities,
                decision_metrics=aggregated_decision_metrics,
                processing_time=processing_time,
                model_info=self._build_model_info(loaded_model),
            )

            db_result = await self._save_detection_result(
                file_path=video_path,
                original_file_name=resolved_original_file_name,
                file_type="video",
                result=detection_result,
                file_size=file_size,
                status="completed",
                model_id=loaded_model.get("model_id"),
                model_name=loaded_model.get("model_name"),
                model_type=loaded_model.get("model_type"),
                video_metadata={
                    "source_total_frames": source_video_metadata.get(
                        "source_total_frames"
                    ),
                    "source_fps": source_video_metadata.get("source_fps"),
                    "source_duration_seconds": source_video_metadata.get(
                        "source_duration_seconds"
                    ),
                    "sampled_frame_count": sampled_frame_count,
                    "analyzed_frame_count": analyzed_frame_count,
                    "fallback_filled_frame_count": fallback_filled_frame_count,
                    "sampled_duration_seconds": sampled_duration_seconds,
                },
            )
            self._write_detection_audit_log(
                action=audit_action,
                status="completed",
                db_result=db_result,
                file_path=video_path,
                original_file_name=resolved_original_file_name,
                file_type="video",
                file_size=file_size,
                model_id=loaded_model.get("model_id"),
                model_name=loaded_model.get("model_name"),
                model_type=loaded_model.get("model_type"),
                prediction=detection_result.prediction,
                confidence=detection_result.confidence,
                processing_time=processing_time,
                audit_context=audit_context,
                extra_details={
                    "source_total_frames": source_video_metadata.get(
                        "source_total_frames"
                    ),
                    "source_fps": source_video_metadata.get("source_fps"),
                    "source_duration_seconds": source_video_metadata.get(
                        "source_duration_seconds"
                    ),
                    "sampled_frame_count": sampled_frame_count,
                    "analyzed_frame_count": analyzed_frame_count,
                    "fallback_filled_frame_count": fallback_filled_frame_count,
                    "sampled_duration_seconds": sampled_duration_seconds,
                    "total_frames": sampled_frame_count,
                    "processed_frames": analyzed_frame_count,
                    "duration": sampled_duration_seconds,
                    "processed_clips": processed_clips,
                    "temporal_strategy": temporal_strategy,
                    "aggregation_strategy": "weighted_topk_persistence",
                },
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
            error_message = str(e)
            error_code = self._resolve_error_code(error_message)
            db_result = await self._persist_failed_detection_result(
                file_path=video_path,
                original_file_name=resolved_original_file_name,
                file_type="video",
                file_size=file_size,
                processing_time=processing_time,
                error_message=error_message,
                loaded_model=loaded_model,
                video_metadata={
                    "source_total_frames": source_video_metadata.get(
                        "source_total_frames"
                    ),
                    "source_fps": source_video_metadata.get("source_fps"),
                    "source_duration_seconds": source_video_metadata.get(
                        "source_duration_seconds"
                    ),
                    "sampled_frame_count": sampled_frame_count,
                    "analyzed_frame_count": analyzed_frame_count,
                    "fallback_filled_frame_count": fallback_filled_frame_count,
                    "sampled_duration_seconds": sampled_duration_seconds,
                },
            )
            self._write_detection_audit_log(
                action=audit_action,
                status="failed",
                db_result=db_result,
                file_path=video_path,
                original_file_name=resolved_original_file_name,
                file_type="video",
                file_size=file_size,
                model_id=(loaded_model or {}).get("model_id"),
                model_name=(loaded_model or {}).get("model_name"),
                model_type=(loaded_model or {}).get("model_type"),
                prediction="failed",
                confidence=0.0,
                processing_time=processing_time,
                error_message=error_message,
                audit_context=audit_context,
                extra_details={
                    "source_total_frames": source_video_metadata.get(
                        "source_total_frames"
                    ),
                    "source_fps": source_video_metadata.get("source_fps"),
                    "source_duration_seconds": source_video_metadata.get(
                        "source_duration_seconds"
                    ),
                    "sampled_frame_count": sampled_frame_count,
                    "analyzed_frame_count": analyzed_frame_count,
                    "fallback_filled_frame_count": fallback_filled_frame_count,
                    "sampled_duration_seconds": sampled_duration_seconds,
                },
            )

            return VideoDetectionResponse(
                success=False,
                record_id=db_result.id if db_result else None,
                video_info={
                    "name": stored_file_name,
                    "size": file_size,
                    "source_total_frames": source_video_metadata.get(
                        "source_total_frames"
                    ),
                    "source_fps": source_video_metadata.get("source_fps"),
                    "source_duration_seconds": source_video_metadata.get(
                        "source_duration_seconds"
                    ),
                    "sampled_frame_count": sampled_frame_count,
                    "analyzed_frame_count": analyzed_frame_count,
                    "sampled_duration_seconds": sampled_duration_seconds,
                },
                summary={},
                error_message=error_message,
                error_code=error_code,
                processing_time=processing_time,
                created_at=db_result.created_at if db_result else datetime.utcnow(),
            )
        finally:
            # Schedule cleanup
            background_tasks.add_task(self._cleanup_file, video_path)

    async def get_history(
        self,
        skip: int = 0,
        limit: int = 100,
        prediction: Optional[str] = None,
        status: Optional[str] = None,
        model_type: Optional[str] = None,
        search: Optional[str] = None,
        order_by: str = "created_at",
        order_desc: bool = True,
    ) -> DetectionHistoryList:
        """Get detection history with filtering, searching, and pagination"""
        try:
            query = self._apply_detection_result_filters(
                self.db.query(DetectionResult).filter(DetectionResult.del_flag == 0),
                prediction=prediction,
                status=status,
                model_type=model_type,
                search=search,
            )

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

                status_enum = self._resolve_detection_status(
                    getattr(result, "status", None),
                    getattr(result, "error_message", None),
                )
                prediction_enum = self._resolve_prediction_type(
                    getattr(result, "prediction", None)
                )
                raw_confidence = getattr(result, "confidence", None)
                confidence_value = (
                    raw_confidence
                    if prediction_enum is not None
                    and status_enum != DetectionStatus.FAILED
                    else None
                )

                detection = DetectionHistory(
                    id=result.id,
                    file_name=result.file_name,
                    file_path=getattr(result, "file_path", None),
                    file_type=file_type_enum,
                    prediction=prediction_enum,
                    confidence=confidence_value,
                    processing_time=getattr(result, "processing_time", None),
                    source_total_frames=getattr(result, "source_total_frames", None),
                    source_fps=getattr(result, "source_fps", None),
                    source_duration_seconds=getattr(
                        result, "source_duration_seconds", None
                    ),
                    sampled_frame_count=getattr(result, "sampled_frame_count", None),
                    analyzed_frame_count=getattr(result, "analyzed_frame_count", None),
                    sampled_duration_seconds=getattr(
                        result, "sampled_duration_seconds", None
                    ),
                    model_name=result.model_name
                    or (result.model.name if result.model else None)
                    or (
                        "No model loaded"
                        if status_enum == DetectionStatus.FAILED
                        else "Unknown model"
                    ),
                    model_type=result.model_type
                    or (result.model.model_type if result.model else None),
                    status=status_enum,
                    error_message=getattr(result, "error_message", None) or None,
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
        return await self.get_statistics_filtered()

    async def get_statistics_filtered(
        self,
        prediction: Optional[str] = None,
        status: Optional[str] = None,
        model_type: Optional[str] = None,
        search: Optional[str] = None,
    ) -> DetectionStatistics:
        """Get detection statistics"""
        try:
            rows = self._apply_detection_result_filters(
                self.db.query(DetectionResult).filter(DetectionResult.del_flag == 0),
                prediction=prediction,
                status=status,
                model_type=model_type,
                search=search,
            ).all()

            total_detections = len(rows)
            real_detections = 0
            fake_detections = 0
            failed_detections = 0
            detections_by_model: Dict[str, int] = {}
            detections_by_file_type: Dict[str, int] = {}
            detections_by_status: Dict[str, int] = {}
            daily_detections: Dict[str, int] = {}
            confidence_values: List[float] = []
            processing_time_values: List[float] = []
            confidence_ranges = {
                "0.0-0.2": 0,
                "0.2-0.4": 0,
                "0.4-0.6": 0,
                "0.6-0.8": 0,
                "0.8-1.0": 0,
            }

            for result in rows:
                resolved_status = self._resolve_detection_status(
                    getattr(result, "status", None),
                    getattr(result, "error_message", None),
                )
                resolved_prediction = self._resolve_prediction_type(
                    getattr(result, "prediction", None)
                )

                detections_by_status[resolved_status.value] = (
                    detections_by_status.get(resolved_status.value, 0) + 1
                )
                if resolved_status == DetectionStatus.FAILED:
                    failed_detections += 1
                elif resolved_prediction == PredictionType.REAL:
                    real_detections += 1
                elif resolved_prediction == PredictionType.FAKE:
                    fake_detections += 1
                else:
                    failed_detections += 1

                model_key = result.model_type or (
                    result.model.model_type if result.model else None
                )
                if not model_key:
                    model_key = "unknown"
                detections_by_model[model_key] = (
                    detections_by_model.get(model_key, 0) + 1
                )

                file_type_key = getattr(result, "file_type", None) or "unknown"
                detections_by_file_type[file_type_key] = (
                    detections_by_file_type.get(file_type_key, 0) + 1
                )

                processing_time = getattr(result, "processing_time", None)
                if processing_time is not None:
                    processing_time_values.append(float(processing_time))

                confidence = getattr(result, "confidence", None)
                if (
                    resolved_prediction is not None
                    and resolved_status != DetectionStatus.FAILED
                    and confidence is not None
                ):
                    confidence_value = float(confidence)
                    confidence_values.append(confidence_value)
                    if confidence_value < 0.2:
                        confidence_ranges["0.0-0.2"] += 1
                    elif confidence_value < 0.4:
                        confidence_ranges["0.2-0.4"] += 1
                    elif confidence_value < 0.6:
                        confidence_ranges["0.4-0.6"] += 1
                    elif confidence_value < 0.8:
                        confidence_ranges["0.6-0.8"] += 1
                    else:
                        confidence_ranges["0.8-1.0"] += 1

                created_at = getattr(result, "created_at", None)
                if created_at:
                    day_key = created_at.strftime("%Y-%m-%d")
                    daily_detections[day_key] = daily_detections.get(day_key, 0) + 1

            average_confidence = (
                float(np.mean(confidence_values)) if confidence_values else 0.0
            )
            average_processing_time = (
                float(np.mean(processing_time_values))
                if processing_time_values
                else 0.0
            )

            return DetectionStatistics(
                total_detections=total_detections,
                real_detections=real_detections,
                fake_detections=fake_detections,
                failed_detections=failed_detections,
                average_confidence=average_confidence,
                average_processing_time=average_processing_time,
                detections_by_model=detections_by_model,
                detections_by_file_type=detections_by_file_type,
                detections_by_status=detections_by_status,
                confidence_distribution=confidence_ranges,
                daily_detections=daily_detections,
            )

        except Exception as e:
            logger.error("Failed to get detection statistics", error=str(e))
            raise

    async def delete_detection_record(
        self,
        detection_id: int,
        audit_context: Optional[Dict[str, Any]] = None,
        audit_action: str = "delete_detection_result",
    ) -> bool:
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
            self._write_audit_log(
                action=audit_action,
                resource_type="detection_result",
                resource_id=result.id,
                details=self._build_detection_audit_details(
                    status="deleted",
                    file_path=result.file_path,
                    original_file_name=result.file_name,
                    file_type=result.file_type,
                    file_size=result.file_size,
                    model_id=result.model_id,
                    model_name=result.model_name,
                    model_type=result.model_type,
                    prediction=result.prediction,
                    confidence=result.confidence,
                    processing_time=result.processing_time,
                ),
                audit_context=audit_context,
            )

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
        requested_model = None
        requested_model_status = None
        requested_failure_reason = None
        attempted_requested_ids = set()
        fallback_errors = []

        if model_id is not None:
            requested_model = (
                self.db.query(ModelRegistry)
                .filter(ModelRegistry.id == model_id, ModelRegistry.del_flag == 0)
                .first()
            )

            if not requested_model:
                requested_model_status = "missing"
                requested_failure_reason = (
                    f"Requested registry model id={model_id} was not found"
                )
            elif requested_model.status not in READY_MODEL_STATUSES:
                requested_model_status = requested_model.status
                requested_failure_reason = (
                    f"Requested registry model id={model_id} "
                    f"({requested_model.name}) is {requested_model.status}, "
                    "not ready/deployed"
                )
            else:
                requested_model_status = requested_model.status
                attempted_requested_ids.add(requested_model.id)
                try:
                    loaded_requested = dict(
                        self._load_registry_model_record(requested_model)
                    )
                except Exception as exc:
                    requested_failure_reason = (
                        f"Requested registry model id={model_id} "
                        f"({requested_model.name}) could not be loaded: {exc}"
                    )
                    logger.warning(
                        "Requested registry model unavailable for inference",
                        requested_model_id=model_id,
                        requested_model_type=model_type,
                        requested_model_name=requested_model.name,
                        error=str(exc),
                    )
                else:
                    loaded_requested.update(
                        {
                            "requested_model_id": model_id,
                            "requested_model_type": model_type,
                            "requested_model_status": requested_model.status,
                            "readiness": "ready",
                            "selection_policy": "primary",
                        }
                    )
                    return loaded_requested
        elif model_type:
            requested_model_status = "fallback_only"
            requested_failure_reason = (
                f"Requested built-in fallback model_type='{model_type}' cannot run "
                "inference without a ready/deployed registry model"
            )

        ready_registry_models = self._get_ready_registry_models()
        for fallback_model in ready_registry_models:
            if fallback_model.id in attempted_requested_ids:
                continue
            try:
                loaded_fallback = dict(self._load_registry_model_record(fallback_model))
            except Exception as exc:
                fallback_errors.append(
                    f"id={fallback_model.id} ({fallback_model.name}, "
                    f"status={fallback_model.status}) failed to load: {exc}"
                )
                logger.warning(
                    "Ready registry model unavailable for deterministic fallback",
                    fallback_model_id=fallback_model.id,
                    fallback_model_name=fallback_model.name,
                    fallback_model_status=fallback_model.status,
                    error=str(exc),
                )
                continue

            if requested_failure_reason:
                logger.warning(
                    "Detection model request falling back to ready registry default",
                    requested_model_id=model_id,
                    requested_model_type=model_type,
                    requested_model_status=requested_model_status,
                    fallback_model_id=fallback_model.id,
                    fallback_model_name=fallback_model.name,
                    reason=requested_failure_reason,
                )

            loaded_fallback.update(
                {
                    "requested_model_id": model_id,
                    "requested_model_type": model_type,
                    "requested_model_status": requested_model_status,
                    "readiness": "ready",
                    "selection_policy": (
                        "fallback_default" if requested_failure_reason else "primary"
                    ),
                    "fallback_reason": requested_failure_reason,
                }
            )
            return loaded_fallback

        raise ValueError(
            self._build_no_usable_model_error(
                model_id=model_id,
                model_type=model_type,
                requested_failure_reason=requested_failure_reason,
                fallback_errors=fallback_errors,
            )
        )

    def _build_model_info(self, loaded_model: Dict[str, Any]) -> Dict[str, Any]:
        model_info = {
            "model_id": loaded_model.get("model_id"),
            "model_name": loaded_model.get("model_name"),
            "model_type": loaded_model["model_type"],
            "input_size": loaded_model.get("input_size", settings.MODEL_INPUT_SIZE),
            "source": loaded_model.get("source"),
        }
        for key in (
            "status",
            "weight_state",
            "readiness",
            "selection_policy",
            "requested_model_id",
            "requested_model_type",
            "requested_model_status",
            "fallback_reason",
            "face_roi_enabled",
            "face_roi_confidence_threshold",
            "face_roi_crop_padding",
            "face_roi_policy_version",
            "face_roi_selection_policy",
            "video_aggregation_topk_ratio",
            "video_aggregation_mean_weight",
            "video_aggregation_peak_weight",
            "video_aggregation_persistence_weight",
            "temporal_bidirectional",
            "temporal_attention_pooling",
        ):
            if key in loaded_model and loaded_model.get(key) is not None:
                model_info[key] = loaded_model.get(key)
        return model_info

    def _resolve_face_roi_policy(
        self, loaded_model: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        parameters = dict((loaded_model or {}).get("parameters") or {})
        for key in (
            "face_roi_enabled",
            "yolo_face_model_path",
            "face_roi_confidence_threshold",
            "face_roi_crop_padding",
            "face_roi_policy_version",
            "face_roi_selection_policy",
        ):
            if loaded_model and loaded_model.get(key) is not None:
                parameters[key] = loaded_model.get(key)
        return self._face_roi_processor.build_policy(parameters)

    def _resolve_video_aggregation_policy(
        self, loaded_model: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        parameters = dict((loaded_model or {}).get("parameters") or {})

        def get_value(key: str, default: float) -> float:
            loaded_value = (loaded_model or {}).get(key) if loaded_model else None
            if loaded_value is not None:
                return float(loaded_value)
            parameter_value = parameters.get(key)
            if parameter_value is not None:
                return float(parameter_value)
            return float(default)

        return {
            "topk_ratio": get_value(
                "video_aggregation_topk_ratio", settings.VIDEO_AGGREGATION_TOPK_RATIO
            ),
            "mean_weight": get_value(
                "video_aggregation_mean_weight", settings.VIDEO_AGGREGATION_MEAN_WEIGHT
            ),
            "peak_weight": get_value(
                "video_aggregation_peak_weight", settings.VIDEO_AGGREGATION_PEAK_WEIGHT
            ),
            "persistence_weight": get_value(
                "video_aggregation_persistence_weight",
                settings.VIDEO_AGGREGATION_PERSISTENCE_WEIGHT,
            ),
        }

    def _get_ready_registry_models(self) -> List[ModelRegistry]:
        return (
            self.db.query(ModelRegistry)
            .filter(
                ModelRegistry.del_flag == 0,
                ModelRegistry.status.in_(READY_MODEL_STATUSES),
            )
            .order_by(
                ModelRegistry.is_default.desc(),
                ModelRegistry.created_at.desc(),
                ModelRegistry.id.desc(),
            )
            .all()
        )

    def _build_no_usable_model_error(
        self,
        model_id: Optional[int],
        model_type: Optional[str],
        requested_failure_reason: Optional[str],
        fallback_errors: List[str],
    ) -> str:
        request_parts = []
        if model_id is not None:
            request_parts.append(f"model_id={model_id}")
        if model_type:
            request_parts.append(f"model_type='{model_type}'")
        request_context = (
            ", ".join(request_parts) if request_parts else "no explicit model request"
        )

        message_parts = [
            "No usable ready/deployed registry model is available for detection",
            f"({request_context}).",
        ]
        if requested_failure_reason:
            message_parts.append(
                f"Requested model decision: {requested_failure_reason}."
            )
        if fallback_errors:
            message_parts.append(
                "Deterministic registry fallback candidates failed in order: "
                + "; ".join(fallback_errors)
                + "."
            )
        else:
            message_parts.append(
                "No ready/deployed registry fallback candidate could be selected."
            )
        message_parts.append(
            "Built-in fallback inference is disabled until a vetted registry model is ready or deployed."
        )
        return " ".join(message_parts)

    def _load_registry_model_record(
        self, model_record: ModelRegistry
    ) -> Dict[str, Any]:
        cache_key = f"registry_{model_record.id}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        checkpoint = self._load_checkpoint(model_record.file_path)
        model = self._build_model_from_checkpoint(
            model_record.model_type, checkpoint, model_record
        )
        state_dict = checkpoint.get("model_state_dict")
        if not state_dict:
            raise ValueError("Checkpoint does not contain model_state_dict")
        self._load_model_state(model, state_dict)

        loaded = {
            "model": model,
            "model_id": model_record.id,
            "model_name": model_record.name,
            "model_type": model_record.model_type,
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
            "video_temporal_enabled": bool(checkpoint.get("video_temporal_enabled")),
            "training_mode": checkpoint.get("training_mode"),
            "source": "registry",
            "status": model_record.status,
            "weight_state": "checkpoint_loaded",
            "parameters": dict(model_record.parameters or {}),
            "face_roi_enabled": checkpoint.get(
                "face_roi_enabled",
                (model_record.parameters or {}).get("face_roi_enabled"),
            ),
            "yolo_face_model_path": checkpoint.get(
                "yolo_face_model_path",
                (model_record.parameters or {}).get("yolo_face_model_path"),
            ),
            "face_roi_confidence_threshold": checkpoint.get(
                "face_roi_confidence_threshold",
                (model_record.parameters or {}).get("face_roi_confidence_threshold"),
            ),
            "face_roi_crop_padding": checkpoint.get(
                "face_roi_crop_padding",
                (model_record.parameters or {}).get("face_roi_crop_padding"),
            ),
            "face_roi_policy_version": checkpoint.get(
                "face_roi_policy_version",
                (model_record.parameters or {}).get("face_roi_policy_version"),
            ),
            "face_roi_selection_policy": checkpoint.get(
                "face_roi_selection_policy",
                (model_record.parameters or {}).get("face_roi_selection_policy"),
            ),
            "video_aggregation_topk_ratio": checkpoint.get(
                "video_aggregation_topk_ratio",
                (model_record.parameters or {}).get("video_aggregation_topk_ratio"),
            ),
            "video_aggregation_mean_weight": checkpoint.get(
                "video_aggregation_mean_weight",
                (model_record.parameters or {}).get("video_aggregation_mean_weight"),
            ),
            "video_aggregation_peak_weight": checkpoint.get(
                "video_aggregation_peak_weight",
                (model_record.parameters or {}).get("video_aggregation_peak_weight"),
            ),
            "video_aggregation_persistence_weight": checkpoint.get(
                "video_aggregation_persistence_weight",
                (model_record.parameters or {}).get(
                    "video_aggregation_persistence_weight"
                ),
            ),
            "temporal_bidirectional": checkpoint.get(
                "temporal_bidirectional",
                (model_record.parameters or {}).get("temporal_bidirectional"),
            ),
            "temporal_attention_pooling": checkpoint.get(
                "temporal_attention_pooling",
                (model_record.parameters or {}).get("temporal_attention_pooling"),
            ),
        }
        loaded["model"].eval()
        self._model_cache[cache_key] = loaded
        return loaded

    def _preprocess_image(
        self,
        image_path: str,
        input_size: Optional[int] = None,
        face_roi_policy: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Preprocess image for inference"""
        target_size = input_size or settings.MODEL_INPUT_SIZE
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            if face_roi_policy:
                rgb_image, _ = self._face_roi_processor.crop_pil(
                    rgb_image, face_roi_policy
                )
            rgb_image = rgb_image.resize((target_size, target_size))
            image_array = np.array(rgb_image, dtype=np.float32) / 255.0
        return self._normalize_image_array(image_array)

    def _preprocess_frame(
        self,
        frame: np.ndarray,
        input_size: Optional[int] = None,
        face_roi_policy: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Preprocess video frame for inference"""
        target_size = input_size or settings.MODEL_INPUT_SIZE
        if face_roi_policy:
            frame, _ = self._face_roi_processor.crop_frame(frame, face_roi_policy)
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
        input_size: Optional[int] = None,
        clip_indices: Optional[List[int]] = None,
        face_roi_policy: Optional[Dict[str, Any]] = None,
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
            self._preprocess_frame(
                frame,
                input_size=input_size,
                face_roi_policy=face_roi_policy,
            ).squeeze(0)
            for frame in selected_frames
        ]
        return torch.stack(processed_frames, dim=0).unsqueeze(0)

    def _preprocess_image_clip(
        self,
        image_path: str,
        sequence_length: int,
        input_size: Optional[int] = None,
        face_roi_policy: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Repeat a single image into a fixed-length clip tensor."""
        frame_tensor = self._preprocess_image(
            image_path,
            input_size=input_size,
            face_roi_policy=face_roi_policy,
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

    def _get_video_source_metadata(self, video_path: str) -> Dict[str, Optional[float]]:
        cap = cv2.VideoCapture(video_path)

        try:
            if hasattr(cap, "isOpened") and not cap.isOpened():
                return {
                    "source_total_frames": None,
                    "source_fps": None,
                    "source_duration_seconds": None,
                }

            raw_total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            raw_fps = cap.get(cv2.CAP_PROP_FPS)
            source_total_frames = (
                int(raw_total_frames)
                if raw_total_frames and raw_total_frames > 0
                else None
            )
            source_fps = float(raw_fps) if raw_fps and raw_fps > 0 else None
            source_duration_seconds = (
                float(source_total_frames / source_fps)
                if source_total_frames is not None and source_fps is not None
                else None
            )
            return {
                "source_total_frames": source_total_frames,
                "source_fps": source_fps,
                "source_duration_seconds": source_duration_seconds,
            }
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
        face_roi_policy: Optional[Dict[str, Any]] = None,
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
                face_roi_policy=face_roi_policy,
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
                        face_roi_policy=face_roi_policy,
                    )
                else:
                    fallback_tensor = self._preprocess_frame(
                        frame,
                        input_size=input_size,
                        face_roi_policy=face_roi_policy,
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

    def _build_decision_metrics(
        self,
        probabilities: Dict[str, float],
        confidence_threshold: float,
        prediction: str,
        confidence: float,
    ) -> DetectionDecisionMetrics:
        fake_probability = float(probabilities.get("fake", 0.0))
        real_probability = float(probabilities.get("real", 0.0))
        predicted_probability = float(confidence)
        other_probability = (
            real_probability if prediction == "fake" else fake_probability
        )
        return DetectionDecisionMetrics(
            confidence_threshold=confidence_threshold,
            fake_probability=max(0.0, min(1.0, fake_probability)),
            real_probability=max(0.0, min(1.0, real_probability)),
            predicted_probability=max(0.0, min(1.0, predicted_probability)),
            decision_margin=max(
                -1.0,
                min(1.0, predicted_probability - other_probability),
            ),
            threshold_gap=max(-1.0, min(1.0, fake_probability - confidence_threshold)),
            threshold_applied_to_fake=True,
        )

    def _probabilities_to_prediction(
        self,
        probabilities: Dict[str, float],
        confidence_threshold: float = 0.5,
    ) -> tuple:
        fake_probability = max(0.0, min(1.0, float(probabilities.get("fake", 0.0))))
        real_probability = max(0.0, min(1.0, float(probabilities.get("real", 0.0))))
        if fake_probability >= confidence_threshold:
            return "fake", fake_probability
        return "real", real_probability

    def _coerce_audit_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: self._coerce_audit_value(item)
                for key, item in value.items()
                if item is not None
            }
        if isinstance(value, (list, tuple)):
            return [self._coerce_audit_value(item) for item in value]
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        return value

    def _build_detection_audit_details(
        self,
        *,
        status: str,
        file_path: str,
        original_file_name: Optional[str],
        file_type: str,
        file_size: Optional[int],
        model_id: Optional[int],
        model_name: Optional[str],
        model_type: Optional[str],
        prediction: Optional[str] = None,
        confidence: Optional[float] = None,
        processing_time: Optional[float] = None,
        error_message: Optional[str] = None,
        extra_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        details: Dict[str, Any] = {
            "status": status,
            "file_name": original_file_name or os.path.basename(file_path),
            "stored_file_name": os.path.basename(file_path),
            "file_type": file_type,
        }

        if file_size is not None:
            details["file_size"] = int(file_size)
        if model_id is not None:
            details["model_id"] = int(model_id)
        if model_name:
            details["model_name"] = model_name
        if model_type:
            details["model_type"] = model_type
        if prediction:
            details["prediction"] = prediction
        if confidence is not None:
            details["confidence"] = float(confidence)
        if processing_time is not None:
            details["processing_time"] = float(processing_time)
        if error_message:
            details["error_message"] = error_message
        if extra_details:
            details.update(self._coerce_audit_value(extra_details))

        return details

    def _write_audit_log(
        self,
        *,
        action: str,
        resource_type: str,
        resource_id: Optional[Any],
        details: Dict[str, Any],
        audit_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.db is None:
            return

        add = getattr(self.db, "add", None)
        commit = getattr(self.db, "commit", None)
        rollback = getattr(self.db, "rollback", None)
        if not callable(add) or not callable(commit):
            return

        resolved_context = audit_context or {}
        audit_log = AuditLog(
            user_id=resolved_context.get("user_id"),
            action=action,
            resource_type=resource_type,
            resource_id=str(resource_id) if resource_id is not None else None,
            details=self._coerce_audit_value(details),
            ip_address=resolved_context.get("ip_address"),
            user_agent=resolved_context.get("user_agent"),
        )

        try:
            add(audit_log)
            commit()
        except Exception as audit_error:
            if callable(rollback):
                rollback()
            logger.warning(
                "Audit log write failed",
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                error=str(audit_error),
            )

    def _write_detection_audit_log(
        self,
        *,
        action: str,
        status: str,
        db_result: Optional[DetectionResult],
        file_path: str,
        original_file_name: Optional[str],
        file_type: str,
        file_size: Optional[int],
        model_id: Optional[int],
        model_name: Optional[str],
        model_type: Optional[str],
        audit_context: Optional[Dict[str, Any]] = None,
        prediction: Optional[str] = None,
        confidence: Optional[float] = None,
        processing_time: Optional[float] = None,
        error_message: Optional[str] = None,
        extra_details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._write_audit_log(
            action=action,
            resource_type="detection_result",
            resource_id=db_result.id if db_result else None,
            details=self._build_detection_audit_details(
                status=status,
                file_path=file_path,
                original_file_name=original_file_name,
                file_type=file_type,
                file_size=file_size,
                model_id=model_id,
                model_name=model_name,
                model_type=model_type,
                prediction=prediction,
                confidence=confidence,
                processing_time=processing_time,
                error_message=error_message,
                extra_details=extra_details,
            ),
            audit_context=audit_context,
        )

    def _build_frame_results_from_probabilities(
        self,
        frames: List[tuple],
        frame_probabilities: List[Dict[str, float]],
        loaded_model: Dict[str, Any],
        confidence_threshold: float,
    ) -> List[Dict[str, Any]]:
        """Build frame result payloads from a probability timeline."""
        frame_results = []
        for index, ((_, timestamp), probabilities) in enumerate(
            zip(frames, frame_probabilities), start=1
        ):
            prediction, confidence = self._probabilities_to_prediction(
                probabilities, confidence_threshold=confidence_threshold
            )
            frame_results.append(
                {
                    "frame_number": index,
                    "timestamp": timestamp,
                    "result": {
                        "prediction": prediction,
                        "confidence": confidence,
                        "probabilities": probabilities,
                        "decision_metrics": self._build_decision_metrics(
                            probabilities,
                            confidence_threshold,
                            prediction,
                            confidence,
                        ).model_dump(),
                        "processing_time": 0.0,
                        "model_info": {
                            **self._build_model_info(loaded_model),
                        },
                    },
                }
            )
        return frame_results

    def _aggregate_probability_sequence(
        self,
        frame_probabilities: List[Dict[str, float]],
        confidence_threshold: float = 0.5,
        aggregation_policy: Optional[Dict[str, float]] = None,
    ) -> tuple:
        """Aggregate a probability timeline into a video-level prediction."""
        if not frame_probabilities:
            empty_probabilities = {"fake": 0.0, "real": 0.0}
            return (
                "unknown",
                0.0,
                empty_probabilities,
                self._build_decision_metrics(
                    empty_probabilities,
                    confidence_threshold,
                    "real",
                    0.0,
                ),
            )

        aggregation_policy = (
            aggregation_policy or self._resolve_video_aggregation_policy()
        )
        probabilities, aggregation_details = aggregate_probability_sequence(
            frame_probabilities,
            confidence_threshold=confidence_threshold,
            topk_ratio=aggregation_policy["topk_ratio"],
            mean_weight=aggregation_policy["mean_weight"],
            peak_weight=aggregation_policy["peak_weight"],
            persistence_weight=aggregation_policy["persistence_weight"],
        )
        prediction, confidence = self._probabilities_to_prediction(
            probabilities,
            confidence_threshold=confidence_threshold,
        )
        decision_metrics = self._build_decision_metrics(
            probabilities,
            confidence_threshold,
            prediction,
            confidence,
        )
        for key, value in aggregation_details.items():
            setattr(decision_metrics, key, value)
        return prediction, confidence, probabilities, decision_metrics

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
        original_file_name: Optional[str],
        file_type: str,
        result: Optional[DetectionResultSchema],
        file_size: Optional[int],
        status: str,
        model_id: Optional[int] = None,
        error_message: Optional[str] = None,
        processing_time: Optional[float] = None,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        video_metadata: Optional[Dict[str, Any]] = None,
    ) -> DetectionResult:
        video_metadata = video_metadata or {}
        db_result = DetectionResult(
            file_path=file_path,
            file_name=original_file_name or os.path.basename(file_path),
            file_type=file_type,
            prediction=result.prediction if result else "failed",
            confidence=result.confidence if result else 0.0,
            processing_time=(result.processing_time if result else processing_time),
            file_size=file_size,
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            status=status,
            error_message=error_message,
            source_total_frames=video_metadata.get("source_total_frames"),
            source_fps=video_metadata.get("source_fps"),
            source_duration_seconds=video_metadata.get("source_duration_seconds"),
            sampled_frame_count=video_metadata.get("sampled_frame_count"),
            analyzed_frame_count=video_metadata.get("analyzed_frame_count"),
            sampled_duration_seconds=video_metadata.get("sampled_duration_seconds"),
        )

        self.db.add(db_result)
        try:
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        self.db.refresh(db_result)

        return db_result

    async def _persist_failed_detection_result(
        self,
        *,
        file_path: str,
        original_file_name: Optional[str],
        file_type: str,
        file_size: Optional[int],
        processing_time: float,
        error_message: str,
        loaded_model: Optional[Dict[str, Any]] = None,
        video_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[DetectionResult]:
        try:
            return await self._save_detection_result(
                file_path=file_path,
                original_file_name=original_file_name,
                file_type=file_type,
                result=None,
                file_size=file_size,
                status="failed",
                error_message=error_message,
                processing_time=processing_time,
                model_id=(loaded_model or {}).get("model_id"),
                model_name=(loaded_model or {}).get("model_name"),
                model_type=(loaded_model or {}).get("model_type"),
                video_metadata=video_metadata,
            )
        except Exception as persistence_error:
            logger.warning(
                "Failed to persist failed detection result",
                file_path=file_path,
                original_file_name=original_file_name,
                error=str(persistence_error),
                detection_error=error_message,
            )
            return None

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
                "source_total_frames": response.video_info.get("source_total_frames"),
                "source_fps": response.video_info.get("source_fps"),
                "source_duration_seconds": response.video_info.get(
                    "source_duration_seconds"
                ),
                "sampled_frame_count": response.video_info.get("sampled_frame_count"),
                "analyzed_frame_count": response.video_info.get("analyzed_frame_count"),
                "sampled_duration_seconds": response.video_info.get(
                    "sampled_duration_seconds"
                ),
                "total_frames": response.video_info.get("total_frames"),
                "processed_frames": response.video_info.get("processed_frames"),
                "duration": response.video_info.get("duration"),
            },
            result=detection_result,
            error_message=response.error_message if not response.success else None,
            error_code=response.error_code if not response.success else None,
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
                    "temporal_bidirectional": checkpoint.get(
                        "temporal_bidirectional",
                        parameters.get(
                            "temporal_bidirectional",
                            settings.TEMPORAL_BIDIRECTIONAL,
                        ),
                    ),
                    "temporal_attention_pooling": checkpoint.get(
                        "temporal_attention_pooling",
                        parameters.get(
                            "temporal_attention_pooling",
                            settings.TEMPORAL_ATTENTION_POOLING,
                        ),
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
                        "hidden_size", parameters.get("hidden_size", 256)
                    ),
                    "num_layers": checkpoint.get(
                        "num_layers", parameters.get("num_layers", 1)
                    ),
                    "frame_projection_size": checkpoint.get(
                        "frame_projection_size",
                        parameters.get("frame_projection_size", 128),
                    ),
                }
            )
        kwargs["pretrained"] = False
        return create_model(model_type, **kwargs)

    def _load_model_state(self, model, state_dict: Dict[str, Any]) -> None:
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            error_message = str(exc)
            if (
                "Missing key(s) in state_dict" in error_message
                or "Unexpected key(s) in state_dict" in error_message
            ):
                raise ValueError(
                    "Checkpoint is not fully compatible with the inference model architecture; partial weight loading is rejected to avoid random-init inference. "
                    + error_message
                ) from exc
            raise
