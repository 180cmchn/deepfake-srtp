"""
Detection API routes for deepfake detection platform
"""

from fastapi import (
    APIRouter,
    Depends,
    UploadFile,
    File,
    Form,
    HTTPException,
    BackgroundTasks,
    Header,
    Query,
)
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import uuid
from datetime import datetime

from app.core.config import settings
from app.core.database import get_db
from app.core.logging import logger
from app.core.auth import get_current_user, get_optional_user
from app.models.database_models import ModelRegistry
from app.schemas.models import ModelStatus
from app.schemas.detection import (
    DetectionRequest,
    DetectionResponse,
    BatchDetectionRequest,
    BatchDetectionResponse,
    VideoDetectionResponse,
    VideoDetectionRequest,
    DetectionHistory,
    DetectionHistoryList,
    DetectionStatistics,
)

router = APIRouter()


def parse_detection_request(
    model_id: Optional[int] = Form(None),
    model_type: Optional[str] = Form(None),
    confidence_threshold: float = Form(0.5),
    return_probabilities: bool = Form(False),
    preprocess: bool = Form(True),
) -> DetectionRequest:
    return DetectionRequest(
        model_id=model_id,
        model_type=model_type,
        confidence_threshold=confidence_threshold,
        return_probabilities=return_probabilities,
        preprocess=preprocess,
    )


def parse_batch_detection_request(
    model_id: Optional[int] = Form(None),
    model_type: Optional[str] = Form(None),
    confidence_threshold: float = Form(0.5),
    return_probabilities: bool = Form(False),
    preprocess: bool = Form(True),
    parallel_processing: bool = Form(True),
    max_workers: int = Form(4),
) -> BatchDetectionRequest:
    return BatchDetectionRequest(
        model_id=model_id,
        model_type=model_type,
        confidence_threshold=confidence_threshold,
        return_probabilities=return_probabilities,
        preprocess=preprocess,
        parallel_processing=parallel_processing,
        max_workers=max_workers,
    )


def parse_video_detection_request(
    model_id: Optional[int] = Form(None),
    model_type: Optional[str] = Form(None),
    confidence_threshold: float = Form(0.5),
    frame_extraction_interval: int = Form(4),
    max_frames: int = Form(20),
    aggregate_results: bool = Form(True),
    return_frame_results: bool = Form(False),
    preprocess: bool = Form(True),
) -> VideoDetectionRequest:
    return VideoDetectionRequest(
        model_id=model_id,
        model_type=model_type,
        confidence_threshold=confidence_threshold,
        frame_extraction_interval=frame_extraction_interval,
        max_frames=max_frames,
        aggregate_results=aggregate_results,
        return_frame_results=return_frame_results,
        preprocess=preprocess,
    )


def _get_model_usable_for(model_type: str) -> List[str]:
    return ["video"] if model_type == "lrcn" else ["image", "video"]


def _build_registry_model_option(model: ModelRegistry):
    return {
        "id": model.id,
        "name": model.name,
        "label": f"{model.name} ({model.model_type.upper()})",
        "model_type": model.model_type,
        "source": "registry",
        "status": model.status,
        "is_default": bool(model.is_default),
        "usable_for": _get_model_usable_for(model.model_type),
        "is_ready": True,
        "is_recommended": True,
        "readiness": "ready",
        "selection_policy": "primary",
        "readiness_reason": (
            f"Registry model is {model.status} and available for detection"
        ),
    }


def _build_builtin_model_option(model_type: str, *, is_fallback_default: bool):
    return {
        "id": None,
        "name": model_type,
        "label": f"Built-in fallback {model_type.upper()}",
        "model_type": model_type,
        "source": "builtin",
        "status": "builtin",
        "is_default": False,
        "usable_for": _get_model_usable_for(model_type),
        "is_ready": False,
        "is_recommended": False,
        "readiness": "fallback_only",
        "selection_policy": (
            "fallback_default" if is_fallback_default else "fallback_only"
        ),
        "readiness_reason": (
            "Built-in model types are exposed as fallback-only until a ready "
            "or deployed registry model exists"
        ),
    }


@router.post("/detect", response_model=DetectionResponse)
async def detect_deepfake(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    request: DetectionRequest = Depends(parse_detection_request),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    """
    Detect deepfake in uploaded file
    """
    file_path = None
    try:
        from app.services.detection_service import DetectionService

        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1].lower()
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join("uploads", unique_filename)

        # Save uploaded file
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Initialize detection service
        detection_service = DetectionService(db)

        # Route video uploads through the dedicated video pipeline while
        # preserving the generic /detect response shape for the frontend.
        if file_extension in [".mp4", ".avi", ".mov", ".mkv", ".wmv"]:
            video_request = VideoDetectionRequest(
                video_path=file_path,
                model_id=request.model_id,
                model_type=request.model_type,
                confidence_threshold=request.confidence_threshold,
                preprocess=request.preprocess,
            )
            video_result = await detection_service.detect_video(
                video_path=file_path,
                request=video_request,
                background_tasks=background_tasks,
            )

            detection_result = video_result.aggregated_result
            if not detection_result and video_result.frame_results:
                first_frame = video_result.frame_results[0]
                detection_result = first_frame.result if first_frame else None

            file_info = {
                "name": file.filename,
                "type": "video",
                "size": os.path.getsize(file_path),
                "resolution": None,
                "total_frames": video_result.video_info.get("total_frames"),
                "processed_frames": video_result.video_info.get("processed_frames"),
                "duration": video_result.video_info.get("duration"),
            }

            return DetectionResponse(
                success=video_result.success,
                file_info=file_info,
                result=detection_result,
                error_message=None
                if video_result.success
                else "Video detection failed",
                processing_time=video_result.processing_time,
                created_at=video_result.created_at,
            )

        # Perform detection
        result = await detection_service.detect_file(
            file_path=file_path, request=request, background_tasks=background_tasks
        )

        if result.success and result.result:
            logger.info(
                "Detection completed",
                file_name=file.filename,
                result=result.result.prediction,
                confidence=result.result.confidence,
                user=current_user,
            )
        else:
            logger.info(
                "Detection completed",
                file_name=file.filename,
                success=result.success,
                error_message=result.error_message,
                user=current_user,
            )

        return result

    except HTTPException:
        # Clean up uploaded file if it exists
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info("Cleaned up uploaded file", file_path=file_path)
            except Exception as cleanup_error:
                logger.error(
                    "Failed to clean up file",
                    file_path=file_path,
                    error=str(cleanup_error),
                )
        raise
    except Exception as e:
        logger.error(
            "Detection failed", error=str(e), file_name=file.filename, user=current_user
        )

        # Clean up uploaded file if it exists
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info("Cleaned up uploaded file after error", file_path=file_path)
            except Exception as cleanup_error:
                logger.error(
                    "Failed to clean up file after error",
                    file_path=file_path,
                    error=str(cleanup_error),
                )

        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.post("/detect/batch", response_model=BatchDetectionResponse)
async def detect_deepfake_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    request: BatchDetectionRequest = Depends(parse_batch_detection_request),
    db: Session = Depends(get_db),
):
    """
    Detect deepfake in multiple files
    """
    try:
        from app.services.detection_service import DetectionService

        if len(files) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 files allowed")

        # Save uploaded files
        file_paths = []
        for file in files:
            if not file.filename:
                continue

            file_extension = os.path.splitext(file.filename)[1].lower()
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = os.path.join("uploads", unique_filename)

            os.makedirs("uploads", exist_ok=True)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            file_paths.append(file_path)

        # Initialize detection service
        detection_service = DetectionService(db)

        # Perform batch detection
        result = await detection_service.detect_batch(
            file_paths=file_paths, request=request, background_tasks=background_tasks
        )

        logger.info(
            "Batch detection completed",
            total_files=len(files),
            processed_files=result.processed_files,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Batch detection failed", error=str(e))
        raise HTTPException(status_code=500, detail="Batch detection failed")


@router.post("/detect/video", response_model=VideoDetectionResponse)
async def detect_deepfake_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    request: VideoDetectionRequest = Depends(parse_video_detection_request),
    db: Session = Depends(get_db),
):
    """
    Detect deepfake in video file
    """
    try:
        from app.services.detection_service import DetectionService

        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Validate video file
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in [".mp4", ".avi", ".mov", ".mkv", ".wmv"]:
            raise HTTPException(status_code=400, detail="Invalid video format")

        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join("uploads", unique_filename)

        # Save uploaded file
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Initialize detection service
        detection_service = DetectionService(db)

        # Perform video detection
        result = await detection_service.detect_video(
            video_path=file_path, request=request, background_tasks=background_tasks
        )

        logger.info(
            "Video detection completed",
            file_name=file.filename,
            frames_analyzed=len(result.frame_results) if result.frame_results else 0,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Video detection failed", error=str(e), file_name=file.filename)
        raise HTTPException(status_code=500, detail="Video detection failed")


@router.get("/history", response_model=DetectionHistoryList)
async def get_detection_history(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(
        100, ge=1, le=1000, description="Maximum number of records to return"
    ),
    prediction: Optional[str] = Query(None, description="Filter by prediction result"),
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    search: Optional[str] = Query(None, description="Search in filename"),
    order_by: str = Query("created_at", description="Field to order by"),
    order_desc: bool = Query(True, description="Order in descending order"),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    """
    Get detection history with filtering, searching, and pagination
    """
    try:
        from app.services.detection_service import DetectionService

        detection_service = DetectionService(db)
        history = await detection_service.get_history(
            skip=skip,
            limit=limit,
            prediction=prediction,
            model_type=model_type,
            user_id=user_id,
            search=search,
            order_by=order_by,
            order_desc=order_desc,
        )
        return history

    except Exception as e:
        logger.error("Failed to get detection history", error=str(e), user=current_user)
        raise HTTPException(status_code=500, detail="Failed to get detection history")


@router.get("/statistics", response_model=DetectionStatistics)
async def get_detection_statistics(db: Session = Depends(get_db)):
    """
    Get detection statistics
    """
    try:
        from app.services.detection_service import DetectionService

        detection_service = DetectionService(db)
        stats = await detection_service.get_statistics()
        return stats

    except Exception as e:
        logger.error("Failed to get detection statistics", error=str(e))
        raise HTTPException(
            status_code=500, detail="Failed to get detection statistics"
        )


@router.get("/models")
async def get_available_models(db: Session = Depends(get_db)):
    """
    Get available detection models
    """
    try:
        from app.models.ml_models import ModelRegistry as MLModelRegistry

        registry_models = (
            db.query(ModelRegistry)
            .filter(
                ModelRegistry.del_flag == 0,
                ModelRegistry.status.in_(
                    [ModelStatus.READY.value, ModelStatus.DEPLOYED.value]
                ),
            )
            .order_by(
                ModelRegistry.is_default.desc(),
                ModelRegistry.created_at.desc(),
                ModelRegistry.id.desc(),
            )
            .all()
        )
        builtin_types = sorted(set(MLModelRegistry.list_models()))
        default_builtin_type = (
            settings.DEFAULT_MODEL_TYPE
            if settings.DEFAULT_MODEL_TYPE in builtin_types
            else (builtin_types[0] if builtin_types else settings.DEFAULT_MODEL_TYPE)
        )
        builtin_types = sorted(
            builtin_types,
            key=lambda model_type: (model_type != default_builtin_type, model_type),
        )

        saved_models = [
            _build_registry_model_option(model) for model in registry_models
        ]

        saved_model_types = {model.model_type for model in registry_models}
        builtin_models = [
            _build_builtin_model_option(
                model_type,
                is_fallback_default=model_type == default_builtin_type,
            )
            for model_type in builtin_types
            if model_type not in saved_model_types
        ]

        default_model = (
            next((model for model in saved_models if model["is_default"]), None)
            or (saved_models[0] if saved_models else None)
            or (
                next(
                    (
                        model
                        for model in builtin_models
                        if model["model_type"] == default_builtin_type
                    ),
                    None,
                )
                or (builtin_models[0] if builtin_models else None)
            )
        )

        default_value = {
            "model_id": default_model["id"] if default_model else None,
            "model_type": (
                default_model["model_type"] if default_model else default_builtin_type
            ),
            "source": default_model["source"] if default_model else "builtin",
            "is_ready": default_model["is_ready"] if default_model else False,
            "is_recommended": (
                default_model["is_recommended"] if default_model else False
            ),
            "readiness": (
                default_model["readiness"] if default_model else "fallback_only"
            ),
            "selection_policy": (
                default_model["selection_policy"]
                if default_model
                else "fallback_default"
            ),
        }

        return {
            "models": saved_models + builtin_models,
            "default": default_value,
            "model_types": builtin_types,
        }

    except Exception as e:
        logger.error("Failed to get available models", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get available models")


@router.delete("/history/{detection_id}")
async def delete_detection_record(detection_id: int, db: Session = Depends(get_db)):
    """
    Delete detection record
    """
    try:
        from app.services.detection_service import DetectionService

        detection_service = DetectionService(db)
        success = await detection_service.delete_detection_record(detection_id)

        if not success:
            raise HTTPException(status_code=404, detail="Detection record not found")

        return {"message": "Detection record deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to delete detection record", error=str(e), detection_id=detection_id
        )
        raise HTTPException(status_code=500, detail="Failed to delete detection record")
