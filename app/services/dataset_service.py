"""
Dataset management service for deepfake detection platform
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.core.logging import logger
from app.core.config import settings
from app.models.database_models import DatasetInfo, DatasetFile
from app.schemas.datasets import (
    DatasetCreate, DatasetResponse, DatasetList, DatasetUpdate,
    DatasetStats, DatasetProcessingConfig, DatasetFileAddRequest,
    DatasetFileAddResponse, DatasetFileInfo
)


class DatasetService:
    """Service for managing datasets and data processing"""

    SUPPORTED_IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    SUPPORTED_VIDEO_TYPES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    
    def __init__(self, db: Session):
        self.db = db
    
    async def create_dataset(self, dataset: DatasetCreate) -> DatasetResponse:
        """Create a new dataset registry entry"""
        try:
            # Check if dataset name already exists
            existing = self.db.query(DatasetInfo).filter(
                DatasetInfo.name == dataset.name,
                DatasetInfo.del_flag == 0
            ).first()
            
            if existing:
                raise ValueError(f"Dataset with name '{dataset.name}' already exists")
            
            # Create dataset entry
            db_dataset = DatasetInfo(
                name=dataset.name,
                description=dataset.description,
                path=dataset.path,
                image_size=dataset.image_size,
                frame_extraction_interval=dataset.frame_extraction_interval,
                max_frames_per_video=dataset.max_frames_per_video
            )
            
            self.db.add(db_dataset)
            self.db.commit()
            self.db.refresh(db_dataset)
            
            logger.info("Dataset created", dataset_name=dataset.name, path=dataset.path)
            
            return self._db_to_response(db_dataset)
            
        except Exception as e:
            logger.error("Failed to create dataset", error=str(e))
            self.db.rollback()
            raise
    
    async def get_datasets(
        self,
        skip: int = 0,
        limit: int = 100,
        is_processed: Optional[bool] = None
    ) -> DatasetList:
        """Get datasets list"""
        try:
            query = self.db.query(DatasetInfo).filter(DatasetInfo.del_flag == 0)
            
            if is_processed is not None:
                query = query.filter(DatasetInfo.is_processed == is_processed)
            
            total = query.count()
            results = query.order_by(DatasetInfo.created_at.desc()).offset(skip).limit(limit).all()
            
            datasets = [self._db_to_response(dataset) for dataset in results]
            
            return DatasetList(
                datasets=datasets,
                total=total,
                page=skip // limit + 1,
                size=limit,
                pages=(total + limit - 1) // limit
            )
            
        except Exception as e:
            logger.error("Failed to get datasets", error=str(e))
            raise
    
    async def get_dataset(self, dataset_id: int) -> Optional[DatasetResponse]:
        """Get dataset by ID"""
        try:
            dataset = self.db.query(DatasetInfo).filter(
                DatasetInfo.id == dataset_id,
                DatasetInfo.del_flag == 0
            ).first()
            
            if not dataset:
                return None
            
            return self._db_to_response(dataset)
            
        except Exception as e:
            logger.error("Failed to get dataset", error=str(e), dataset_id=dataset_id)
            raise
    
    async def update_dataset(self, dataset_id: int, dataset_update: DatasetUpdate) -> Optional[DatasetResponse]:
        """Update dataset"""
        try:
            dataset = self.db.query(DatasetInfo).filter(
                DatasetInfo.id == dataset_id,
                DatasetInfo.del_flag == 0
            ).first()
            
            if not dataset:
                return None
            
            # Update fields
            update_data = dataset_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                if hasattr(dataset, field):
                    setattr(dataset, field, value)
            
            self.db.commit()
            self.db.refresh(dataset)
            
            logger.info("Dataset updated", dataset_id=dataset_id)
            
            return self._db_to_response(dataset)
            
        except Exception as e:
            logger.error("Failed to update dataset", error=str(e), dataset_id=dataset_id)
            self.db.rollback()
            raise
    
    async def delete_dataset(self, dataset_id: int) -> bool:
        """Delete dataset"""
        try:
            dataset = self.db.query(DatasetInfo).filter(
                DatasetInfo.id == dataset_id,
                DatasetInfo.del_flag == 0
            ).first()
            
            if not dataset:
                return False
            
            dataset.del_flag = 1
            self.db.commit()
            
            logger.info("Dataset deleted", dataset_id=dataset_id)
            return True
            
        except Exception as e:
            logger.error("Failed to delete dataset", error=str(e), dataset_id=dataset_id)
            self.db.rollback()
            raise
    
    async def process_dataset(self, dataset_id: int, config: Dict[str, Any]):
        """Run complete feature engineering pipeline in background."""
        try:
            dataset = self.db.query(DatasetInfo).filter(
                DatasetInfo.id == dataset_id,
                DatasetInfo.del_flag == 0
            ).first()
            
            if not dataset:
                logger.error("Dataset not found for processing", dataset_id=dataset_id)
                return
            
            # Update status
            dataset.processing_status = "processing"
            dataset.error_message = None
            dataset.is_processed = False
            self.db.commit()

            processing_config = self._normalize_processing_config(config, dataset)
            file_entries = self._collect_dataset_files(dataset)

            if not file_entries:
                raise ValueError("No supported image/video files found for dataset processing")

            processed_files: List[Dict[str, Any]] = []
            failed_files: List[Dict[str, Any]] = []

            for db_file, file_path, file_type in file_entries:
                try:
                    features = self._extract_features(file_path, file_type, processing_config)
                    label = self._infer_label(file_path)

                    processed_files.append(
                        {
                            "filename": os.path.basename(file_path),
                            "file_path": file_path,
                            "file_type": file_type,
                            "label": label,
                            "features": features,
                        }
                    )

                    db_file.is_processed = True
                    db_file.processing_status = "completed"
                    db_file.error_message = None
                except Exception as file_error:
                    failed_files.append(
                        {
                            "filename": db_file.filename,
                            "file_path": file_path,
                            "file_type": file_type,
                            "error": str(file_error),
                        }
                    )
                    db_file.is_processed = False
                    db_file.processing_status = "failed"
                    db_file.error_message = str(file_error)

            self.db.commit()

            if not processed_files:
                raise ValueError("All files failed during feature engineering")

            summary = self._build_summary(processed_files, failed_files, processing_config)
            feature_artifact_path = self._save_feature_artifact(dataset_id, summary, processed_files, failed_files)

            dataset.total_samples = summary["total_samples"]
            dataset.real_samples = summary["real_samples"]
            dataset.fake_samples = summary["fake_samples"]
            dataset.train_samples = summary["train_samples"]
            dataset.val_samples = summary["val_samples"]
            dataset.test_samples = summary["test_samples"]
            dataset.is_processed = True
            dataset.processing_status = "completed"
            dataset.error_message = None
            self.db.commit()

            logger.info(
                "Dataset feature engineering completed",
                dataset_id=dataset_id,
                processed_files=len(processed_files),
                failed_files=len(failed_files),
                artifact_path=feature_artifact_path,
            )
            
        except Exception as e:
            logger.error("Dataset processing failed", error=str(e), dataset_id=dataset_id)
            
            # Update status to failed
            dataset = self.db.query(DatasetInfo).filter(DatasetInfo.id == dataset_id).first()
            if dataset:
                dataset.processing_status = "failed"
                dataset.error_message = str(e)
                dataset.is_processed = False
                self.db.commit()
    
    async def get_processing_status(self, dataset_id: int) -> Optional[Dict[str, Any]]:
        """Get dataset processing status"""
        try:
            dataset = self.db.query(DatasetInfo).filter(
                DatasetInfo.id == dataset_id,
                DatasetInfo.del_flag == 0
            ).first()
            
            if not dataset:
                return None
            
            return {
                "dataset_id": dataset_id,
                "processing_status": dataset.processing_status,
                "is_processed": dataset.is_processed,
                "error_message": dataset.error_message,
                "stats": self._get_dataset_stats(dataset) if dataset.is_processed else None,
                "feature_artifact": self._get_feature_artifact_path(dataset_id)
            }
            
        except Exception as e:
            logger.error("Failed to get dataset status", error=str(e), dataset_id=dataset_id)
            raise
    
    def _normalize_processing_config(self, config: Dict[str, Any], dataset: DatasetInfo) -> Dict[str, Any]:
        """Normalize processing config with dataset defaults."""
        normalized = dict(config or {})
        normalized.setdefault("image_size", dataset.image_size or settings.MODEL_INPUT_SIZE)
        normalized.setdefault("frame_extraction_interval", dataset.frame_extraction_interval or settings.FRAME_EXTRACTION_INTERVAL)
        normalized.setdefault("max_frames_per_video", dataset.max_frames_per_video or settings.MAX_FRAMES_PER_VIDEO)
        normalized.setdefault("validation_split", 0.2)
        normalized.setdefault("test_split", 0.1)
        normalized.setdefault("min_face_size", 50)
        normalized.setdefault("max_face_size", 500)
        return normalized

    def _collect_dataset_files(self, dataset: DatasetInfo) -> List[Tuple[DatasetFile, str, str]]:
        """Collect or register all processable files for a dataset."""
        existing_files = self.db.query(DatasetFile).filter(
            DatasetFile.dataset_id == dataset.id,
            DatasetFile.del_flag == 0
        ).all()

        entries: List[Tuple[DatasetFile, str, str]] = []

        if existing_files:
            for db_file in existing_files:
                if not os.path.exists(db_file.file_path):
                    db_file.processing_status = "failed"
                    db_file.error_message = "File does not exist"
                    continue
                file_type = self._detect_file_type(db_file.file_path)
                if file_type is None:
                    db_file.processing_status = "failed"
                    db_file.error_message = "Unsupported file type"
                    continue
                entries.append((db_file, db_file.file_path, file_type))
            self.db.commit()
            return entries

        source_paths: List[str] = []
        dataset_path = Path(dataset.path)

        if dataset_path.is_file():
            source_paths = [str(dataset_path)]
        elif dataset_path.is_dir():
            for root, _, files in os.walk(dataset_path):
                for filename in files:
                    source_paths.append(str(Path(root) / filename))
        else:
            return []

        for path in source_paths:
            file_type = self._detect_file_type(path)
            if file_type is None:
                continue

            db_file = DatasetFile(
                dataset_id=dataset.id,
                filename=os.path.basename(path),
                file_path=path,
                file_type=file_type,
                file_size=os.path.getsize(path) if os.path.exists(path) else 0,
                description="Auto-registered during dataset processing",
            )
            self.db.add(db_file)
            self.db.flush()
            entries.append((db_file, path, file_type))

        self.db.commit()
        return entries

    def _detect_file_type(self, file_path: str) -> Optional[str]:
        """Return normalized file type (image/video) if supported."""
        suffix = Path(file_path).suffix.lower()
        if suffix in self.SUPPORTED_IMAGE_TYPES:
            return "image"
        if suffix in self.SUPPORTED_VIDEO_TYPES:
            return "video"
        return None

    def _extract_features(self, file_path: str, file_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract engineered features from an image or video file."""
        if file_type == "image":
            return self._extract_image_features(file_path, config)
        if file_type == "video":
            return self._extract_video_features(file_path, config)
        raise ValueError(f"Unsupported file type: {file_type}")

    def _extract_image_features(self, file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract spatial and quality features from image."""
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError("Cannot read image")

        resized = cv2.resize(image, (int(config["image_size"]), int(config["image_size"])))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        features = {
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "channels": int(image.shape[2]) if len(image.shape) == 3 else 1,
            "mean_b": float(np.mean(resized[:, :, 0])),
            "mean_g": float(np.mean(resized[:, :, 1])),
            "mean_r": float(np.mean(resized[:, :, 2])),
            "std_b": float(np.std(resized[:, :, 0])),
            "std_g": float(np.std(resized[:, :, 1])),
            "std_r": float(np.std(resized[:, :, 2])),
            "brightness_mean": float(np.mean(gray)),
            "brightness_std": float(np.std(gray)),
            "sharpness_laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "edge_density": float(np.mean(cv2.Canny(gray, 100, 200) > 0)),
            "entropy": self._compute_entropy(gray),
            "face_count": self._detect_faces(gray, config),
        }
        return features

    def _extract_video_features(self, file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temporal + frame-level aggregate features from video."""
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = float(frame_count / fps) if fps > 0 else 0.0

        interval = max(1, int(config["frame_extraction_interval"]))
        max_frames = max(1, int(config["max_frames_per_video"]))

        sampled_features: List[Dict[str, float]] = []
        prev_gray = None
        motion_values: List[float] = []

        current_index = 0
        sampled = 0

        try:
            while sampled < max_frames:
                success, frame = cap.read()
                if not success:
                    break

                if current_index % interval == 0:
                    resized = cv2.resize(frame, (int(config["image_size"]), int(config["image_size"])))
                    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                    sampled_features.append(
                        {
                            "brightness_mean": float(np.mean(gray)),
                            "brightness_std": float(np.std(gray)),
                            "sharpness_laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                            "edge_density": float(np.mean(cv2.Canny(gray, 100, 200) > 0)),
                            "entropy": self._compute_entropy(gray),
                        }
                    )

                    if prev_gray is not None:
                        motion = np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))) / 255.0
                        motion_values.append(float(motion))
                    prev_gray = gray
                    sampled += 1

                current_index += 1
        finally:
            cap.release()

        if not sampled_features:
            raise ValueError("No frames sampled from video")

        aggregated = self._aggregate_numeric_features(sampled_features)

        features = {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": duration,
            "sampled_frames": len(sampled_features),
            "mean_motion": float(np.mean(motion_values)) if motion_values else 0.0,
        }
        features.update({f"frame_{k}": v for k, v in aggregated.items()})
        return features

    def _aggregate_numeric_features(self, rows: List[Dict[str, float]]) -> Dict[str, float]:
        """Average numeric features across rows."""
        if not rows:
            return {}

        keys = rows[0].keys()
        aggregated: Dict[str, float] = {}
        for key in keys:
            values = [float(row[key]) for row in rows if key in row]
            aggregated[key] = float(np.mean(values)) if values else 0.0
        return aggregated

    def _compute_entropy(self, gray_image: np.ndarray) -> float:
        """Compute grayscale entropy."""
        histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).ravel().astype(np.float64)
        total = histogram.sum()
        if total <= 0:
            return 0.0
        probabilities = histogram / total
        entropy = -float(np.sum(probabilities * np.log2(probabilities + 1e-12)))
        return entropy

    def _detect_faces(self, gray_image: np.ndarray, config: Dict[str, Any]) -> int:
        """Detect approximate face count using Haar cascade."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        classifier = cv2.CascadeClassifier(cascade_path)
        if classifier.empty():
            return 0

        min_face_size = max(10, int(config.get("min_face_size", 50)))
        max_face_size = max(min_face_size, int(config.get("max_face_size", 500)))

        faces = classifier.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_face_size, min_face_size),
            maxSize=(max_face_size, max_face_size),
        )
        return int(len(faces))

    def _infer_label(self, file_path: str) -> Optional[int]:
        """Infer label from file path keywords: fake=0, real=1."""
        normalized = str(file_path).lower()
        fake_keywords = ("fake", "deepfake", "manipulated", "forged", "tampered", "class1", "\\1\\", "/1/")
        real_keywords = ("real", "authentic", "original", "genuine", "pristine", "class0", "\\0\\", "/0/")

        is_fake = any(token in normalized for token in fake_keywords)
        is_real = any(token in normalized for token in real_keywords)

        if is_fake and not is_real:
            return 0
        if is_real and not is_fake:
            return 1
        return None

    def _build_summary(
        self,
        processed_files: List[Dict[str, Any]],
        failed_files: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build dataset summary and split counts."""
        total_samples = len(processed_files)
        real_samples = sum(1 for item in processed_files if item["label"] == 1)
        fake_samples = sum(1 for item in processed_files if item["label"] == 0)

        val_ratio = float(config.get("validation_split", 0.2))
        test_ratio = float(config.get("test_split", 0.1))
        train_samples, val_samples, test_samples = self._compute_split_counts(total_samples, val_ratio, test_ratio)

        average_sharpness = np.mean([
            item["features"].get("sharpness_laplacian_var", item["features"].get("frame_sharpness_laplacian_var", 0.0))
            for item in processed_files
        ])
        success_rate = total_samples / (total_samples + len(failed_files)) if (total_samples + len(failed_files)) > 0 else 0.0
        quality_score = min(1.0, (0.7 * success_rate) + (0.3 * min(1.0, float(average_sharpness) / 300.0)))

        return {
            "total_samples": total_samples,
            "real_samples": real_samples,
            "fake_samples": fake_samples,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "test_samples": test_samples,
            "success_rate": round(success_rate, 4),
            "failed_files": len(failed_files),
            "data_quality_score": round(float(quality_score), 4),
        }

    def _compute_split_counts(self, total: int, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
        """Compute train/val/test counts with safe bounds."""
        if total <= 0:
            return 0, 0, 0

        val_ratio = max(0.0, min(0.5, val_ratio))
        test_ratio = max(0.0, min(0.4, test_ratio))

        val_count = int(math.floor(total * val_ratio))
        test_count = int(math.floor(total * test_ratio))

        if val_count + test_count >= total:
            overflow = (val_count + test_count) - (total - 1)
            test_count = max(0, test_count - overflow)

        train_count = total - val_count - test_count
        return train_count, val_count, test_count

    def _save_feature_artifact(
        self,
        dataset_id: int,
        summary: Dict[str, Any],
        processed_files: List[Dict[str, Any]],
        failed_files: List[Dict[str, Any]],
    ) -> str:
        """Persist extracted features to an artifact JSON file."""
        artifact_path = self._get_feature_artifact_path(dataset_id)
        artifact_dir = Path(artifact_path).parent
        artifact_dir.mkdir(parents=True, exist_ok=True)

        artifact_payload = {
            "dataset_id": dataset_id,
            "generated_at": datetime.utcnow().isoformat(),
            "summary": summary,
            "processed_files": processed_files,
            "failed_files": failed_files,
        }

        with open(artifact_path, "w", encoding="utf-8") as artifact_file:
            json.dump(artifact_payload, artifact_file, ensure_ascii=True, indent=2)

        return artifact_path

    def _get_feature_artifact_path(self, dataset_id: int) -> str:
        """Return deterministic artifact file path for dataset features."""
        return str(Path(settings.DATA_DIR) / "features" / f"dataset_{dataset_id}_features.json")
    
    def _get_dataset_stats(self, dataset: DatasetInfo) -> DatasetStats:
        """Get dataset statistics"""
        quality_score = None
        artifact_path = self._get_feature_artifact_path(dataset.id)
        if os.path.exists(artifact_path):
            try:
                with open(artifact_path, "r", encoding="utf-8") as artifact_file:
                    artifact_data = json.load(artifact_file)
                quality_score = artifact_data.get("summary", {}).get("data_quality_score")
            except Exception:
                quality_score = None

        return DatasetStats(
            total_samples=dataset.total_samples,
            real_samples=dataset.real_samples,
            fake_samples=dataset.fake_samples,
            train_samples=dataset.train_samples,
            val_samples=dataset.val_samples,
            test_samples=dataset.test_samples,
            class_distribution={
                "real": dataset.real_samples or 0,
                "fake": dataset.fake_samples or 0
            } if dataset.real_samples and dataset.fake_samples else None,
            data_quality_score=quality_score
        )
    
    async def add_files_to_dataset(
        self, 
        dataset_id: int, 
        files: List[tuple], 
        request: DatasetFileAddRequest
    ) -> DatasetFileAddResponse:
        """Add files to existing dataset"""
        try:
            # Check if dataset exists
            dataset = self.db.query(DatasetInfo).filter(
                DatasetInfo.id == dataset_id,
                DatasetInfo.del_flag == 0
            ).first()
            
            if not dataset:
                raise ValueError(f"Dataset with ID {dataset_id} not found")
            
            added_files = []
            file_paths = []
            
            # Process each file
            for file_info in files:
                file_obj, file_path, file_type, file_size = file_info
                
                # Create file record
                db_file = DatasetFile(
                    dataset_id=dataset_id,
                    filename=file_obj.filename,
                    file_path=file_path,
                    file_type=file_type,
                    file_size=file_size,
                    description=request.description
                )
                
                self.db.add(db_file)
                added_files.append(file_obj.filename)
                file_paths.append(file_path)
            
            # Commit all file records
            self.db.commit()
            
            # Update dataset status if needed
            if request.reprocess:
                dataset.is_processed = False
                dataset.processing_status = "pending"
                self.db.commit()
            
            logger.info("Files added to dataset", 
                       dataset_id=dataset_id, 
                       files_count=len(added_files),
                       files=added_files)
            
            return DatasetFileAddResponse(
                dataset_id=dataset_id,
                files_added=added_files,
                total_files_added=len(added_files),
                message=f"Successfully added {len(added_files)} files to dataset",
                processing_started=request.reprocess
            )
            
        except Exception as e:
            logger.error("Failed to add files to dataset", 
                       error=str(e), 
                       dataset_id=dataset_id)
            self.db.rollback()
            
            # Clean up uploaded files on error
            for file_info in files:
                _, file_path, _, _ = file_info
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.info("Cleaned up file after error", file_path=file_path)
                    except Exception as cleanup_error:
                        logger.error("Failed to clean up file", 
                                   file_path=file_path, 
                                   error=str(cleanup_error))
            
            raise
    
    async def get_dataset_files(self, dataset_id: int) -> List[DatasetFileInfo]:
        """Get all files in a dataset"""
        try:
            files = self.db.query(DatasetFile).filter(
                DatasetFile.dataset_id == dataset_id,
                DatasetFile.del_flag == 0
            ).all()
            
            return [
                DatasetFileInfo(
                    filename=file.filename,
                    file_path=file.file_path,
                    file_type=file.file_type,
                    file_size=file.file_size,
                    created_at=file.created_at,
                    description=file.description
                )
                for file in files
            ]
            
        except Exception as e:
            logger.error("Failed to get dataset files", error=str(e), dataset_id=dataset_id)
            raise
    
    def _db_to_response(self, dataset: DatasetInfo) -> DatasetResponse:
        """Convert database model to response schema"""
        stats = None
        if dataset.is_processed:
            stats = self._get_dataset_stats(dataset)
        
        return DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            path=dataset.path,
            image_size=dataset.image_size,
            frame_extraction_interval=dataset.frame_extraction_interval,
            max_frames_per_video=dataset.max_frames_per_video,
            stats=stats,
            is_processed=dataset.is_processed,
            processing_status=dataset.processing_status,
            error_message=dataset.error_message,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at
        )
