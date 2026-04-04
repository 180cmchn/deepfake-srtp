"""
Training service for deepfake detection platform
"""

import asyncio
from collections import Counter
import html
import hashlib
import json
import random
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
from typing import List, Optional, Dict, Any, Tuple, Callable

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from sqlalchemy.orm import Session
from sqlalchemy import or_
from fastapi import BackgroundTasks

from app.core.database import get_db_session
from app.core.logging import logger
from app.core.config import settings
from app.core.video_aggregation import aggregate_logit_sequence
from app.core.video_face_roi import SingleFaceRoiProcessor
from app.models.database_models import ModelRegistry, TrainingJob
from app.schemas.models import ModelStatus
from app.schemas.training import (
    EpochMetricPoint,
    TrainingJobCreate,
    TrainingJobResponse,
    TrainingJobList,
    TrainingJobUpdate,
    TrainingEpochMetricsResponse,
    TrainingProgress,
    TrainingMetrics,
    TrainingParameters,
    JobStatus,
)
from app.models.ml_models import create_model


FAKE_LABEL_KEYWORDS = {
    "fake",
    "deepfake",
    "manipulated",
    "forged",
    "tampered",
    "class1",
    "1",
}
REAL_LABEL_KEYWORDS = {
    "real",
    "authentic",
    "original",
    "genuine",
    "pristine",
    "class0",
    "0",
}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
CELEB_DF_SUBGROUPS = ("fake", "celeb-real", "youtube-real")


@dataclass(frozen=True)
class MediaClassificationSample:
    """A single image sample or a sampled frame from a raw video."""

    path: str
    label: int
    frame_index: Optional[int] = None
    source_id: Optional[str] = None


@dataclass(frozen=True)
class TemporalMediaSource:
    kind: str
    label: int
    path: Optional[str] = None
    frame_paths: Optional[List[str]] = None
    source_id: Optional[str] = None


@dataclass(frozen=True)
class TemporalClipSample:
    frame_paths: List[str]
    label: int
    source_id: str
    roi_materialized: bool = False


class TrainingCancelledError(Exception):
    """Raised when a training job is cancelled during execution."""


class ImageClassificationDataset(Dataset):
    """Simple image classification dataset for real/fake labels."""

    def __init__(
        self,
        samples: List[MediaClassificationSample],
        transform=None,
        include_source_id: bool = False,
        roi_processor: Optional[SingleFaceRoiProcessor] = None,
        roi_policy: Optional[Dict[str, Any]] = None,
    ):
        self.samples = samples
        self.transform = transform
        self.include_source_id = include_source_id
        self.roi_processor = roi_processor
        self.roi_policy = roi_policy

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        label = sample.label

        if sample.frame_index is None:
            with Image.open(sample.path) as img:
                image = img.convert("RGB")
        else:
            image = self._load_video_frame(sample.path, sample.frame_index)

        if self.roi_processor and self.roi_policy:
            image, _ = self.roi_processor.crop_pil(image, self.roi_policy)

        if self.transform:
            image = self.transform(image)
        if self.include_source_id:
            return image, label, sample.source_id or sample.path
        return image, label

    @staticmethod
    def _load_video_frame(video_path: str, frame_index: int) -> Image.Image:
        """Load a single RGB frame from a raw video file."""
        frame = ImageClassificationDataset._read_frame_direct(video_path, frame_index)

        if frame is None and frame_index > 0:
            fallback_candidates = [
                max(0, frame_index - 1),
                max(0, frame_index - 4),
                max(0, frame_index - 12),
                0,
            ]
            for fallback_index in fallback_candidates:
                frame = ImageClassificationDataset._read_frame_direct(
                    video_path, fallback_index
                )
                if frame is not None:
                    break

        if frame is None and frame_index > 0:
            frame = ImageClassificationDataset._read_frame_sequential(
                video_path, frame_index
            )

        if frame is None:
            raise ValueError(
                f"Cannot read frame {frame_index} from video file: {video_path}"
            )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    @staticmethod
    def _read_frame_direct(video_path: str, frame_index: int):
        """Attempt to read a specific frame by random access."""
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            return None

        try:
            if frame_index > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            return frame if success and frame is not None else None
        finally:
            capture.release()

    @staticmethod
    def _read_frame_sequential(video_path: str, frame_index: int):
        """Sequential fallback for codecs where random seek is unreliable."""
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            return None

        current_index = 0
        last_valid_frame = None
        try:
            while current_index <= frame_index:
                success, frame = capture.read()
                if not success or frame is None:
                    break
                last_valid_frame = frame
                current_index += 1
            return last_valid_frame
        finally:
            capture.release()


class SequenceClassificationDataset(Dataset):
    """Dataset for video frame clips used by sequence models."""

    def __init__(
        self,
        samples: List[TemporalClipSample],
        transform=None,
        augment: bool = False,
        include_source_id: bool = False,
        roi_processor: Optional[SingleFaceRoiProcessor] = None,
        roi_policy: Optional[Dict[str, Any]] = None,
    ):
        self.samples = samples
        self.transform = transform
        self.augment = augment
        self.include_source_id = include_source_id
        self.roi_processor = roi_processor
        self.roi_policy = roi_policy

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        frame_paths = sample.frame_paths
        label = sample.label
        frames = []

        for frame_path in frame_paths:
            with Image.open(frame_path) as img:
                image = img.convert("RGB")
            if self.roi_processor and self.roi_policy and not sample.roi_materialized:
                image, _ = self.roi_processor.crop_pil(image, self.roi_policy)
            frames.append(image)

        if self.augment and frames:
            frames = self._apply_clip_augmentation(frames)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        clip = torch.stack(frames, dim=0)
        if self.include_source_id:
            return clip, label, sample.source_id
        return clip, label

    def _apply_clip_augmentation(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Apply temporally consistent augmentation to an entire clip."""
        augmented = list(frames)

        if random.random() < 0.5:
            augmented = [ImageOps.mirror(frame) for frame in augmented]

        if random.random() < 0.35:
            width, height = augmented[0].size
            crop_scale = random.uniform(0.84, 1.0)
            crop_width = max(1, int(width * crop_scale))
            crop_height = max(1, int(height * crop_scale))
            left = 0 if crop_width >= width else random.randint(0, width - crop_width)
            top = (
                0 if crop_height >= height else random.randint(0, height - crop_height)
            )
            crop_box = (left, top, left + crop_width, top + crop_height)
            augmented = [
                frame.crop(crop_box).resize((width, height), Image.BILINEAR)
                for frame in augmented
            ]

        if random.random() < 0.4:
            brightness = random.uniform(0.88, 1.12)
            contrast = random.uniform(0.88, 1.12)
            saturation = random.uniform(0.9, 1.1)
            augmented = [
                ImageEnhance.Color(
                    ImageEnhance.Contrast(
                        ImageEnhance.Brightness(frame).enhance(brightness)
                    ).enhance(contrast)
                ).enhance(saturation)
                for frame in augmented
            ]

        if random.random() < 0.18:
            angle = random.uniform(-4.0, 4.0)
            augmented = [
                frame.rotate(angle, resample=Image.BILINEAR) for frame in augmented
            ]

        if random.random() < 0.16:
            blur_radius = random.uniform(0.35, 0.8)
            augmented = [
                frame.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                for frame in augmented
            ]

        return augmented


class TrainingService:
    """Service for handling model training operations"""

    _shared_active_jobs: Dict[int, Dict[str, Any]] = {}

    def __init__(self, db: Session):
        self.db = db
        self._active_jobs = self._shared_active_jobs
        self._face_roi_processor = SingleFaceRoiProcessor()

    def _build_face_roi_policy(
        self, parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return self._face_roi_processor.build_policy(parameters or {})

    def _estimate_temporal_clip_quality(
        self, frame_paths: List[str]
    ) -> Optional[Dict[str, float]]:
        if not frame_paths:
            return None

        grayscale_frames: List[np.ndarray] = []
        for frame_path in frame_paths:
            try:
                with Image.open(frame_path) as image:
                    grayscale = image.convert("L").resize((48, 48), Image.BILINEAR)
                    grayscale_frames.append(np.array(grayscale, dtype=np.float32))
            except Exception:
                return None

        if not grayscale_frames:
            return None

        frame_stds = [float(frame.std()) for frame in grayscale_frames]
        average_frame_std = float(sum(frame_stds) / len(frame_stds))

        if len(grayscale_frames) == 1:
            average_motion_delta = 0.0
        else:
            motion_deltas = [
                float(np.mean(np.abs(current - previous)))
                for previous, current in zip(grayscale_frames, grayscale_frames[1:])
            ]
            average_motion_delta = float(sum(motion_deltas) / len(motion_deltas))

        return {
            "average_frame_std": average_frame_std,
            "average_motion_delta": average_motion_delta,
        }

    def _filter_temporal_clip_samples(
        self,
        samples: List[TemporalClipSample],
        *,
        min_frame_std: float,
        min_motion_delta: float,
        progress_callback: Optional[
            Callable[[float, str, Optional[Dict[str, Any]]], None]
        ] = None,
        progress_label: str = "正在过滤低质量时序片段",
    ) -> Tuple[List[TemporalClipSample], Dict[str, Any]]:
        if not samples:
            return [], {"kept": 0, "dropped": 0}

        kept_samples: List[TemporalClipSample] = []
        dropped_count = 0
        total_samples = len(samples)
        progress_interval = max(1, total_samples // 20)

        for index, sample in enumerate(samples, start=1):
            quality = self._estimate_temporal_clip_quality(sample.frame_paths)
            keep_sample = True
            if quality is None:
                keep_sample = False
            else:
                average_frame_std = quality["average_frame_std"]
                average_motion_delta = quality["average_motion_delta"]
                keep_sample = (
                    average_frame_std >= min_frame_std
                    and average_motion_delta >= min_motion_delta
                )

            if keep_sample:
                kept_samples.append(sample)
            else:
                dropped_count += 1

            if progress_callback and (
                index == 1 or index % progress_interval == 0 or index == total_samples
            ):
                self._report_indexed_progress(
                    progress_callback,
                    index,
                    total_samples,
                    f"{progress_label}：{index}/{total_samples}",
                    stage="filter_temporal_clips",
                    unit="clips",
                )

        if not kept_samples:
            kept_samples = list(samples)
            dropped_count = 0

        return kept_samples, {
            "kept": len(kept_samples),
            "dropped": dropped_count,
            "min_frame_std": float(min_frame_std),
            "min_motion_delta": float(min_motion_delta),
        }

    def _resolve_training_device(
        self, requested_device: Optional[str] = None
    ) -> torch.device:
        """Resolve training device from explicit selection or config."""
        configured = (requested_device or settings.TRAINING_DEVICE).lower()
        mps_available = (
            torch.backends.mps.is_available() and torch.backends.mps.is_built()
        )
        cuda_available = torch.cuda.is_available()

        if configured == "cuda":
            if cuda_available:
                return torch.device("cuda")
            raise ValueError("TRAINING_DEVICE is set to cuda but CUDA is not available")

        if configured == "mps":
            if mps_available:
                return torch.device("mps")
            raise ValueError("TRAINING_DEVICE is set to mps but MPS is not available")

        if configured == "cpu":
            return torch.device("cpu")

        available_accelerators = []
        if mps_available:
            available_accelerators.append("mps")
        if cuda_available:
            available_accelerators.append("cuda")

        if len(available_accelerators) > 1:
            raise ValueError(
                "Both MPS and CUDA are available; set training_device explicitly to 'mps' or 'cuda'"
            )
        if len(available_accelerators) == 1:
            return torch.device(available_accelerators[0])
        return torch.device("cpu")

    def _using_mps(self, device: torch.device) -> bool:
        return device.type == "mps"

    def _is_apple_silicon(self) -> bool:
        machine = os.uname().machine.lower() if hasattr(os, "uname") else ""
        return machine == "arm64" and torch.backends.mps.is_built()

    def _tune_batch_size(
        self,
        requested_batch_size: int,
        device: torch.device,
        sequence_mode: bool = False,
    ) -> int:
        """Clamp batch size for Apple Silicon unified memory machines."""
        batch_size = max(1, requested_batch_size)
        if self._is_apple_silicon() and self._using_mps(device):
            cap = (
                settings.APPLE_SILICON_SEQUENCE_BATCH_SIZE_CAP
                if sequence_mode
                else settings.APPLE_SILICON_BATCH_SIZE_CAP
            )
            batch_size = min(batch_size, cap)
        return batch_size

    def _dataloader_kwargs(self, dataset_size: int) -> Dict[str, Any]:
        """Return DataLoader kwargs tuned for local hardware."""
        workers = min(settings.TRAINING_NUM_WORKERS, max(1, dataset_size))
        kwargs: Dict[str, Any] = {
            "num_workers": workers,
            "pin_memory": torch.cuda.is_available(),
        }
        if workers > 0:
            kwargs["persistent_workers"] = settings.TRAINING_PERSISTENT_WORKERS
            kwargs["prefetch_factor"] = settings.TRAINING_PREFETCH_FACTOR
        return kwargs

    def _uses_video_backed_samples(
        self, samples: List[MediaClassificationSample]
    ) -> bool:
        """Return True when image-model training pulls frames directly from videos."""
        return any(sample.frame_index is not None for sample in samples)

    def _tune_loader_kwargs_for_media_samples(
        self, dataset_size: int, samples: List[MediaClassificationSample]
    ) -> Dict[str, Any]:
        """Avoid multiprocessing instability when OpenCV reads raw videos."""
        if self._uses_video_backed_samples(samples):
            return {"num_workers": 0, "pin_memory": False}
        return self._dataloader_kwargs(dataset_size)

    def _configure_acceleration_backend(self, device: torch.device):
        """Enable safe CUDA runtime optimizations."""
        if device.type != "cuda":
            return
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    def _count_labels(self, labels: List[int]) -> Dict[int, int]:
        """Count labels and keep output stable for logging/debugging."""
        counts = Counter(int(label) for label in labels)
        return {label: counts[label] for label in sorted(counts)}

    def _log_label_distribution(self, stage: str, labels: List[int]) -> None:
        """Log label counts to make class imbalance visible during training."""
        counts = self._count_labels(labels)
        if not counts:
            return
        logger.info(
            "Training label distribution",
            stage=stage,
            total=sum(counts.values()),
            counts=counts,
        )

    def _ensure_label_coverage(
        self,
        train_labels: List[int],
        val_labels: List[int],
        context: str,
    ) -> None:
        """Require both train and validation splits to cover every present label."""
        train_set = {int(label) for label in train_labels}
        val_set = {int(label) for label in val_labels}
        observed = train_set | val_set
        if len(observed) < 2:
            raise ValueError(f"{context} must contain at least 2 distinct labels")

        missing_train = sorted(observed - train_set)
        missing_val = sorted(observed - val_set)
        if missing_train or missing_val:
            raise ValueError(
                f"{context} must keep every label in both train and validation splits; "
                f"missing_train={missing_train}, missing_val={missing_val}"
            )

    def _stratified_split_items(
        self,
        items: List[Any],
        validation_split: float,
        label_getter: Callable[[Any], int],
        context: str,
    ) -> Tuple[List[Any], List[Any]]:
        """Split sources per label so train/val keep class coverage and ratio."""
        label_groups: Dict[int, List[Any]] = {}
        for item in items:
            label = int(label_getter(item))
            label_groups.setdefault(label, []).append(item)

        if len(label_groups) < 2:
            raise ValueError(f"{context} requires at least 2 labels to create a split")

        rng = random.Random(42)
        train_items: List[Any] = []
        val_items: List[Any] = []

        for label, group in sorted(label_groups.items()):
            if len(group) < 2:
                raise ValueError(
                    f"{context} needs at least 2 items for label {label} to stratify train/val"
                )
            shuffled = list(group)
            rng.shuffle(shuffled)

            val_count = max(1, int(round(len(shuffled) * validation_split)))
            if val_count >= len(shuffled):
                val_count = len(shuffled) - 1

            val_items.extend(shuffled[:val_count])
            train_items.extend(shuffled[val_count:])

        rng.shuffle(train_items)
        rng.shuffle(val_items)
        self._ensure_label_coverage(
            [int(label_getter(item)) for item in train_items],
            [int(label_getter(item)) for item in val_items],
            context,
        )
        return train_items, val_items

    def _build_balanced_sampler(
        self, labels: List[int], context: str
    ) -> Optional[WeightedRandomSampler]:
        """Disable sampler reweighting when imbalance is handled explicitly elsewhere."""
        counts = self._count_labels(labels)
        if counts:
            logger.info(
                "Skipping sampler reweighting; using class-weighted loss only",
                context=context,
                counts=counts,
            )
        return None

    def _resolve_temporal_clip_budget(
        self, sequence_length: int, training: bool
    ) -> int:
        """Limit per-source clip counts while still sampling enough temporal windows."""
        reference_frames = max(1, int(settings.MAX_FRAMES_PER_VIDEO))
        if training:
            divisor = max(1, sequence_length // 2)
            base_budget = (reference_frames + divisor - 1) // divisor
            return min(8, max(4, base_budget))

        divisor = max(1, sequence_length)
        base_budget = (reference_frames + divisor - 1) // divisor
        return min(4, max(2, base_budget))

    def _sample_evenly_spaced_items(
        self, values: List[int], target_count: Optional[int]
    ) -> List[int]:
        """Subsample an ordered list while preserving temporal coverage."""
        if target_count is None or target_count <= 0 or len(values) <= target_count:
            return values
        if target_count == 1:
            return [values[len(values) // 2]]

        step = (len(values) - 1) / float(target_count - 1)
        sampled = [values[int(round(step * index))] for index in range(target_count)]
        deduplicated: List[int] = []
        for value in sampled:
            if not deduplicated or deduplicated[-1] != value:
                deduplicated.append(value)
        return deduplicated or values[:target_count]

    def _infer_validation_subgroup_from_path(
        self, media_path: Optional[str], root_path: Optional[Path] = None
    ) -> str:
        if not media_path:
            return "other"

        normalized_path = (
            self._relative_media_key(media_path, root_path)
            if root_path is not None
            else Path(media_path).as_posix().lower()
        )
        path_parts = [part for part in normalized_path.split("/") if part]
        head = path_parts[0] if path_parts else normalized_path
        is_celeb_df_context = (
            root_path is not None
            and self._celeb_df_manifest_path(root_path) is not None
        ) or ("celeb-df" in normalized_path)

        if head in {"fake", "celeb-synthesis"}:
            return "fake"
        if head in {"youtube-real", "youtube_real"}:
            return "youtube-real"
        if head in {"real", "celeb-real", "celeb_real"}:
            return "celeb-real" if is_celeb_df_context else "real"

        normalized_text = normalized_path.replace("_", "-")
        if "youtube-real" in normalized_text:
            return "youtube-real"
        if "celeb-real" in normalized_text or (
            is_celeb_df_context and "/real/" in normalized_text
        ):
            return "celeb-real"
        if "celeb-synthesis" in normalized_text or "/fake/" in normalized_text:
            return "fake"
        if self._matches_keyword(normalized_text, FAKE_LABEL_KEYWORDS - {"0", "1"}):
            return "fake"
        if self._matches_keyword(normalized_text, REAL_LABEL_KEYWORDS - {"0", "1"}):
            return "real"
        return "other"

    def _stratified_split_items_by_group(
        self,
        items: List[Any],
        validation_split: float,
        group_getter: Callable[[Any], str],
        context: str,
    ) -> Tuple[List[Any], List[Any]]:
        group_buckets: Dict[str, List[Any]] = {}
        for item in items:
            group = group_getter(item) or "other"
            group_buckets.setdefault(group, []).append(item)

        if len(group_buckets) < 2:
            raise ValueError(f"{context} requires at least 2 groups to create a split")

        rng = random.Random(42)
        train_items: List[Any] = []
        val_items: List[Any] = []
        for group, bucket in sorted(group_buckets.items()):
            if len(bucket) < 2:
                raise ValueError(
                    f"{context} needs at least 2 items for subgroup {group} to stratify train/val"
                )
            shuffled = list(bucket)
            rng.shuffle(shuffled)

            val_count = max(1, int(round(len(shuffled) * validation_split)))
            if val_count >= len(shuffled):
                val_count = len(shuffled) - 1

            val_items.extend(shuffled[:val_count])
            train_items.extend(shuffled[val_count:])

        rng.shuffle(train_items)
        rng.shuffle(val_items)
        return train_items, val_items

    def _compute_subgroup_macro_accuracy(
        self, subgroup_metrics: Dict[str, Dict[str, Any]]
    ) -> Optional[float]:
        normalized_metrics = dict(subgroup_metrics)
        if "real" in normalized_metrics and "celeb-real" not in normalized_metrics:
            normalized_metrics["celeb-real"] = normalized_metrics["real"]

        candidate_groups = []
        if any(group in normalized_metrics for group in CELEB_DF_SUBGROUPS):
            candidate_groups = [
                group for group in CELEB_DF_SUBGROUPS if group in normalized_metrics
            ]
        else:
            candidate_groups = [
                group for group in normalized_metrics.keys() if group not in {"other"}
            ]

        scores = [
            normalized_metrics[group].get("video_accuracy")
            for group in candidate_groups
            if normalized_metrics[group].get("video_accuracy") is not None
        ]
        if not scores:
            return None
        numeric_scores = [float(score) for score in scores if score is not None]
        if not numeric_scores:
            return None
        return float(sum(numeric_scores) / len(numeric_scores))

    def _celeb_df_manifest_path(self, root_path: Path) -> Optional[Path]:
        manifest_path = root_path / "List_of_testing_videos.txt"
        if manifest_path.exists() and manifest_path.is_file():
            return manifest_path
        return None

    def _normalize_celeb_df_relative_path(self, relative_path: str) -> str:
        alias_map = {
            "celeb-synthesis": "fake",
            "celeb_real": "real",
            "celeb-real": "real",
            "youtube_real": "youtube-real",
            "youtube-real": "youtube-real",
        }
        normalized_parts = []
        for part in Path(relative_path).as_posix().split("/"):
            token = part.strip()
            if not token or token == ".":
                continue
            normalized_parts.append(alias_map.get(token.lower(), token.lower()))
        return "/".join(normalized_parts)

    def _load_celeb_df_test_paths(self, root_path: Path) -> Optional[set[str]]:
        manifest_path = self._celeb_df_manifest_path(root_path)
        if manifest_path is None:
            return None

        test_paths: set[str] = set()
        with manifest_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                candidate_path = parts[1] if len(parts) == 2 else parts[0]
                normalized = self._normalize_celeb_df_relative_path(candidate_path)
                if normalized:
                    test_paths.add(normalized)
        return test_paths

    def _relative_media_key(self, media_path: str, root_path: Path) -> str:
        path_obj = Path(media_path)
        try:
            relative_path = (
                path_obj.resolve().relative_to(root_path.resolve()).as_posix()
            )
        except Exception:
            try:
                relative_path = path_obj.relative_to(root_path).as_posix()
            except Exception:
                relative_path = path_obj.as_posix()
        return self._normalize_celeb_df_relative_path(relative_path)

    def _partition_items_by_official_test_list(
        self,
        items: List[Any],
        root_path: Path,
        path_getter: Callable[[Any], Optional[str]],
        context: str,
    ) -> Tuple[List[Any], List[Any]]:
        official_test_paths = self._load_celeb_df_test_paths(root_path)
        if not official_test_paths:
            return items, []

        train_val_items: List[Any] = []
        official_test_items: List[Any] = []
        for item in items:
            media_path = path_getter(item)
            if not media_path:
                train_val_items.append(item)
                continue
            relative_key = self._relative_media_key(media_path, root_path)
            if relative_key in official_test_paths:
                official_test_items.append(item)
            else:
                train_val_items.append(item)

        logger.info(
            "Applied official Celeb-DF test split",
            context=context,
            root_path=str(root_path),
            train_val_items=len(train_val_items),
            official_test_items=len(official_test_items),
        )
        return train_val_items, official_test_items

    def _downsample_majority_items(
        self,
        items: List[Any],
        label_getter: Callable[[Any], int],
        context: str,
    ) -> List[Any]:
        """Downsample the majority class in the training split to the minority count."""
        if not items:
            return items

        label_groups: Dict[int, List[Any]] = {}
        for item in items:
            label = int(label_getter(item))
            label_groups.setdefault(label, []).append(item)

        if len(label_groups) < 2:
            return items

        counts_before = {
            label: len(group) for label, group in sorted(label_groups.items())
        }
        minority_count = min(counts_before.values())
        majority_count = max(counts_before.values())
        if majority_count <= minority_count:
            return items

        rng = random.Random(42)
        balanced_items: List[Any] = []
        counts_after: Dict[int, int] = {}
        for label, group in sorted(label_groups.items()):
            selected = list(group)
            rng.shuffle(selected)
            if len(selected) > minority_count:
                selected = selected[:minority_count]
            counts_after[label] = len(selected)
            balanced_items.extend(selected)

        rng.shuffle(balanced_items)
        logger.info(
            "Downsampled majority class for training split",
            context=context,
            before=counts_before,
            after=counts_after,
        )
        return balanced_items

    def _compute_class_weights(
        self,
        labels: List[int],
        num_classes: int = 2,
        strategy: str = "balanced",
    ) -> Optional[List[float]]:
        if strategy == "none":
            return None

        counts = self._count_labels(labels)
        if len(counts) < 2:
            return None

        fake_count = counts.get(0, 0)
        real_count = counts.get(1, 0)
        total_count = sum(counts.values())
        weights: List[float] = []
        for label in range(num_classes):
            count = counts.get(label)
            if not count:
                weights.append(1.0)
                continue

            if strategy == "fake_prior":
                if label == 0 and fake_count and real_count:
                    imbalance_ratio = max(fake_count, real_count) / float(
                        max(1, min(fake_count, real_count))
                    )
                    weight = max(1.0, imbalance_ratio**0.5)
                else:
                    weight = 1.0
            else:
                balanced_weight = (total_count / float(num_classes * count)) ** 0.5
                weight = min(3.0, max(0.6, balanced_weight))
            weights.append(min(3.0, weight))

        logger.info(
            "Computed class weights for imbalanced training",
            counts=counts,
            weights=weights,
            strategy=strategy,
        )
        return weights

    def _build_training_criterion(
        self,
        label_smoothing: float,
        class_weights: Optional[List[float]] = None,
        device: Optional[torch.device] = None,
    ) -> nn.Module:
        """Create a classification loss with optional smoothing fallback."""
        weight_tensor = None
        if class_weights:
            weight_tensor = torch.tensor(
                class_weights,
                dtype=torch.float32,
                device=device or torch.device("cpu"),
            )
        try:
            return nn.CrossEntropyLoss(
                weight=weight_tensor,
                label_smoothing=label_smoothing,
            )
        except TypeError:
            if label_smoothing > 0:
                logger.warning(
                    "CrossEntropyLoss label_smoothing unsupported, falling back to plain CE",
                    label_smoothing=label_smoothing,
                )
            return nn.CrossEntropyLoss(weight=weight_tensor)

    def _build_training_optimizer(
        self,
        model: nn.Module,
        learning_rate: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        """Create an optimizer with decoupled weight decay when enabled."""
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        params = trainable_params if trainable_params else model.parameters()
        if weight_decay > 0:
            return torch.optim.AdamW(
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        return torch.optim.Adam(params, lr=learning_rate)

    def _build_plateau_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        patience: int,
    ) -> Optional[torch.optim.lr_scheduler.ReduceLROnPlateau]:
        """Create a conservative LR scheduler for long plateaus."""
        if patience <= 1:
            return None
        scheduler_patience = max(1, patience // 2)
        min_lr = min(1e-5, learning_rate) if learning_rate > 0 else 1e-6
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=scheduler_patience,
            threshold=1e-4,
            min_lr=min_lr,
        )

    def _get_optimizer_learning_rate(self, optimizer: torch.optim.Optimizer) -> float:
        """Read the current learning rate from the first param group."""
        for group in optimizer.param_groups:
            return float(group.get("lr", 0.0))
        return 0.0

    def _is_validation_improved(
        self,
        selection_score: float,
        val_accuracy: float,
        val_loss: Optional[float],
        best_selection_score: float,
        best_val_accuracy: float,
        best_val_loss: Optional[float],
        min_delta: float,
    ) -> bool:
        """Determine whether validation metrics improved enough to keep training."""
        if best_selection_score < 0:
            return True
        if selection_score > best_selection_score + min_delta:
            return True

        score_gap = abs(selection_score - best_selection_score)
        if score_gap <= min_delta and val_accuracy > best_val_accuracy + min_delta:
            return True

        accuracy_gap = abs(val_accuracy - best_val_accuracy)
        if accuracy_gap <= min_delta and val_loss is not None:
            if best_val_loss is None:
                return True
            loss_delta = max(min_delta * 0.1, 1e-5)
            return val_loss < best_val_loss - loss_delta

        return False

    def _video_frame_cache_root(self, input_size: int) -> Path:
        return Path(settings.DATA_DIR) / "frame_cache" / f"{input_size}px"

    def _create_phase_progress_reporter(
        self,
        job_id: int,
        start_progress: float,
        end_progress: float,
        total_epochs: int,
        current_epoch: int = 0,
    ) -> Callable[[float, str, Optional[Dict[str, Any]]], None]:
        """Map a normalized phase progress value onto the job progress scale."""

        progress_span = max(0.0, end_progress - start_progress)

        def reporter(
            ratio: float,
            message: str,
            detail: Optional[Dict[str, Any]] = None,
        ):
            normalized_ratio = min(max(ratio, 0.0), 1.0)
            progress_value = start_progress + (progress_span * normalized_ratio)
            self._update_running_progress(
                job_id,
                progress_value,
                epoch=current_epoch,
                total_epochs=total_epochs,
                message=message,
            )
            self._update_preprocessing_metadata(
                job_id,
                normalized_ratio,
                detail or {},
            )

        return reporter

    def _chain_progress_reporter(
        self,
        parent_reporter: Optional[
            Callable[[float, str, Optional[Dict[str, Any]]], None]
        ],
        start_ratio: float,
        end_ratio: float,
    ) -> Optional[Callable[[float, str, Optional[Dict[str, Any]]], None]]:
        """Create a nested reporter for a sub-range inside another phase."""
        if parent_reporter is None:
            return None

        ratio_span = max(0.0, end_ratio - start_ratio)

        def reporter(
            ratio: float,
            message: str,
            detail: Optional[Dict[str, Any]] = None,
        ):
            normalized_ratio = min(max(ratio, 0.0), 1.0)
            parent_reporter(
                start_ratio + (ratio_span * normalized_ratio),
                message,
                detail,
            )

        return reporter

    def _report_indexed_progress(
        self,
        progress_callback: Optional[
            Callable[[float, str, Optional[Dict[str, Any]]], None]
        ],
        index: int,
        total: int,
        message: str,
        stage: Optional[str] = None,
        unit: Optional[str] = None,
    ) -> None:
        """Report progress for an indexed loop without duplicating math."""
        if progress_callback is None:
            return
        detail: Dict[str, Any] = {
            "stage": stage,
            "current": index,
            "total": total,
            "unit": unit,
        }
        progress_callback(index / max(total, 1), message, detail)

    def _update_preprocessing_metadata(
        self,
        job_id: int,
        ratio: float,
        detail: Dict[str, Any],
    ) -> None:
        """Store structured preprocessing progress details for API responses."""
        updates: Dict[str, Any] = {
            "preprocessing_progress": round(min(max(ratio, 0.0), 1.0) * 100.0, 2),
        }
        if detail.get("stage") is not None:
            updates["preprocessing_stage"] = detail.get("stage")
        current_value = detail.get("current")
        if current_value is not None:
            updates["preprocessing_current"] = int(current_value)
        total_value = detail.get("total")
        if total_value is not None:
            updates["preprocessing_total"] = int(total_value)
        if detail.get("unit") is not None:
            updates["preprocessing_unit"] = detail.get("unit")
        self._set_active_job_metadata(job_id, updates)

    def _epoch_metrics_root(self) -> Path:
        """Directory for persisted per-epoch metric history files."""
        return Path(settings.DATA_DIR) / "training_metrics"

    def _epoch_metrics_history_path(self, job_id: int) -> Path:
        """Sidecar JSON file used to persist epoch metrics for a job."""
        return self._epoch_metrics_root() / f"job_{job_id}_epoch_metrics.json"

    def _write_epoch_metrics_payload(
        self, job_id: int, payload: Dict[str, Any]
    ) -> None:
        """Atomically write the epoch metric payload to disk."""
        output_path = self._epoch_metrics_history_path(job_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temp_path, output_path)

    def _read_epoch_metrics_payload(self, job_id: int) -> Dict[str, Any]:
        """Read a persisted epoch metric payload from disk if available."""
        history_path = self._epoch_metrics_history_path(job_id)
        if not history_path.exists() or not history_path.is_file():
            return {"job_id": job_id, "metrics": []}

        try:
            with history_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError, ValueError):
            logger.warning(
                "Epoch metrics history is unreadable",
                job_id=job_id,
                path=str(history_path),
            )
            return {"job_id": job_id, "metrics": []}

        payload.setdefault("job_id", job_id)
        payload.setdefault("metrics", [])
        return payload

    def _reset_epoch_metrics_history(
        self,
        job: TrainingJob,
        total_epochs: Optional[int] = None,
    ) -> None:
        """Initialize a fresh epoch metric history file for a training run."""
        payload = {
            "job_id": job.id,
            "job_name": job.name,
            "model_type": job.model_type,
            "total_epochs": int(total_epochs or job.epochs or 0),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metrics": [],
        }
        self._write_epoch_metrics_payload(job.id, payload)

    def _delete_epoch_metrics_history(self, job_id: int) -> None:
        """Remove persisted epoch metric history for a deleted job."""
        history_path = self._epoch_metrics_history_path(job_id)
        if history_path.exists() and history_path.is_file():
            history_path.unlink()

    def _record_epoch_metrics(
        self,
        job_id: int,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        val_loss: float,
        val_accuracy: float,
        learning_rate: float,
        val_sample_loss: Optional[float] = None,
        val_sample_accuracy: Optional[float] = None,
        val_video_count: Optional[int] = None,
        val_subgroup_metrics: Optional[Dict[str, Any]] = None,
        val_subgroup_macro_accuracy: Optional[float] = None,
        checkpoint_selection_score: Optional[float] = None,
    ) -> None:
        """Append or update one epoch of metric history for charting."""
        payload = self._read_epoch_metrics_payload(job_id)
        metrics = [
            metric
            for metric in payload.get("metrics", [])
            if int(metric.get("epoch", 0)) != int(epoch)
        ]
        metrics.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_accuracy": float(train_accuracy),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_accuracy),
                "val_sample_loss": float(val_sample_loss)
                if val_sample_loss is not None
                else None,
                "val_sample_accuracy": float(val_sample_accuracy)
                if val_sample_accuracy is not None
                else None,
                "val_video_count": int(val_video_count)
                if val_video_count is not None
                else None,
                "val_subgroup_metrics": val_subgroup_metrics,
                "val_subgroup_macro_accuracy": float(val_subgroup_macro_accuracy)
                if val_subgroup_macro_accuracy is not None
                else None,
                "checkpoint_selection_score": float(checkpoint_selection_score)
                if checkpoint_selection_score is not None
                else None,
                "learning_rate": float(learning_rate),
                "recorded_at": datetime.now().isoformat(),
            }
        )
        metrics.sort(key=lambda metric: int(metric.get("epoch", 0)))
        payload["metrics"] = metrics
        payload["updated_at"] = datetime.now().isoformat()
        self._write_epoch_metrics_payload(job_id, payload)

    async def get_epoch_metrics_history(
        self, job_id: int
    ) -> Optional[TrainingEpochMetricsResponse]:
        """Return persisted per-epoch metrics for a training job."""
        job = (
            self.db.query(TrainingJob)
            .filter(TrainingJob.id == job_id, TrainingJob.del_flag == 0)
            .first()
        )
        if not job:
            return None

        payload = self._read_epoch_metrics_payload(job_id)
        metric_points = [
            EpochMetricPoint(**metric)
            for metric in sorted(
                payload.get("metrics", []),
                key=lambda metric: int(metric.get("epoch", 0)),
            )
        ]

        return TrainingEpochMetricsResponse(
            job_id=job.id,
            job_name=job.name,
            model_type=job.model_type,
            total_epochs=job.epochs,
            completed_epochs=len(metric_points),
            available=bool(metric_points),
            metrics=metric_points,
        )

    async def get_epoch_metrics_chart_svg(self, job_id: int) -> Optional[str]:
        """Render an SVG chart for per-epoch loss and accuracy."""
        history = await self.get_epoch_metrics_history(job_id)
        if history is None:
            return None
        return self._build_epoch_metrics_svg(history)

    def _build_epoch_metrics_svg(self, history: TrainingEpochMetricsResponse) -> str:
        """Create a lightweight SVG chart with accuracy and loss curves."""
        width = 980
        height = 640
        background = "#f8fafc"
        panel_fill = "#ffffff"
        grid = "#dbe4ee"
        axis = "#334155"
        text = "#0f172a"
        muted = "#64748b"
        train_color = "#0f766e"
        val_color = "#dc2626"

        escaped_name = html.escape(history.job_name or f"Training Job {history.job_id}")
        metrics = list(history.metrics)

        if not metrics:
            return (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="240" '
                f'viewBox="0 0 {width} 240" role="img" aria-label="No epoch metrics available">'
                f'<rect width="{width}" height="240" fill="{background}" />'
                f'<rect x="24" y="24" width="{width - 48}" height="192" rx="18" fill="{panel_fill}" stroke="{grid}" />'
                f'<text x="48" y="84" fill="{text}" font-size="24" font-family="Arial, sans-serif">{escaped_name}</text>'
                f'<text x="48" y="124" fill="{muted}" font-size="16" font-family="Arial, sans-serif">No epoch metrics available yet.</text>'
                f'<text x="48" y="152" fill="{muted}" font-size="16" font-family="Arial, sans-serif">Start or complete training first, then request this chart again.</text>'
                "</svg>"
            )

        chart_left = 80
        chart_right = width - 40
        chart_width = chart_right - chart_left
        top_panel_top = 120
        panel_height = 180
        panel_gap = 90
        bottom_panel_top = top_panel_top + panel_height + panel_gap

        epochs = [point.epoch for point in metrics]
        min_epoch = min(epochs)
        max_epoch = max(epochs)

        def scale_x(epoch_value: int) -> float:
            if max_epoch == min_epoch:
                return chart_left + (chart_width / 2.0)
            return chart_left + (
                ((epoch_value - min_epoch) / float(max_epoch - min_epoch)) * chart_width
            )

        def build_line_points(
            values: List[Optional[float]],
            y_min: float,
            y_max: float,
            panel_top: float,
        ) -> str:
            points = []
            panel_bottom = panel_top + panel_height
            value_span = y_max - y_min if y_max > y_min else 1.0
            for point, value in zip(metrics, values):
                if value is None:
                    continue
                ratio = (float(value) - y_min) / value_span
                y = panel_bottom - (ratio * panel_height)
                points.append(f"{scale_x(point.epoch):.2f},{y:.2f}")
            return " ".join(points)

        def build_dot_marks(
            values: List[Optional[float]],
            y_min: float,
            y_max: float,
            panel_top: float,
            color: str,
        ) -> str:
            circles = []
            panel_bottom = panel_top + panel_height
            value_span = y_max - y_min if y_max > y_min else 1.0
            for point, value in zip(metrics, values):
                if value is None:
                    continue
                ratio = (float(value) - y_min) / value_span
                y = panel_bottom - (ratio * panel_height)
                circles.append(
                    f'<circle cx="{scale_x(point.epoch):.2f}" cy="{y:.2f}" r="3.5" fill="{color}" />'
                )
            return "".join(circles)

        def build_y_grid(
            ticks: List[float],
            y_min: float,
            y_max: float,
            panel_top: float,
        ) -> str:
            panel_bottom = panel_top + panel_height
            value_span = y_max - y_min if y_max > y_min else 1.0
            grid_lines = []
            for tick in ticks:
                ratio = (tick - y_min) / value_span
                y = panel_bottom - (ratio * panel_height)
                grid_lines.append(
                    f'<line x1="{chart_left}" y1="{y:.2f}" x2="{chart_right}" y2="{y:.2f}" stroke="{grid}" stroke-width="1" />'
                )
                grid_lines.append(
                    f'<text x="{chart_left - 12}" y="{y + 5:.2f}" text-anchor="end" fill="{muted}" font-size="12" font-family="Arial, sans-serif">{tick:.2f}</text>'
                )
            return "".join(grid_lines)

        def build_x_ticks(panel_top: float) -> str:
            panel_bottom = panel_top + panel_height
            tick_step = max(1, len(metrics) // 8)
            ticks = []
            for index, epoch_value in enumerate(epochs):
                if index not in {0, len(epochs) - 1} and index % tick_step != 0:
                    continue
                x = scale_x(epoch_value)
                ticks.append(
                    f'<line x1="{x:.2f}" y1="{panel_bottom}" x2="{x:.2f}" y2="{panel_bottom + 6}" stroke="{axis}" stroke-width="1" />'
                )
                ticks.append(
                    f'<text x="{x:.2f}" y="{panel_bottom + 24}" text-anchor="middle" fill="{muted}" font-size="12" font-family="Arial, sans-serif">{epoch_value}</text>'
                )
            return "".join(ticks)

        train_accuracy_values = [point.train_accuracy for point in metrics]
        val_accuracy_values = [point.val_accuracy for point in metrics]
        train_loss_values = [point.train_loss for point in metrics]
        val_loss_values = [point.val_loss for point in metrics]

        loss_values = [
            float(value)
            for value in train_loss_values + val_loss_values
            if value is not None
        ]
        loss_min = min(loss_values) if loss_values else 0.0
        loss_max = max(loss_values) if loss_values else 1.0
        if loss_max <= loss_min:
            loss_min = max(0.0, loss_min - 0.1)
            loss_max = loss_max + 0.1
        else:
            padding = max((loss_max - loss_min) * 0.1, 0.05)
            loss_min = max(0.0, loss_min - padding)
            loss_max = loss_max + padding

        accuracy_grid = build_y_grid(
            [0.0, 0.25, 0.5, 0.75, 1.0], 0.0, 1.0, top_panel_top
        )
        loss_tick_count = 5
        loss_ticks = [
            loss_min + ((loss_max - loss_min) * tick_index / float(loss_tick_count - 1))
            for tick_index in range(loss_tick_count)
        ]
        loss_grid = build_y_grid(loss_ticks, loss_min, loss_max, bottom_panel_top)

        legend = (
            f'<line x1="600" y1="58" x2="628" y2="58" stroke="{train_color}" stroke-width="3" />'
            f'<text x="636" y="62" fill="{text}" font-size="13" font-family="Arial, sans-serif">Train</text>'
            f'<line x1="710" y1="58" x2="738" y2="58" stroke="{val_color}" stroke-width="3" />'
            f'<text x="746" y="62" fill="{text}" font-size="13" font-family="Arial, sans-serif">Validation</text>'
        )

        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="Training accuracy and loss by epoch">'
            f'<rect width="{width}" height="{height}" fill="{background}" />'
            f'<text x="40" y="50" fill="{text}" font-size="28" font-family="Arial, sans-serif">{escaped_name}</text>'
            f'<text x="40" y="78" fill="{muted}" font-size="15" font-family="Arial, sans-serif">Model: {html.escape(history.model_type or "unknown")} | Epochs recorded: {history.completed_epochs}/{history.total_epochs or history.completed_epochs}</text>'
            f"{legend}"
            f'<rect x="24" y="96" width="{width - 48}" height="{panel_height + 48}" rx="18" fill="{panel_fill}" stroke="{grid}" />'
            f'<text x="40" y="116" fill="{text}" font-size="18" font-family="Arial, sans-serif">Accuracy</text>'
            f"{accuracy_grid}"
            f'<line x1="{chart_left}" y1="{top_panel_top + panel_height}" x2="{chart_right}" y2="{top_panel_top + panel_height}" stroke="{axis}" stroke-width="1.4" />'
            f'<line x1="{chart_left}" y1="{top_panel_top}" x2="{chart_left}" y2="{top_panel_top + panel_height}" stroke="{axis}" stroke-width="1.4" />'
            f'<polyline fill="none" stroke="{train_color}" stroke-width="3" points="{build_line_points(train_accuracy_values, 0.0, 1.0, top_panel_top)}" />'
            f'<polyline fill="none" stroke="{val_color}" stroke-width="3" points="{build_line_points(val_accuracy_values, 0.0, 1.0, top_panel_top)}" />'
            f"{build_dot_marks(train_accuracy_values, 0.0, 1.0, top_panel_top, train_color)}"
            f"{build_dot_marks(val_accuracy_values, 0.0, 1.0, top_panel_top, val_color)}"
            f"{build_x_ticks(top_panel_top)}"
            f'<text x="{width / 2:.2f}" y="{top_panel_top + panel_height + 44}" text-anchor="middle" fill="{muted}" font-size="12" font-family="Arial, sans-serif">Epoch</text>'
            f'<rect x="24" y="{bottom_panel_top - 24}" width="{width - 48}" height="{panel_height + 48}" rx="18" fill="{panel_fill}" stroke="{grid}" />'
            f'<text x="40" y="{bottom_panel_top - 4}" fill="{text}" font-size="18" font-family="Arial, sans-serif">Loss</text>'
            f"{loss_grid}"
            f'<line x1="{chart_left}" y1="{bottom_panel_top + panel_height}" x2="{chart_right}" y2="{bottom_panel_top + panel_height}" stroke="{axis}" stroke-width="1.4" />'
            f'<line x1="{chart_left}" y1="{bottom_panel_top}" x2="{chart_left}" y2="{bottom_panel_top + panel_height}" stroke="{axis}" stroke-width="1.4" />'
            f'<polyline fill="none" stroke="{train_color}" stroke-width="3" points="{build_line_points(train_loss_values, loss_min, loss_max, bottom_panel_top)}" />'
            f'<polyline fill="none" stroke="{val_color}" stroke-width="3" points="{build_line_points(val_loss_values, loss_min, loss_max, bottom_panel_top)}" />'
            f"{build_dot_marks(train_loss_values, loss_min, loss_max, bottom_panel_top, train_color)}"
            f"{build_dot_marks(val_loss_values, loss_min, loss_max, bottom_panel_top, val_color)}"
            f"{build_x_ticks(bottom_panel_top)}"
            f'<text x="{width / 2:.2f}" y="{bottom_panel_top + panel_height + 44}" text-anchor="middle" fill="{muted}" font-size="12" font-family="Arial, sans-serif">Epoch</text>'
            "</svg>"
        )

    def _materialize_video_backed_samples(
        self,
        samples: List[MediaClassificationSample],
        input_size: int,
        progress_callback: Optional[
            Callable[[float, str, Optional[Dict[str, Any]]], None]
        ] = None,
        progress_label: str = "正在缓存视频帧",
    ) -> List[MediaClassificationSample]:
        """Extract sampled raw-video frames once so later epochs can use image IO."""
        if not self._uses_video_backed_samples(samples):
            if progress_callback is not None:
                progress_callback(
                    1.0,
                    f"{progress_label}：无需取帧缓存",
                    {
                        "stage": "cache_video_frames",
                        "current": 0,
                        "total": 0,
                        "unit": "frames",
                    },
                )
            return samples

        cache_root = self._video_frame_cache_root(input_size)
        cache_root.mkdir(parents=True, exist_ok=True)
        materialized: List[MediaClassificationSample] = []
        extracted_count = 0
        video_backed_total = sum(
            1 for sample in samples if sample.frame_index is not None
        )
        processed_video_backed = 0
        progress_interval = max(1, video_backed_total // 20)

        for sample in samples:
            if sample.frame_index is None:
                materialized.append(sample)
                continue

            cached_frame_path = self._materialize_video_frame(
                sample.path,
                sample.frame_index,
                cache_root,
                input_size,
            )
            if cached_frame_path != sample.path:
                extracted_count += 1
            materialized.append(
                MediaClassificationSample(
                    path=cached_frame_path,
                    label=sample.label,
                    source_id=sample.source_id,
                )
            )
            processed_video_backed += 1

            if progress_callback and (
                processed_video_backed == 1
                or processed_video_backed % progress_interval == 0
                or processed_video_backed == video_backed_total
            ):
                progress_callback(
                    processed_video_backed / max(video_backed_total, 1),
                    f"{progress_label}：{processed_video_backed}/{video_backed_total}",
                    {
                        "stage": "cache_video_frames",
                        "current": processed_video_backed,
                        "total": video_backed_total,
                        "unit": "frames",
                    },
                )

        logger.info(
            "Materialized video-backed frame samples",
            total_samples=len(samples),
            cached_samples=len(materialized),
            extracted_count=extracted_count,
            cache_root=str(cache_root),
        )
        return materialized

    def _materialize_video_frame(
        self,
        video_path: str,
        frame_index: int,
        cache_root: Path,
        input_size: int,
        face_roi_policy: Optional[Dict[str, Any]] = None,
    ) -> str:
        source_path = Path(video_path).expanduser().resolve()
        source_stat = source_path.stat()
        policy_fingerprint = self._face_roi_processor.get_policy_fingerprint(
            face_roi_policy
        )
        cache_key = hashlib.sha1(
            f"{source_path}:{source_stat.st_mtime_ns}:{source_stat.st_size}:{frame_index}:{input_size}:{policy_fingerprint}".encode(
                "utf-8"
            )
        ).hexdigest()
        cache_dir = cache_root / cache_key[:2]
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_frame_path = cache_dir / f"{cache_key}.jpg"
        if cached_frame_path.exists():
            return str(cached_frame_path)

        frame_image = ImageClassificationDataset._load_video_frame(
            str(source_path), frame_index
        )
        if face_roi_policy:
            frame_image, _ = self._face_roi_processor.crop_pil(
                frame_image, face_roi_policy
            )
        resized_frame = frame_image.resize((input_size, input_size), Image.BILINEAR)
        temp_path = cached_frame_path.with_suffix(".tmp")
        resized_frame.save(temp_path, format="JPEG", quality=95)
        os.replace(temp_path, cached_frame_path)
        return str(cached_frame_path)

    async def create_job(
        self,
        job: TrainingJobCreate,
        background_tasks: BackgroundTasks,
        auto_start: bool = True,
    ) -> TrainingJobResponse:
        """Create a new training job"""
        try:
            # Check concurrent job limit
            active_count = (
                self.db.query(TrainingJob)
                .filter(
                    TrainingJob.status.in_(
                        [JobStatus.PENDING.value, JobStatus.RUNNING.value]
                    ),
                    TrainingJob.del_flag == 0,
                )
                .count()
            )

            if active_count >= settings.MAX_CONCURRENT_TRAINING_JOBS:
                raise ValueError(
                    f"Maximum concurrent training jobs ({settings.MAX_CONCURRENT_TRAINING_JOBS}) reached"
                )

            # Create training job
            parameters = job.parameters or TrainingParameters()
            db_job = TrainingJob(
                name=job.name,
                description=job.description,
                model_type=job.model_type,
                dataset_path=job.dataset_path,
                epochs=parameters.epochs,
                learning_rate=parameters.learning_rate,
                batch_size=parameters.batch_size,
            )

            self.db.add(db_job)
            self.db.commit()
            self.db.refresh(db_job)
            self._set_active_job_metadata(
                db_job.id,
                {
                    "validation_split": parameters.validation_split,
                    "early_stopping": parameters.early_stopping,
                    "patience": parameters.patience,
                    "early_stopping_min_delta": parameters.early_stopping_min_delta,
                    "weight_decay": parameters.weight_decay,
                    "label_smoothing": parameters.label_smoothing,
                    "class_weight_strategy": parameters.class_weight_strategy,
                    "use_official_test_as_validation": parameters.use_official_test_as_validation,
                    "requested_device": parameters.training_device,
                    "sequence_length": parameters.sequence_length,
                    "frame_stride": parameters.frame_stride,
                    "temporal_hidden_size": parameters.temporal_hidden_size,
                    "temporal_num_layers": parameters.temporal_num_layers,
                    "feature_projection_size": parameters.feature_projection_size,
                    "face_roi_enabled": parameters.face_roi_enabled,
                    "face_roi_confidence_threshold": parameters.face_roi_confidence_threshold,
                    "face_roi_crop_padding": parameters.face_roi_crop_padding,
                    "temporal_bidirectional": parameters.temporal_bidirectional,
                    "temporal_attention_pooling": parameters.temporal_attention_pooling,
                    "train_clip_overlap_ratio": parameters.train_clip_overlap_ratio,
                    "val_clip_overlap_ratio": parameters.val_clip_overlap_ratio,
                },
            )

            # Start training in background only when requested
            if auto_start:
                background_tasks.add_task(
                    self._run_training, db_job.id, parameters.dict()
                )

            logger.info(
                "Training job created",
                job_id=db_job.id,
                model_type=job.model_type,
                auto_start=auto_start,
            )

            return self._db_to_response(db_job)

        except Exception as e:
            logger.error("Failed to create training job", error=str(e))
            self.db.rollback()
            raise

    async def get_jobs(
        self,
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None,
        model_type: Optional[str] = None,
        created_by: Optional[str] = None,
        search: Optional[str] = None,
        order_by: str = "created_at",
        order_desc: bool = True,
    ) -> TrainingJobList:
        """Get training jobs list"""
        try:
            query = self.db.query(TrainingJob).filter(TrainingJob.del_flag == 0)

            if status:
                query = query.filter(TrainingJob.status == status)

            if model_type:
                query = query.filter(TrainingJob.model_type == model_type)

            # created_by is accepted for API compatibility, but not persisted yet
            if created_by:
                logger.warning(
                    "created_by filter ignored: field not implemented",
                    created_by=created_by,
                )

            if search:
                search_term = f"%{search}%"
                query = query.filter(
                    or_(
                        TrainingJob.name.ilike(search_term),
                        TrainingJob.description.ilike(search_term),
                    )
                )

            if hasattr(TrainingJob, order_by):
                order_column = getattr(TrainingJob, order_by)
                query = query.order_by(
                    order_column.desc() if order_desc else order_column.asc()
                )
            else:
                query = query.order_by(TrainingJob.created_at.desc())

            total = query.count()
            results = query.offset(skip).limit(limit).all()

            jobs = [self._db_to_response(job) for job in results]

            return TrainingJobList(
                jobs=jobs,
                total=total,
                page=skip // limit + 1,
                size=limit,
                pages=(total + limit - 1) // limit,
            )

        except Exception as e:
            logger.error("Failed to get training jobs", error=str(e))
            raise

    async def get_job(self, job_id: int) -> Optional[TrainingJobResponse]:
        """Get training job by ID"""
        try:
            job = (
                self.db.query(TrainingJob)
                .filter(TrainingJob.id == job_id, TrainingJob.del_flag == 0)
                .first()
            )

            if not job:
                return None

            return self._db_to_response(job)

        except Exception as e:
            logger.error("Failed to get training job", error=str(e), job_id=job_id)
            raise

    async def update_job(
        self, job_id: int, job_update: TrainingJobUpdate
    ) -> Optional[TrainingJobResponse]:
        """Update training job"""
        try:
            job = (
                self.db.query(TrainingJob)
                .filter(TrainingJob.id == job_id, TrainingJob.del_flag == 0)
                .first()
            )

            if not job:
                return None

            # Update fields
            update_data = job_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                if hasattr(job, field):
                    setattr(job, field, value)

            self.db.commit()
            self.db.refresh(job)

            logger.info("Training job updated", job_id=job_id)

            return self._db_to_response(job)

        except Exception as e:
            logger.error("Failed to update training job", error=str(e), job_id=job_id)
            self.db.rollback()
            raise

    async def delete_job(self, job_id: int) -> bool:
        """Delete training job"""
        try:
            job = (
                self.db.query(TrainingJob)
                .filter(TrainingJob.id == job_id, TrainingJob.del_flag == 0)
                .first()
            )

            if not job:
                return False

            # Cancel if running
            if job.status == JobStatus.RUNNING.value:
                await self._cancel_training(job_id)

            job.del_flag = 1
            self.db.commit()
            self._delete_epoch_metrics_history(job_id)

            logger.info("Training job deleted", job_id=job_id)
            return True

        except Exception as e:
            logger.error("Failed to delete training job", error=str(e), job_id=job_id)
            self.db.rollback()
            raise

    async def get_progress(self, job_id: int) -> Optional[TrainingProgress]:
        """Get training job progress"""
        try:
            job = (
                self.db.query(TrainingJob)
                .filter(TrainingJob.id == job_id, TrainingJob.del_flag == 0)
                .first()
            )

            if not job:
                return None

            active_meta = self._active_jobs.get(job_id, {})

            current_epoch = active_meta.get("current_epoch")
            current_loss = active_meta.get("current_loss")
            current_accuracy = active_meta.get("current_accuracy")
            message = active_meta.get("message")
            preprocessing_stage = active_meta.get("preprocessing_stage")
            preprocessing_progress = active_meta.get("preprocessing_progress")
            preprocessing_current = active_meta.get("preprocessing_current")
            preprocessing_total = active_meta.get("preprocessing_total")
            preprocessing_unit = active_meta.get("preprocessing_unit")

            if current_epoch is None and job.status in [
                JobStatus.COMPLETED.value,
                JobStatus.FAILED.value,
                JobStatus.CANCELLED.value,
            ]:
                current_epoch = job.epochs

            if current_loss is None:
                current_loss = job.loss

            if current_accuracy is None:
                current_accuracy = job.accuracy

            return TrainingProgress(
                job_id=job.id,
                status=self._normalize_job_status(job.status),
                progress=job.progress,
                started_at=job.started_at,
                completed_at=job.completed_at,
                current_epoch=current_epoch,
                total_epochs=job.epochs,
                current_loss=current_loss,
                current_accuracy=current_accuracy,
                estimated_time_remaining=None,  # TODO: Calculate ETA
                message=message,
                preprocessing_stage=preprocessing_stage,
                preprocessing_progress=preprocessing_progress,
                preprocessing_current=preprocessing_current,
                preprocessing_total=preprocessing_total,
                preprocessing_unit=preprocessing_unit,
            )

        except Exception as e:
            logger.error("Failed to get training progress", error=str(e), job_id=job_id)
            raise

    async def get_metrics(self) -> TrainingMetrics:
        """Get training metrics"""
        try:
            # Get job counts
            total_jobs = (
                self.db.query(TrainingJob).filter(TrainingJob.del_flag == 0).count()
            )
            active_jobs = (
                self.db.query(TrainingJob)
                .filter(
                    TrainingJob.status.in_(
                        [JobStatus.PENDING.value, JobStatus.RUNNING.value]
                    ),
                    TrainingJob.del_flag == 0,
                )
                .count()
            )
            completed_jobs = (
                self.db.query(TrainingJob)
                .filter(
                    TrainingJob.status == JobStatus.COMPLETED.value,
                    TrainingJob.del_flag == 0,
                )
                .count()
            )
            failed_jobs = (
                self.db.query(TrainingJob)
                .filter(
                    TrainingJob.status == JobStatus.FAILED.value,
                    TrainingJob.del_flag == 0,
                )
                .count()
            )

            # Calculate average accuracy
            completed_with_accuracy = (
                self.db.query(TrainingJob.accuracy)
                .filter(
                    TrainingJob.status == JobStatus.COMPLETED.value,
                    TrainingJob.accuracy.isnot(None),
                    TrainingJob.del_flag == 0,
                )
                .all()
            )
            average_accuracy = (
                sum(acc[0] for acc in completed_with_accuracy)
                / len(completed_with_accuracy)
                if completed_with_accuracy
                else None
            )

            # Calculate average training time
            completed_with_time = (
                self.db.query(TrainingJob.started_at, TrainingJob.completed_at)
                .filter(
                    TrainingJob.status == JobStatus.COMPLETED.value,
                    TrainingJob.started_at.isnot(None),
                    TrainingJob.completed_at.isnot(None),
                    TrainingJob.del_flag == 0,
                )
                .all()
            )

            training_times = []
            for started, completed in completed_with_time:
                if started and completed:
                    training_times.append((completed - started).total_seconds())

            average_training_time = (
                sum(training_times) / len(training_times) if training_times else None
            )

            # Jobs by model type
            jobs_by_model_type = {}
            for model_type in settings.SUPPORTED_MODELS:
                count = (
                    self.db.query(TrainingJob)
                    .filter(
                        TrainingJob.model_type == model_type, TrainingJob.del_flag == 0
                    )
                    .count()
                )
                if count > 0:
                    jobs_by_model_type[model_type] = count

            # Jobs by status
            jobs_by_status = {
                "pending": self.db.query(TrainingJob)
                .filter(
                    TrainingJob.status == JobStatus.PENDING.value,
                    TrainingJob.del_flag == 0,
                )
                .count(),
                "running": self.db.query(TrainingJob)
                .filter(
                    TrainingJob.status == JobStatus.RUNNING.value,
                    TrainingJob.del_flag == 0,
                )
                .count(),
                "completed": completed_jobs,
                "failed": failed_jobs,
                "cancelled": self.db.query(TrainingJob)
                .filter(
                    TrainingJob.status == JobStatus.CANCELLED.value,
                    TrainingJob.del_flag == 0,
                )
                .count(),
            }

            success_rate = completed_jobs / total_jobs if total_jobs > 0 else 0.0

            return TrainingMetrics(
                total_jobs=total_jobs,
                active_jobs=active_jobs,
                completed_jobs=completed_jobs,
                failed_jobs=failed_jobs,
                average_accuracy=average_accuracy,
                average_training_time=average_training_time,
                success_rate=success_rate,
                jobs_by_model_type=jobs_by_model_type,
                jobs_by_status=jobs_by_status,
            )

        except Exception as e:
            logger.error("Failed to get training metrics", error=str(e))
            raise

    async def start_job(
        self, job_id: int, background_tasks: BackgroundTasks, current_user: str
    ) -> bool:
        """Start a pending training job manually"""
        try:
            job = (
                self.db.query(TrainingJob)
                .filter(TrainingJob.id == job_id, TrainingJob.del_flag == 0)
                .first()
            )

            if not job:
                return False

            if job.status != JobStatus.PENDING.value:
                return False

            job.error_message = None
            self.db.commit()
            pending_meta = self._active_jobs.get(job_id, {})

            background_tasks.add_task(
                self._run_training,
                job_id,
                {
                    "epochs": job.epochs,
                    "learning_rate": job.learning_rate,
                    "batch_size": job.batch_size,
                    "validation_split": pending_meta.get("validation_split", 0.2),
                    "early_stopping": pending_meta.get("early_stopping", True),
                    "patience": pending_meta.get("patience", 10),
                    "early_stopping_min_delta": pending_meta.get(
                        "early_stopping_min_delta", 0.002
                    ),
                    "weight_decay": pending_meta.get("weight_decay", 0.0001),
                    "label_smoothing": pending_meta.get("label_smoothing", 0.05),
                    "class_weight_strategy": pending_meta.get(
                        "class_weight_strategy", "balanced"
                    ),
                    "use_official_test_as_validation": pending_meta.get(
                        "use_official_test_as_validation", False
                    ),
                    "training_device": pending_meta.get(
                        "requested_device", settings.TRAINING_DEVICE
                    ),
                },
            )

            logger.info(
                "Training job start scheduled", job_id=job_id, user=current_user
            )
            return True
        except Exception as e:
            logger.error(
                "Failed to schedule training job",
                error=str(e),
                job_id=job_id,
                user=current_user,
            )
            self.db.rollback()
            raise

    async def stop_job(self, job_id: int, current_user: str) -> bool:
        """Stop a pending or running training job"""
        try:
            job = (
                self.db.query(TrainingJob)
                .filter(TrainingJob.id == job_id, TrainingJob.del_flag == 0)
                .first()
            )

            if not job:
                return False

            if job.status not in [JobStatus.PENDING.value, JobStatus.RUNNING.value]:
                return False

            job.status = JobStatus.CANCELLED.value
            job.completed_at = datetime.now()
            self.db.commit()

            logger.info("Training job cancelled", job_id=job_id, user=current_user)
            return True
        except Exception as e:
            logger.error(
                "Failed to stop training job",
                error=str(e),
                job_id=job_id,
                user=current_user,
            )
            self.db.rollback()
            raise

    async def get_job_logs(self, job_id: int, tail_lines: int = 100) -> List[str]:
        """Get lightweight job logs from job state"""
        try:
            job = (
                self.db.query(TrainingJob)
                .filter(TrainingJob.id == job_id, TrainingJob.del_flag == 0)
                .first()
            )

            if not job:
                return []

            lines = [
                f"[{job.created_at}] Job created: {job.name} (type={job.model_type})",
                f"Status: {job.status}",
                f"Progress: {job.progress:.2f}%",
            ]

            if job.started_at:
                lines.append(f"[{job.started_at}] Training started")
            if job.completed_at:
                lines.append(
                    f"[{job.completed_at}] Training finished with status={job.status}"
                )
            if job.accuracy is not None:
                lines.append(f"Best video validation accuracy: {job.accuracy:.4f}")
            if job.loss is not None:
                lines.append(f"Best video validation loss: {job.loss:.4f}")
            if job.error_message:
                lines.append(f"Error: {job.error_message}")

            return lines[-tail_lines:] if tail_lines > 0 else lines
        except Exception as e:
            logger.error("Failed to get training job logs", error=str(e), job_id=job_id)
            raise

    async def retain_model_file(self, job_id: int, current_user: str) -> Optional[str]:
        """Confirm retaining trained model file for a completed job."""
        try:
            job = (
                self.db.query(TrainingJob)
                .filter(TrainingJob.id == job_id, TrainingJob.del_flag == 0)
                .first()
            )

            if not job:
                return None

            if job.status != JobStatus.COMPLETED.value:
                raise ValueError("Only completed training jobs can retain model files")

            if not job.model_path:
                raise ValueError("No model file available for this job")

            model_path = Path(job.model_path).expanduser()
            if not model_path.is_absolute():
                model_path = (Path.cwd() / model_path).resolve()

            if not model_path.exists() or not model_path.is_file():
                raise ValueError("Model file does not exist on disk")

            registry_model = self._register_retained_model(
                job, model_path, current_user
            )

            logger.info(
                "Model file retention confirmed",
                job_id=job_id,
                user=current_user,
                model_path=str(model_path),
                model_registry_id=registry_model.id,
            )
            return str(model_path)
        except Exception as e:
            logger.error(
                "Failed to retain model file",
                error=str(e),
                job_id=job_id,
                user=current_user,
            )
            raise

    async def discard_model_file(self, job_id: int, current_user: str) -> bool:
        """Delete trained model file for a completed job and clear model path."""
        try:
            job = (
                self.db.query(TrainingJob)
                .filter(TrainingJob.id == job_id, TrainingJob.del_flag == 0)
                .first()
            )

            if not job:
                return False

            if job.status != JobStatus.COMPLETED.value:
                raise ValueError("Only completed training jobs can discard model files")

            if not job.model_path:
                return True

            model_path = Path(job.model_path).expanduser()
            if not model_path.is_absolute():
                model_path = (Path.cwd() / model_path).resolve()

            if model_path.exists() and model_path.is_file():
                model_path.unlink()

            self.db.query(ModelRegistry).filter(
                ModelRegistry.training_job_id == job.id,
                ModelRegistry.del_flag == 0,
            ).update(
                {
                    "del_flag": 1,
                    "status": ModelStatus.ARCHIVED.value,
                    "is_default": False,
                },
                synchronize_session=False,
            )
            job.model_path = None
            self.db.commit()

            logger.info(
                "Model file discarded",
                job_id=job_id,
                user=current_user,
                model_path=str(model_path),
            )
            return True
        except Exception as e:
            logger.error(
                "Failed to discard model file",
                error=str(e),
                job_id=job_id,
                user=current_user,
            )
            self.db.rollback()
            raise

    async def get_job_statistics(self) -> Dict[str, Any]:
        """Get training job statistics overview"""
        try:
            base_query = self.db.query(TrainingJob).filter(TrainingJob.del_flag == 0)
            total_jobs = base_query.count()

            by_status = {
                status.value: self.db.query(TrainingJob)
                .filter(TrainingJob.del_flag == 0, TrainingJob.status == status.value)
                .count()
                for status in JobStatus
            }

            by_model_type: Dict[str, int] = {}
            for model_type in settings.SUPPORTED_MODELS:
                count = (
                    self.db.query(TrainingJob)
                    .filter(
                        TrainingJob.del_flag == 0, TrainingJob.model_type == model_type
                    )
                    .count()
                )
                if count > 0:
                    by_model_type[model_type] = count

            avg_progress_rows = base_query.with_entities(TrainingJob.progress).all()
            average_progress = (
                sum(row[0] for row in avg_progress_rows if row[0] is not None)
                / len(avg_progress_rows)
                if avg_progress_rows
                else 0.0
            )

            return {
                "total_jobs": total_jobs,
                "jobs_by_status": by_status,
                "jobs_by_model_type": by_model_type,
                "average_progress": round(average_progress, 2),
            }
        except Exception as e:
            logger.error("Failed to get training job statistics", error=str(e))
            raise

    # Private methods

    async def _run_training(self, job_id: int, parameters: Dict[str, Any]):
        """Run training job in background."""
        model_type = None
        dataset_path = None
        run_mode = "image"
        try:
            with get_db_session() as db:
                job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
                if not job:
                    return

                if job.status == JobStatus.CANCELLED.value:
                    logger.info("Training skipped for cancelled job", job_id=job_id)
                    return

                model_type = job.model_type
                dataset_path = job.dataset_path
                job.status = JobStatus.RUNNING.value
                job.started_at = datetime.now()
                job.progress = 0.0
                job.error_message = None
                self._reset_epoch_metrics_history(
                    job,
                    total_epochs=int(parameters.get("epochs", job.epochs or 0)),
                )

            if model_type == "lrcn":
                run_mode = "sequence"
            elif dataset_path and self._dataset_has_temporal_media(dataset_path):
                run_mode = "video_hybrid"
            else:
                run_mode = "image"
            self._set_active_job_metadata(
                job_id,
                {
                    "status": JobStatus.RUNNING.value,
                    "mode": run_mode,
                    "total_epochs": int(parameters.get("epochs", 0)),
                    "current_epoch": 0,
                    "current_loss": None,
                    "current_accuracy": None,
                    "message": "开始时序训练"
                    if run_mode == "sequence"
                    else (
                        "开始视频时序融合训练"
                        if run_mode == "video_hybrid"
                        else "开始图像训练"
                    ),
                },
            )

            trainer = (
                self._train_sequence_model
                if model_type == "lrcn"
                else self._train_image_model
            )
            await asyncio.to_thread(trainer, job_id, parameters)

        except TrainingCancelledError:
            logger.info("Training cancelled", job_id=job_id)

            with get_db_session() as db:
                job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
                if job and job.status != JobStatus.CANCELLED.value:
                    job.status = JobStatus.CANCELLED.value
                if job and not job.completed_at:
                    job.completed_at = datetime.now()

            self._set_active_job_metadata(
                job_id,
                {
                    "status": JobStatus.CANCELLED.value,
                    "message": "训练已取消",
                },
            )

        except Exception as e:
            logger.error("Training failed", error=str(e), job_id=job_id)

            with get_db_session() as db:
                job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
                if job:
                    job.status = JobStatus.FAILED.value
                    job.error_message = str(e)
                    job.completed_at = datetime.now()

            self._set_active_job_metadata(
                job_id,
                {
                    "status": JobStatus.FAILED.value,
                    "message": f"训练失败：{str(e)}",
                },
            )

    def _train_image_model(self, job_id: int, parameters: Dict[str, Any]):
        """Train an image classification model and persist epoch progress."""
        with get_db_session() as db:
            job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if not job:
                raise ValueError(f"Training job {job_id} not found")

            dataset_path = job.dataset_path
            model_type = job.model_type
            epochs = int(parameters.get("epochs", job.epochs))
            learning_rate = float(parameters.get("learning_rate", job.learning_rate))
            batch_size = int(parameters.get("batch_size", job.batch_size))
            validation_split = float(parameters.get("validation_split", 0.2))
            early_stopping = bool(parameters.get("early_stopping", True))
            patience = max(1, int(parameters.get("patience", 10)))
            early_stopping_min_delta = max(
                0.0, float(parameters.get("early_stopping_min_delta", 0.002))
            )
            weight_decay = max(0.0, float(parameters.get("weight_decay", 0.0001)))
            label_smoothing = min(
                0.2, max(0.0, float(parameters.get("label_smoothing", 0.05)))
            )
            class_weight_strategy = str(
                parameters.get("class_weight_strategy", "balanced")
            )
            use_official_test_as_validation = bool(
                parameters.get("use_official_test_as_validation", False)
            )
            sequence_length = int(parameters.get("sequence_length", 8))
            frame_stride = int(parameters.get("frame_stride", 2))
            temporal_hidden_size = int(parameters.get("temporal_hidden_size", 256))
            temporal_num_layers = int(parameters.get("temporal_num_layers", 2))
            feature_projection_size = int(
                parameters.get("feature_projection_size", 256)
            )
            face_roi_enabled = bool(parameters.get("face_roi_enabled", True))
            face_roi_confidence_threshold = float(
                parameters.get(
                    "face_roi_confidence_threshold",
                    settings.YOLO_FACE_CONFIDENCE_THRESHOLD,
                )
            )
            face_roi_crop_padding = float(
                parameters.get(
                    "face_roi_crop_padding",
                    settings.YOLO_FACE_CROP_PADDING,
                )
            )
            temporal_bidirectional = bool(
                parameters.get(
                    "temporal_bidirectional", settings.TEMPORAL_BIDIRECTIONAL
                )
            )
            temporal_attention_pooling = bool(
                parameters.get(
                    "temporal_attention_pooling",
                    settings.TEMPORAL_ATTENTION_POOLING,
                )
            )
            train_clip_overlap_ratio = float(
                parameters.get(
                    "train_clip_overlap_ratio",
                    settings.TEMPORAL_TRAIN_CLIP_OVERLAP_RATIO,
                )
            )
            val_clip_overlap_ratio = float(
                parameters.get(
                    "val_clip_overlap_ratio",
                    settings.TEMPORAL_VAL_CLIP_OVERLAP_RATIO,
                )
            )
            temporal_clip_filter_enabled = bool(
                parameters.get(
                    "temporal_clip_filter_enabled",
                    settings.TEMPORAL_CLIP_FILTER_ENABLED,
                )
            )
            temporal_clip_min_frame_std = float(
                parameters.get(
                    "temporal_clip_min_frame_std",
                    settings.TEMPORAL_CLIP_MIN_FRAME_STD,
                )
            )
            temporal_clip_min_motion_delta = float(
                parameters.get(
                    "temporal_clip_min_motion_delta",
                    settings.TEMPORAL_CLIP_MIN_MOTION_DELTA,
                )
            )
            training_device = str(
                parameters.get("training_device", settings.TRAINING_DEVICE)
            )

        sequence_length = max(2, sequence_length)
        frame_stride = max(1, frame_stride)
        temporal_hidden_size = max(32, temporal_hidden_size)
        temporal_num_layers = max(1, temporal_num_layers)
        feature_projection_size = max(32, feature_projection_size)
        face_roi_confidence_threshold = min(
            1.0, max(0.0, face_roi_confidence_threshold)
        )
        face_roi_crop_padding = min(1.0, max(0.0, face_roi_crop_padding))
        train_clip_overlap_ratio = min(0.95, max(0.0, train_clip_overlap_ratio))
        val_clip_overlap_ratio = min(0.95, max(0.0, val_clip_overlap_ratio))
        temporal_clip_min_frame_std = max(0.0, temporal_clip_min_frame_std)
        temporal_clip_min_motion_delta = max(0.0, temporal_clip_min_motion_delta)
        uses_temporal_context = self._dataset_has_temporal_media(dataset_path)
        training_mode = "video_hybrid" if uses_temporal_context else "image"
        preprocessing_progress_end = 25.0
        training_progress_span = 99.0 - preprocessing_progress_end
        preprocessing_reporter = self._create_phase_progress_reporter(
            job_id,
            0.0,
            preprocessing_progress_end,
            epochs,
            current_epoch=0,
        )

        self._set_active_job_metadata(
            job_id,
            {
                "total_epochs": epochs,
                "mode": training_mode,
                "validation_split": validation_split,
                "early_stopping": early_stopping,
                "patience": patience,
                "early_stopping_min_delta": early_stopping_min_delta,
                "weight_decay": weight_decay,
                "label_smoothing": label_smoothing,
                "class_weight_strategy": class_weight_strategy,
                "use_official_test_as_validation": use_official_test_as_validation,
                "face_roi_enabled": face_roi_enabled,
                "face_roi_confidence_threshold": face_roi_confidence_threshold,
                "face_roi_crop_padding": face_roi_crop_padding,
                "temporal_bidirectional": temporal_bidirectional,
                "temporal_attention_pooling": temporal_attention_pooling,
                "train_clip_overlap_ratio": train_clip_overlap_ratio,
                "val_clip_overlap_ratio": val_clip_overlap_ratio,
                "temporal_clip_filter_enabled": temporal_clip_filter_enabled,
                "temporal_clip_min_frame_std": temporal_clip_min_frame_std,
                "temporal_clip_min_motion_delta": temporal_clip_min_motion_delta,
            },
        )

        face_roi_policy = self._build_face_roi_policy(
            {
                "face_roi_enabled": face_roi_enabled,
                "face_roi_confidence_threshold": face_roi_confidence_threshold,
                "face_roi_crop_padding": face_roi_crop_padding,
            }
        )
        aggregation_policy = {
            "video_aggregation_topk_ratio": settings.VIDEO_AGGREGATION_TOPK_RATIO,
            "video_aggregation_mean_weight": settings.VIDEO_AGGREGATION_MEAN_WEIGHT,
            "video_aggregation_peak_weight": settings.VIDEO_AGGREGATION_PEAK_WEIGHT,
            "video_aggregation_persistence_weight": settings.VIDEO_AGGREGATION_PERSISTENCE_WEIGHT,
        }

        device = self._resolve_training_device(training_device)
        self._configure_acceleration_backend(device)
        batch_size = self._tune_batch_size(
            batch_size, device, sequence_mode=uses_temporal_context
        )
        self._set_active_job_metadata(
            job_id,
            {
                "device": str(device),
                "effective_batch_size": batch_size,
                "requested_device": training_device,
                "sequence_length": sequence_length if uses_temporal_context else None,
                "frame_stride": frame_stride if uses_temporal_context else None,
            },
        )
        logger.info(
            "Using training device",
            job_id=job_id,
            device=str(device),
            batch_size=batch_size,
            apple_silicon=self._is_apple_silicon(),
            requested_device=training_device,
            uses_temporal_context=uses_temporal_context,
        )

        if uses_temporal_context:
            preprocessing_reporter(
                0.05,
                f"正在准备视频时序样本（长度 {sequence_length}，步长 {frame_stride}）",
                {"stage": "prepare_temporal_dataset", "unit": "phase"},
            )
            train_loader, val_loader, class_weights, split_metadata = (
                self._build_temporal_dataloaders(
                    dataset_path,
                    batch_size,
                    validation_split,
                    sequence_length,
                    frame_stride,
                    class_weight_strategy=class_weight_strategy,
                    use_official_test_as_validation=use_official_test_as_validation,
                    face_roi_policy=face_roi_policy,
                    train_clip_overlap_ratio=train_clip_overlap_ratio,
                    val_clip_overlap_ratio=val_clip_overlap_ratio,
                    temporal_clip_filter_enabled=temporal_clip_filter_enabled,
                    temporal_clip_min_frame_std=temporal_clip_min_frame_std,
                    temporal_clip_min_motion_delta=temporal_clip_min_motion_delta,
                    progress_callback=preprocessing_reporter,
                )
            )
            model = create_model(
                model_type=model_type,
                num_classes=2,
                pretrained=settings.MODEL_USE_PRETRAINED_WEIGHTS,
                video_temporal_enabled=True,
                temporal_hidden_size=temporal_hidden_size,
                temporal_num_layers=temporal_num_layers,
                feature_projection_size=feature_projection_size,
                temporal_bidirectional=temporal_bidirectional,
                temporal_attention_pooling=temporal_attention_pooling,
            ).to(device)
        else:
            preprocessing_reporter(
                0.05,
                "正在扫描训练数据集并整理样本",
                {"stage": "scan_dataset", "unit": "phase"},
            )
            train_loader, val_loader, class_weights, split_metadata = (
                self._build_dataloaders(
                    dataset_path,
                    batch_size,
                    validation_split,
                    class_weight_strategy=class_weight_strategy,
                    use_official_test_as_validation=use_official_test_as_validation,
                    face_roi_policy=face_roi_policy,
                    progress_callback=preprocessing_reporter,
                )
            )
            model = create_model(
                model_type=model_type,
                num_classes=2,
                pretrained=settings.MODEL_USE_PRETRAINED_WEIGHTS,
            ).to(device)

        setattr(model, "video_aggregation_policy", aggregation_policy)

        preprocessing_reporter(
            1.0,
            "数据集准备完成，开始训练",
            {"stage": "dataset_ready", "current": 1, "total": 1, "unit": "phase"},
        )

        channels_last = device.type == "cuda" and not uses_temporal_context
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        amp_enabled = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        criterion = self._build_training_criterion(
            label_smoothing,
            class_weights=class_weights,
            device=device,
        )
        self._set_active_job_metadata(
            job_id,
            {
                "class_weights": class_weights,
                "majority_downsampling": False,
                "split_strategy": split_metadata.get("split_strategy"),
                "official_test_holdout_count": split_metadata.get(
                    "official_test_holdout_count"
                ),
                "train_subgroup_counts": split_metadata.get("train_subgroup_counts"),
                "val_subgroup_counts": split_metadata.get("val_subgroup_counts"),
                "train_clip_filter": split_metadata.get("train_clip_filter"),
            },
        )
        optimizer = self._build_training_optimizer(
            model,
            learning_rate,
            weight_decay,
        )
        scheduler = self._build_plateau_scheduler(optimizer, learning_rate, patience)

        model_dir = Path(settings.MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)
        best_checkpoint_path = model_dir / f"training_job_{job_id}_best.pth"

        best_val_accuracy = -1.0
        best_val_loss = None
        best_selection_score = -1.0
        best_val_sample_accuracy = None
        best_val_sample_loss = None
        best_val_video_count = None
        best_val_subgroup_metrics = None
        best_val_subgroup_macro_accuracy = None
        best_epoch = 0
        epochs_without_improvement = 0
        epoch_prefix = "视频时序融合" if uses_temporal_context else "Epoch"
        epoch = 0

        for epoch in range(1, epochs + 1):
            if self._is_job_cancelled(job_id):
                raise TrainingCancelledError(
                    f"Training job {job_id} cancelled at epoch {epoch}"
                )

            total_batches = max(1, len(train_loader))

            def image_progress_callback(
                batch_index: int,
                batch_total: int,
                _running_loss: float,
                _running_accuracy: float,
            ):
                incremental_progress = (
                    preprocessing_progress_end
                    + (((epoch - 1) + (batch_index / max(batch_total, 1))) / epochs)
                    * training_progress_span
                )
                self._update_running_progress(
                    job_id,
                    incremental_progress,
                    epoch=epoch,
                    total_epochs=epochs,
                    message=f"{epoch_prefix} {epoch}/{epochs} 训练中（批次 {batch_index}/{batch_total}）",
                )

            epoch_learning_rate = self._get_optimizer_learning_rate(optimizer)
            train_loss, train_accuracy = self._run_train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                amp_enabled=amp_enabled,
                scaler=scaler,
                channels_last=channels_last,
                progress_callback=image_progress_callback,
            )
            self._set_active_job_metadata(
                job_id,
                {
                    "message": f"{epoch_prefix} {epoch}/{epochs} 验证中",
                    "current_epoch": epoch,
                    "total_epochs": epochs,
                },
            )
            val_metrics = self._run_eval_epoch(
                model,
                val_loader,
                criterion,
                device,
                amp_enabled=amp_enabled,
                channels_last=channels_last,
            )
            val_loss = float(val_metrics["loss"])
            val_accuracy = float(val_metrics["accuracy"])
            val_sample_loss = float(val_metrics["sample_loss"])
            val_sample_accuracy = float(val_metrics["sample_accuracy"])
            val_video_count = int(val_metrics["video_count"])
            val_subgroup_metrics = val_metrics.get("subgroup_metrics")
            val_subgroup_macro_accuracy = val_metrics.get("subgroup_macro_accuracy")
            checkpoint_selection_score = float(
                val_metrics.get("selection_score", val_accuracy)
            )

            progress = preprocessing_progress_end + (
                (epoch / epochs) * training_progress_span
            )
            self._update_epoch_metrics(
                job_id,
                progress,
                val_accuracy,
                val_loss,
                sample_accuracy=val_sample_accuracy,
                sample_loss=val_sample_loss,
                video_count=val_video_count,
                subgroup_metrics=val_subgroup_metrics,
                subgroup_macro_accuracy=val_subgroup_macro_accuracy,
                checkpoint_selection_score=checkpoint_selection_score,
                epoch=epoch,
                total_epochs=epochs,
                message=f"{epoch_prefix} {epoch}/{epochs} completed",
            )
            self._record_epoch_metrics(
                job_id,
                epoch,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
                epoch_learning_rate,
                val_sample_loss=val_sample_loss,
                val_sample_accuracy=val_sample_accuracy,
                val_video_count=val_video_count,
                val_subgroup_metrics=val_subgroup_metrics,
                val_subgroup_macro_accuracy=val_subgroup_macro_accuracy,
                checkpoint_selection_score=checkpoint_selection_score,
            )

            if scheduler is not None:
                scheduler.step(val_loss)

            logger.info(
                "Training epoch completed",
                job_id=job_id,
                epoch=epoch,
                total_epochs=epochs,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                progress=progress,
            )

            improved = self._is_validation_improved(
                checkpoint_selection_score,
                val_accuracy,
                val_loss,
                best_selection_score,
                best_val_accuracy,
                best_val_loss,
                early_stopping_min_delta,
            )

            if improved:
                best_selection_score = checkpoint_selection_score
                best_val_accuracy = val_accuracy
                best_val_loss = val_loss
                best_val_sample_accuracy = val_sample_accuracy
                best_val_sample_loss = val_sample_loss
                best_val_video_count = val_video_count
                best_val_subgroup_metrics = val_subgroup_metrics
                best_val_subgroup_macro_accuracy = val_subgroup_macro_accuracy
                best_epoch = epoch
                epochs_without_improvement = 0
                self._set_active_job_metadata(
                    job_id,
                    {
                        "message": f"{epoch_prefix} {epoch}/{epochs} 正在保存最佳模型",
                        "current_epoch": epoch,
                        "total_epochs": epochs,
                        "best_epoch": best_epoch,
                        "epochs_trained": epoch,
                        "val_sample_loss": val_sample_loss,
                        "val_sample_accuracy": val_sample_accuracy,
                        "val_video_count": val_video_count,
                        "val_subgroup_metrics": val_subgroup_metrics,
                        "val_subgroup_macro_accuracy": val_subgroup_macro_accuracy,
                        "checkpoint_selection_score": checkpoint_selection_score,
                    },
                )
                checkpoint_payload = {
                    "job_id": job_id,
                    "model_type": model_type,
                    "epoch": epoch,
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "validation_split": validation_split,
                    "training_device": training_device,
                    "weight_decay": weight_decay,
                    "label_smoothing": label_smoothing,
                    "class_weight_strategy": class_weight_strategy,
                    "class_weights": class_weights,
                    "majority_downsampling": False,
                    "use_official_test_as_validation": use_official_test_as_validation,
                    "split_strategy": split_metadata.get("split_strategy"),
                    "official_test_holdout_count": split_metadata.get(
                        "official_test_holdout_count"
                    ),
                    "train_subgroup_counts": split_metadata.get(
                        "train_subgroup_counts"
                    ),
                    "val_subgroup_counts": split_metadata.get("val_subgroup_counts"),
                    "early_stopping": early_stopping,
                    "patience": patience,
                    "early_stopping_min_delta": early_stopping_min_delta,
                    "num_classes": 2,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss,
                    "val_sample_accuracy": val_sample_accuracy,
                    "val_sample_loss": val_sample_loss,
                    "val_video_count": val_video_count,
                    "val_subgroup_metrics": val_subgroup_metrics,
                    "val_subgroup_macro_accuracy": val_subgroup_macro_accuracy,
                    "checkpoint_selection_score": checkpoint_selection_score,
                    "best_epoch": best_epoch,
                    "epochs_trained": epoch,
                    "input_size": settings.MODEL_INPUT_SIZE,
                    "video_temporal_enabled": uses_temporal_context,
                    "training_mode": training_mode,
                    "saved_at": datetime.now().isoformat(),
                }
                if uses_temporal_context:
                    checkpoint_payload.update(
                        {
                            "sequence_length": sequence_length,
                            "frame_stride": frame_stride,
                            "temporal_hidden_size": temporal_hidden_size,
                            "temporal_num_layers": temporal_num_layers,
                            "feature_projection_size": feature_projection_size,
                            "temporal_bidirectional": temporal_bidirectional,
                            "temporal_attention_pooling": temporal_attention_pooling,
                        }
                    )
                checkpoint_payload.update(
                    {
                        "face_roi_enabled": face_roi_enabled,
                        "yolo_face_model_path": settings.YOLO_FACE_MODEL_PATH,
                        "face_roi_confidence_threshold": face_roi_confidence_threshold,
                        "face_roi_crop_padding": face_roi_crop_padding,
                        "face_roi_policy_version": face_roi_policy[
                            "face_roi_policy_version"
                        ],
                        "face_roi_selection_policy": face_roi_policy[
                            "face_roi_selection_policy"
                        ],
                        "video_aggregation_topk_ratio": aggregation_policy[
                            "video_aggregation_topk_ratio"
                        ],
                        "video_aggregation_mean_weight": aggregation_policy[
                            "video_aggregation_mean_weight"
                        ],
                        "video_aggregation_peak_weight": aggregation_policy[
                            "video_aggregation_peak_weight"
                        ],
                        "video_aggregation_persistence_weight": aggregation_policy[
                            "video_aggregation_persistence_weight"
                        ],
                        "train_clip_overlap_ratio": train_clip_overlap_ratio,
                        "val_clip_overlap_ratio": val_clip_overlap_ratio,
                        "train_clip_filter": split_metadata.get("train_clip_filter"),
                        "temporal_clip_filter_enabled": temporal_clip_filter_enabled,
                        "temporal_clip_min_frame_std": temporal_clip_min_frame_std,
                        "temporal_clip_min_motion_delta": temporal_clip_min_motion_delta,
                    }
                )
                torch.save(checkpoint_payload, str(best_checkpoint_path))
            else:
                epochs_without_improvement += 1
                self._set_active_job_metadata(
                    job_id,
                    {
                        "best_epoch": best_epoch or None,
                        "epochs_trained": epoch,
                        "val_sample_loss": val_sample_loss,
                        "val_sample_accuracy": val_sample_accuracy,
                        "val_video_count": val_video_count,
                        "val_subgroup_metrics": val_subgroup_metrics,
                        "val_subgroup_macro_accuracy": val_subgroup_macro_accuracy,
                        "checkpoint_selection_score": checkpoint_selection_score,
                    },
                )

            if (
                early_stopping
                and epochs_without_improvement >= patience
                and epoch < epochs
            ):
                logger.info(
                    "Early stopping triggered for image training",
                    job_id=job_id,
                    epoch=epoch,
                    patience=patience,
                    best_epoch=best_epoch,
                    best_val_accuracy=best_val_accuracy,
                    best_val_loss=best_val_loss,
                )
                self._set_active_job_metadata(
                    job_id,
                    {
                        "message": f"{epoch_prefix} {epoch}/{epochs} 验证指标连续 {patience} 轮未提升，提前停止",
                        "current_epoch": epoch,
                        "total_epochs": epochs,
                        "best_epoch": best_epoch or None,
                        "epochs_trained": epoch,
                    },
                )
                break

        self._finalize_successful_training(
            job_id,
            best_val_accuracy,
            best_val_loss,
            best_checkpoint_path,
            epochs_trained=epoch,
            best_epoch=best_epoch or None,
            stopped_early=early_stopping and epoch < epochs,
            val_sample_accuracy=best_val_sample_accuracy,
            val_sample_loss=best_val_sample_loss,
            val_video_count=best_val_video_count,
            val_subgroup_metrics=best_val_subgroup_metrics,
            val_subgroup_macro_accuracy=best_val_subgroup_macro_accuracy,
            checkpoint_selection_score=best_selection_score
            if best_selection_score >= 0
            else None,
        )

    def _train_sequence_model(self, job_id: int, parameters: Dict[str, Any]):
        """Train the LRCN model using frame-sequence clips."""
        with get_db_session() as db:
            job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if not job:
                raise ValueError(f"Training job {job_id} not found")

            dataset_path = job.dataset_path
            model_type = job.model_type
            epochs = int(parameters.get("epochs", job.epochs))
            learning_rate = float(parameters.get("learning_rate", job.learning_rate))
            batch_size = int(parameters.get("batch_size", job.batch_size))
            validation_split = float(parameters.get("validation_split", 0.2))
            early_stopping = bool(parameters.get("early_stopping", True))
            patience = max(1, int(parameters.get("patience", 10)))
            early_stopping_min_delta = max(
                0.0, float(parameters.get("early_stopping_min_delta", 0.002))
            )
            weight_decay = max(0.0, float(parameters.get("weight_decay", 0.0001)))
            label_smoothing = min(
                0.2, max(0.0, float(parameters.get("label_smoothing", 0.05)))
            )
            class_weight_strategy = str(
                parameters.get("class_weight_strategy", "balanced")
            )
            use_official_test_as_validation = bool(
                parameters.get("use_official_test_as_validation", False)
            )
            sequence_length = int(parameters.get("sequence_length", 16))
            frame_stride = int(parameters.get("frame_stride", 1))
            hidden_size = int(parameters.get("hidden_size", 256))
            num_layers = int(parameters.get("num_layers", 1))
            frame_projection_size = int(parameters.get("frame_projection_size", 128))
            face_roi_enabled = bool(parameters.get("face_roi_enabled", True))
            face_roi_confidence_threshold = float(
                parameters.get(
                    "face_roi_confidence_threshold",
                    settings.YOLO_FACE_CONFIDENCE_THRESHOLD,
                )
            )
            face_roi_crop_padding = float(
                parameters.get(
                    "face_roi_crop_padding",
                    settings.YOLO_FACE_CROP_PADDING,
                )
            )
            train_clip_overlap_ratio = float(
                parameters.get(
                    "train_clip_overlap_ratio",
                    settings.TEMPORAL_TRAIN_CLIP_OVERLAP_RATIO,
                )
            )
            val_clip_overlap_ratio = float(
                parameters.get(
                    "val_clip_overlap_ratio",
                    settings.TEMPORAL_VAL_CLIP_OVERLAP_RATIO,
                )
            )
            temporal_clip_filter_enabled = bool(
                parameters.get(
                    "temporal_clip_filter_enabled",
                    settings.TEMPORAL_CLIP_FILTER_ENABLED,
                )
            )
            temporal_clip_min_frame_std = float(
                parameters.get(
                    "temporal_clip_min_frame_std",
                    settings.TEMPORAL_CLIP_MIN_FRAME_STD,
                )
            )
            temporal_clip_min_motion_delta = float(
                parameters.get(
                    "temporal_clip_min_motion_delta",
                    settings.TEMPORAL_CLIP_MIN_MOTION_DELTA,
                )
            )
            training_device = str(
                parameters.get("training_device", settings.TRAINING_DEVICE)
            )

        if sequence_length <= 0:
            raise ValueError("sequence_length must be greater than 0")
        if frame_stride <= 0:
            raise ValueError("frame_stride must be greater than 0")
        hidden_size = max(64, hidden_size)
        num_layers = max(1, num_layers)
        frame_projection_size = max(32, frame_projection_size)
        face_roi_confidence_threshold = min(
            1.0, max(0.0, face_roi_confidence_threshold)
        )
        face_roi_crop_padding = min(1.0, max(0.0, face_roi_crop_padding))
        train_clip_overlap_ratio = min(0.95, max(0.0, train_clip_overlap_ratio))
        val_clip_overlap_ratio = min(0.95, max(0.0, val_clip_overlap_ratio))
        temporal_clip_min_frame_std = max(0.0, temporal_clip_min_frame_std)
        temporal_clip_min_motion_delta = max(0.0, temporal_clip_min_motion_delta)
        preprocessing_progress_end = 25.0
        training_progress_span = 99.0 - preprocessing_progress_end
        preprocessing_reporter = self._create_phase_progress_reporter(
            job_id,
            0.0,
            preprocessing_progress_end,
            epochs,
            current_epoch=0,
        )

        self._set_active_job_metadata(
            job_id,
            {
                "total_epochs": epochs,
                "mode": "sequence",
                "validation_split": validation_split,
                "early_stopping": early_stopping,
                "patience": patience,
                "early_stopping_min_delta": early_stopping_min_delta,
                "weight_decay": weight_decay,
                "label_smoothing": label_smoothing,
                "class_weight_strategy": class_weight_strategy,
                "use_official_test_as_validation": use_official_test_as_validation,
                "face_roi_enabled": face_roi_enabled,
                "face_roi_confidence_threshold": face_roi_confidence_threshold,
                "face_roi_crop_padding": face_roi_crop_padding,
                "train_clip_overlap_ratio": train_clip_overlap_ratio,
                "val_clip_overlap_ratio": val_clip_overlap_ratio,
                "temporal_clip_filter_enabled": temporal_clip_filter_enabled,
                "temporal_clip_min_frame_std": temporal_clip_min_frame_std,
                "temporal_clip_min_motion_delta": temporal_clip_min_motion_delta,
            },
        )
        face_roi_policy = self._build_face_roi_policy(
            {
                "face_roi_enabled": face_roi_enabled,
                "face_roi_confidence_threshold": face_roi_confidence_threshold,
                "face_roi_crop_padding": face_roi_crop_padding,
            }
        )
        aggregation_policy = {
            "video_aggregation_topk_ratio": settings.VIDEO_AGGREGATION_TOPK_RATIO,
            "video_aggregation_mean_weight": settings.VIDEO_AGGREGATION_MEAN_WEIGHT,
            "video_aggregation_peak_weight": settings.VIDEO_AGGREGATION_PEAK_WEIGHT,
            "video_aggregation_persistence_weight": settings.VIDEO_AGGREGATION_PERSISTENCE_WEIGHT,
        }
        preprocessing_reporter(
            0.05,
            f"正在准备视频片段（长度 {sequence_length}，步长 {frame_stride}）",
            {"stage": "prepare_sequence_dataset", "unit": "phase"},
        )

        device = self._resolve_training_device(training_device)
        self._configure_acceleration_backend(device)
        batch_size = self._tune_batch_size(batch_size, device, sequence_mode=True)
        self._set_active_job_metadata(
            job_id,
            {
                "device": str(device),
                "effective_batch_size": batch_size,
                "requested_device": training_device,
            },
        )
        logger.info(
            "Using sequence training device",
            job_id=job_id,
            device=str(device),
            batch_size=batch_size,
            apple_silicon=self._is_apple_silicon(),
            requested_device=training_device,
        )
        train_loader, val_loader, class_weights, split_metadata = (
            self._build_sequence_dataloaders(
                dataset_path,
                batch_size,
                validation_split,
                sequence_length,
                frame_stride,
                class_weight_strategy=class_weight_strategy,
                use_official_test_as_validation=use_official_test_as_validation,
                face_roi_policy=face_roi_policy,
                train_clip_overlap_ratio=train_clip_overlap_ratio,
                val_clip_overlap_ratio=val_clip_overlap_ratio,
                temporal_clip_filter_enabled=temporal_clip_filter_enabled,
                temporal_clip_min_frame_std=temporal_clip_min_frame_std,
                temporal_clip_min_motion_delta=temporal_clip_min_motion_delta,
                progress_callback=preprocessing_reporter,
            )
        )
        preprocessing_reporter(
            1.0,
            "视频片段准备完成，开始训练",
            {
                "stage": "sequence_dataset_ready",
                "current": 1,
                "total": 1,
                "unit": "phase",
            },
        )

        model = create_model(
            model_type=model_type,
            num_classes=2,
            input_size=25088,
            hidden_size=hidden_size,
            num_layers=num_layers,
            frame_projection_size=frame_projection_size,
            pretrained=settings.MODEL_USE_PRETRAINED_WEIGHTS,
        ).to(device)
        setattr(model, "video_aggregation_policy", aggregation_policy)
        amp_enabled = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        criterion = self._build_training_criterion(
            label_smoothing,
            class_weights=class_weights,
            device=device,
        )
        self._set_active_job_metadata(
            job_id,
            {
                "class_weights": class_weights,
                "majority_downsampling": False,
                "split_strategy": split_metadata.get("split_strategy"),
                "official_test_holdout_count": split_metadata.get(
                    "official_test_holdout_count"
                ),
                "train_subgroup_counts": split_metadata.get("train_subgroup_counts"),
                "val_subgroup_counts": split_metadata.get("val_subgroup_counts"),
                "train_clip_filter": split_metadata.get("train_clip_filter"),
            },
        )
        optimizer = self._build_training_optimizer(
            model,
            learning_rate,
            weight_decay,
        )
        scheduler = self._build_plateau_scheduler(optimizer, learning_rate, patience)

        model_dir = Path(settings.MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)
        best_checkpoint_path = model_dir / f"training_job_{job_id}_best.pth"

        best_val_accuracy = -1.0
        best_val_loss = None
        best_selection_score = -1.0
        best_val_sample_accuracy = None
        best_val_sample_loss = None
        best_val_video_count = None
        best_val_subgroup_metrics = None
        best_val_subgroup_macro_accuracy = None
        best_epoch = 0
        epochs_without_improvement = 0
        epoch = 0

        for epoch in range(1, epochs + 1):
            if self._is_job_cancelled(job_id):
                raise TrainingCancelledError(
                    f"Training job {job_id} cancelled at epoch {epoch}"
                )

            total_batches = max(1, len(train_loader))

            def sequence_progress_callback(
                batch_index: int,
                batch_total: int,
                _running_loss: float,
                _running_accuracy: float,
            ):
                incremental_progress = (
                    preprocessing_progress_end
                    + (((epoch - 1) + (batch_index / max(batch_total, 1))) / epochs)
                    * training_progress_span
                )
                self._update_running_progress(
                    job_id,
                    incremental_progress,
                    epoch=epoch,
                    total_epochs=epochs,
                    message=f"Sequence epoch {epoch}/{epochs} 训练中（批次 {batch_index}/{batch_total}）",
                )

            epoch_learning_rate = self._get_optimizer_learning_rate(optimizer)
            train_loss, train_accuracy = self._run_train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                amp_enabled=amp_enabled,
                scaler=scaler,
                progress_callback=sequence_progress_callback,
            )
            self._set_active_job_metadata(
                job_id,
                {
                    "message": f"Sequence epoch {epoch}/{epochs} 验证中",
                    "current_epoch": epoch,
                    "total_epochs": epochs,
                },
            )
            val_metrics = self._run_eval_epoch(
                model,
                val_loader,
                criterion,
                device,
                amp_enabled=amp_enabled,
            )
            val_loss = float(val_metrics["loss"])
            val_accuracy = float(val_metrics["accuracy"])
            val_sample_loss = float(val_metrics["sample_loss"])
            val_sample_accuracy = float(val_metrics["sample_accuracy"])
            val_video_count = int(val_metrics["video_count"])
            val_subgroup_metrics = val_metrics.get("subgroup_metrics")
            val_subgroup_macro_accuracy = val_metrics.get("subgroup_macro_accuracy")
            checkpoint_selection_score = float(
                val_metrics.get("selection_score", val_accuracy)
            )

            progress = preprocessing_progress_end + (
                (epoch / epochs) * training_progress_span
            )
            self._update_epoch_metrics(
                job_id,
                progress,
                val_accuracy,
                val_loss,
                sample_accuracy=val_sample_accuracy,
                sample_loss=val_sample_loss,
                video_count=val_video_count,
                subgroup_metrics=val_subgroup_metrics,
                subgroup_macro_accuracy=val_subgroup_macro_accuracy,
                checkpoint_selection_score=checkpoint_selection_score,
                epoch=epoch,
                total_epochs=epochs,
                message=f"Sequence epoch {epoch}/{epochs} completed",
            )
            self._record_epoch_metrics(
                job_id,
                epoch,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
                epoch_learning_rate,
                val_sample_loss=val_sample_loss,
                val_sample_accuracy=val_sample_accuracy,
                val_video_count=val_video_count,
                val_subgroup_metrics=val_subgroup_metrics,
                val_subgroup_macro_accuracy=val_subgroup_macro_accuracy,
                checkpoint_selection_score=checkpoint_selection_score,
            )

            if scheduler is not None:
                scheduler.step(val_loss)

            logger.info(
                "Sequence training epoch completed",
                job_id=job_id,
                epoch=epoch,
                total_epochs=epochs,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                progress=progress,
                sequence_length=sequence_length,
                frame_stride=frame_stride,
            )

            improved = self._is_validation_improved(
                checkpoint_selection_score,
                val_accuracy,
                val_loss,
                best_selection_score,
                best_val_accuracy,
                best_val_loss,
                early_stopping_min_delta,
            )

            if improved:
                best_selection_score = checkpoint_selection_score
                best_val_accuracy = val_accuracy
                best_val_loss = val_loss
                best_val_sample_accuracy = val_sample_accuracy
                best_val_sample_loss = val_sample_loss
                best_val_video_count = val_video_count
                best_val_subgroup_metrics = val_subgroup_metrics
                best_val_subgroup_macro_accuracy = val_subgroup_macro_accuracy
                best_epoch = epoch
                epochs_without_improvement = 0
                self._set_active_job_metadata(
                    job_id,
                    {
                        "message": f"Sequence epoch {epoch}/{epochs} 正在保存最佳模型",
                        "current_epoch": epoch,
                        "total_epochs": epochs,
                        "best_epoch": best_epoch,
                        "epochs_trained": epoch,
                        "val_sample_loss": val_sample_loss,
                        "val_sample_accuracy": val_sample_accuracy,
                        "val_video_count": val_video_count,
                        "val_subgroup_metrics": val_subgroup_metrics,
                        "val_subgroup_macro_accuracy": val_subgroup_macro_accuracy,
                        "checkpoint_selection_score": checkpoint_selection_score,
                    },
                )
                torch.save(
                    {
                        "job_id": job_id,
                        "model_type": model_type,
                        "epoch": epoch,
                        "epochs": epochs,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "validation_split": validation_split,
                        "training_device": training_device,
                        "weight_decay": weight_decay,
                        "label_smoothing": label_smoothing,
                        "class_weight_strategy": class_weight_strategy,
                        "class_weights": class_weights,
                        "majority_downsampling": False,
                        "use_official_test_as_validation": use_official_test_as_validation,
                        "split_strategy": split_metadata.get("split_strategy"),
                        "official_test_holdout_count": split_metadata.get(
                            "official_test_holdout_count"
                        ),
                        "train_subgroup_counts": split_metadata.get(
                            "train_subgroup_counts"
                        ),
                        "val_subgroup_counts": split_metadata.get(
                            "val_subgroup_counts"
                        ),
                        "early_stopping": early_stopping,
                        "patience": patience,
                        "early_stopping_min_delta": early_stopping_min_delta,
                        "num_classes": 2,
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "feature_input_size": 25088,
                        "frame_projection_size": frame_projection_size,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_accuracy": val_accuracy,
                        "val_loss": val_loss,
                        "val_sample_accuracy": val_sample_accuracy,
                        "val_sample_loss": val_sample_loss,
                        "val_video_count": val_video_count,
                        "val_subgroup_metrics": val_subgroup_metrics,
                        "val_subgroup_macro_accuracy": val_subgroup_macro_accuracy,
                        "checkpoint_selection_score": checkpoint_selection_score,
                        "best_epoch": best_epoch,
                        "epochs_trained": epoch,
                        "input_size": settings.MODEL_INPUT_SIZE,
                        "sequence_length": sequence_length,
                        "frame_stride": frame_stride,
                        "video_temporal_enabled": True,
                        "training_mode": "sequence_temporal",
                        "face_roi_enabled": face_roi_enabled,
                        "yolo_face_model_path": settings.YOLO_FACE_MODEL_PATH,
                        "face_roi_confidence_threshold": face_roi_confidence_threshold,
                        "face_roi_crop_padding": face_roi_crop_padding,
                        "face_roi_policy_version": face_roi_policy[
                            "face_roi_policy_version"
                        ],
                        "face_roi_selection_policy": face_roi_policy[
                            "face_roi_selection_policy"
                        ],
                        "video_aggregation_topk_ratio": aggregation_policy[
                            "video_aggregation_topk_ratio"
                        ],
                        "video_aggregation_mean_weight": aggregation_policy[
                            "video_aggregation_mean_weight"
                        ],
                        "video_aggregation_peak_weight": aggregation_policy[
                            "video_aggregation_peak_weight"
                        ],
                        "video_aggregation_persistence_weight": aggregation_policy[
                            "video_aggregation_persistence_weight"
                        ],
                        "train_clip_overlap_ratio": train_clip_overlap_ratio,
                        "val_clip_overlap_ratio": val_clip_overlap_ratio,
                        "train_clip_filter": split_metadata.get("train_clip_filter"),
                        "temporal_clip_filter_enabled": temporal_clip_filter_enabled,
                        "temporal_clip_min_frame_std": temporal_clip_min_frame_std,
                        "temporal_clip_min_motion_delta": temporal_clip_min_motion_delta,
                        "saved_at": datetime.now().isoformat(),
                    },
                    str(best_checkpoint_path),
                )
            else:
                epochs_without_improvement += 1
                self._set_active_job_metadata(
                    job_id,
                    {
                        "best_epoch": best_epoch or None,
                        "epochs_trained": epoch,
                        "val_sample_loss": val_sample_loss,
                        "val_sample_accuracy": val_sample_accuracy,
                        "val_video_count": val_video_count,
                        "val_subgroup_metrics": val_subgroup_metrics,
                        "val_subgroup_macro_accuracy": val_subgroup_macro_accuracy,
                        "checkpoint_selection_score": checkpoint_selection_score,
                    },
                )

            if (
                early_stopping
                and epochs_without_improvement >= patience
                and epoch < epochs
            ):
                logger.info(
                    "Early stopping triggered for sequence training",
                    job_id=job_id,
                    epoch=epoch,
                    patience=patience,
                    best_epoch=best_epoch,
                    best_val_accuracy=best_val_accuracy,
                    best_val_loss=best_val_loss,
                )
                self._set_active_job_metadata(
                    job_id,
                    {
                        "message": f"Sequence epoch {epoch}/{epochs} 验证指标连续 {patience} 轮未提升，提前停止",
                        "current_epoch": epoch,
                        "total_epochs": epochs,
                        "best_epoch": best_epoch or None,
                        "epochs_trained": epoch,
                    },
                )
                break

        self._finalize_successful_training(
            job_id,
            best_val_accuracy,
            best_val_loss,
            best_checkpoint_path,
            epochs_trained=epoch,
            best_epoch=best_epoch or None,
            stopped_early=early_stopping and epoch < epochs,
            val_sample_accuracy=best_val_sample_accuracy,
            val_sample_loss=best_val_sample_loss,
            val_video_count=best_val_video_count,
            val_subgroup_metrics=best_val_subgroup_metrics,
            val_subgroup_macro_accuracy=best_val_subgroup_macro_accuracy,
            checkpoint_selection_score=best_selection_score
            if best_selection_score >= 0
            else None,
        )

    def _finalize_successful_training(
        self,
        job_id: int,
        best_val_accuracy: float,
        best_val_loss: Optional[float],
        best_checkpoint_path: Path,
        epochs_trained: Optional[int] = None,
        best_epoch: Optional[int] = None,
        stopped_early: bool = False,
        val_sample_accuracy: Optional[float] = None,
        val_sample_loss: Optional[float] = None,
        val_video_count: Optional[int] = None,
        val_subgroup_metrics: Optional[Dict[str, Any]] = None,
        val_subgroup_macro_accuracy: Optional[float] = None,
        checkpoint_selection_score: Optional[float] = None,
    ):
        """Persist final job state after successful training."""
        with get_db_session() as db:
            job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if not job:
                return
            if job.status == JobStatus.CANCELLED.value:
                job.completed_at = job.completed_at or datetime.now()
                return

            job.status = JobStatus.COMPLETED.value
            job.progress = 100.0
            job.completed_at = datetime.now()
            job.accuracy = best_val_accuracy if best_val_accuracy >= 0 else None
            job.loss = best_val_loss
            job.model_path = str(best_checkpoint_path)

        self._set_active_job_metadata(
            job_id,
            {
                "status": JobStatus.COMPLETED.value,
                "progress": 100.0,
                "current_epoch": None,
                "total_epochs": None,
                "current_loss": best_val_loss,
                "current_accuracy": best_val_accuracy
                if best_val_accuracy >= 0
                else None,
                "message": (
                    f"训练已提前停止并完成，最佳轮次 {best_epoch}，可决定是否保留模型文件"
                    if stopped_early and best_epoch
                    else "训练完成，可决定是否保留模型文件"
                ),
                "epochs_trained": epochs_trained,
                "best_epoch": best_epoch,
                "val_sample_accuracy": val_sample_accuracy,
                "val_sample_loss": val_sample_loss,
                "val_video_count": val_video_count,
                "val_subgroup_metrics": val_subgroup_metrics,
                "val_subgroup_macro_accuracy": val_subgroup_macro_accuracy,
                "checkpoint_selection_score": checkpoint_selection_score,
            },
        )

    def _build_dataloaders(
        self,
        dataset_path: str,
        batch_size: int,
        validation_split: float,
        class_weight_strategy: str = "balanced",
        use_official_test_as_validation: bool = False,
        face_roi_policy: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[
            Callable[[float, str, Optional[Dict[str, Any]]], None]
        ] = None,
    ) -> Tuple[DataLoader, DataLoader, Optional[List[float]], Dict[str, Any]]:
        """Build train/validation dataloaders from image dataset path."""
        root_path = Path(dataset_path).expanduser()
        if not root_path.exists() or not root_path.is_dir():
            raise ValueError(
                f"Dataset path does not exist or is not a directory: {dataset_path}"
            )

        if progress_callback is not None:
            progress_callback(
                0.05,
                "正在扫描数据集目录",
                {"stage": "scan_dataset", "unit": "phase"},
            )

        input_size = int(settings.MODEL_INPUT_SIZE)
        train_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.12,
                            contrast=0.12,
                            saturation=0.08,
                            hue=0.02,
                        )
                    ],
                    p=0.35,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        split_dirs = self._find_split_dirs(root_path)
        split_metadata: Dict[str, Any] = {
            "split_strategy": "explicit_train_val_dirs"
            if split_dirs
            else "internal_label_validation",
            "official_test_holdout_count": 0,
        }
        if split_dirs:
            if progress_callback is not None:
                progress_callback(
                    0.15,
                    "检测到 train/val 划分，正在收集训练样本",
                    {"stage": "collect_train_samples", "unit": "phase"},
                )
            train_samples = self._collect_labeled_media_samples(
                split_dirs["train"], split_dirs["train"]
            )
            if progress_callback is not None:
                progress_callback(
                    0.25,
                    "正在收集验证样本",
                    {"stage": "collect_val_samples", "unit": "phase"},
                )
            val_samples = self._collect_labeled_media_samples(
                split_dirs["val"], split_dirs["val"]
            )
            self._ensure_label_coverage(
                [sample.label for sample in train_samples],
                [sample.label for sample in val_samples],
                "Image dataset split",
            )
        else:
            if progress_callback is not None:
                progress_callback(
                    0.15,
                    "正在收集图像与视频源",
                    {"stage": "collect_media_sources", "unit": "phase"},
                )
            image_sources = self._collect_labeled_images(root_path, root_path)
            video_sources = self._collect_labeled_videos(root_path, root_path)
            total_sources = len(image_sources) + len(video_sources)
            if total_sources < 2:
                raise ValueError(
                    "Dataset must contain at least 2 labeled image/video sources for train/validation split"
                )

            validation_split = max(0.0, min(0.9, validation_split))

            source_entries: List[Dict[str, Any]] = [
                {"kind": "image", "sample": sample} for sample in image_sources
            ]
            source_entries.extend(
                {
                    "kind": "video",
                    "path": video_path,
                    "label": label,
                }
                for video_path, label in video_sources
            )

            def entry_media_path(entry: Dict[str, Any]) -> Optional[str]:
                if entry.get("kind") == "image":
                    sample = entry.get("sample")
                    return sample.path if sample is not None else None
                return entry.get("path")

            source_entries, official_test_entries = (
                self._partition_items_by_official_test_list(
                    source_entries,
                    root_path,
                    entry_media_path,
                    "image_video_sources",
                )
            )
            if len(source_entries) < 2:
                raise ValueError(
                    "Celeb-DF official test split leaves fewer than 2 train/validation sources"
                )
            split_metadata["official_test_holdout_count"] = len(official_test_entries)
            if official_test_entries and use_official_test_as_validation:
                train_entries = source_entries
                val_entries = official_test_entries
                split_metadata["split_strategy"] = "official_test_as_validation"
            else:
                try:
                    train_entries, val_entries = self._stratified_split_items_by_group(
                        source_entries,
                        validation_split,
                        lambda entry: self._infer_validation_subgroup_from_path(
                            entry_media_path(entry),
                            root_path,
                        ),
                        "Image/video subgroup split",
                    )
                    split_metadata["split_strategy"] = "internal_subgroup_validation"
                except ValueError:
                    train_entries, val_entries = self._stratified_split_items(
                        source_entries,
                        validation_split,
                        lambda entry: int(
                            entry["sample"].label
                            if entry.get("kind") == "image"
                            else entry["label"]
                        ),
                        "Image/video source split",
                    )
                    split_metadata["split_strategy"] = "internal_label_validation"

            split_metadata["train_subgroup_counts"] = dict(
                Counter(
                    self._infer_validation_subgroup_from_path(
                        entry_media_path(entry),
                        root_path,
                    )
                    for entry in train_entries
                )
            )
            split_metadata["val_subgroup_counts"] = dict(
                Counter(
                    self._infer_validation_subgroup_from_path(
                        entry_media_path(entry),
                        root_path,
                    )
                    for entry in val_entries
                )
            )
            if progress_callback is not None:
                progress_callback(
                    0.3,
                    "正在展开训练集视频采样帧",
                    {"stage": "expand_train_video_samples", "unit": "phase"},
                )
            train_samples = self._expand_media_sources(train_entries)
            if progress_callback is not None:
                progress_callback(
                    0.4,
                    "正在展开验证集视频采样帧",
                    {"stage": "expand_val_video_samples", "unit": "phase"},
                )
            val_samples = self._expand_media_sources(val_entries)

        if not train_samples:
            raise ValueError("No labeled training samples found")
        if not val_samples:
            raise ValueError("No labeled validation samples found")

        train_class_weights = self._compute_class_weights(
            [sample.label for sample in train_samples],
            strategy=class_weight_strategy,
        )

        self._ensure_label_coverage(
            [sample.label for sample in train_samples],
            [sample.label for sample in val_samples],
            "Image dataset samples",
        )
        self._log_label_distribution(
            "image_train_samples", [sample.label for sample in train_samples]
        )
        self._log_label_distribution(
            "image_val_samples", [sample.label for sample in val_samples]
        )

        train_cache_progress = self._chain_progress_reporter(
            progress_callback, 0.45, 0.75
        )
        val_cache_progress = self._chain_progress_reporter(
            progress_callback, 0.75, 0.92
        )
        train_samples = self._materialize_video_backed_samples(
            train_samples,
            input_size,
            progress_callback=train_cache_progress,
            progress_label="正在缓存训练集视频帧",
        )
        val_samples = self._materialize_video_backed_samples(
            val_samples,
            input_size,
            progress_callback=val_cache_progress,
            progress_label="正在缓存验证集视频帧",
        )

        if progress_callback is not None:
            progress_callback(
                0.96,
                "正在构建 DataLoader",
                {"stage": "build_dataloader", "unit": "phase"},
            )

        train_dataset = ImageClassificationDataset(
            train_samples,
            transform=train_transform,
            roi_processor=self._face_roi_processor,
            roi_policy=face_roi_policy,
        )
        val_dataset = ImageClassificationDataset(
            val_samples,
            transform=val_transform,
            include_source_id=True,
            roi_processor=self._face_roi_processor,
            roi_policy=face_roi_policy,
        )
        train_sampler = self._build_balanced_sampler(
            [sample.label for sample in train_samples],
            "image_train_samples",
        )

        loader_kwargs = self._dataloader_kwargs(len(train_dataset))
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            **loader_kwargs,
        )
        val_loader_kwargs = self._dataloader_kwargs(len(val_dataset))
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            **val_loader_kwargs,
        )

        if progress_callback is not None:
            progress_callback(
                1.0,
                f"数据集准备完成：训练样本 {len(train_dataset)}，验证样本 {len(val_dataset)}",
                {
                    "stage": "dataset_ready",
                    "current": len(train_dataset) + len(val_dataset),
                    "total": len(train_dataset) + len(val_dataset),
                    "unit": "samples",
                },
            )

        return train_loader, val_loader, train_class_weights, split_metadata

    def _build_sequence_dataloaders(
        self,
        dataset_path: str,
        batch_size: int,
        validation_split: float,
        sequence_length: int,
        frame_stride: int,
        class_weight_strategy: str = "balanced",
        use_official_test_as_validation: bool = False,
        face_roi_policy: Optional[Dict[str, Any]] = None,
        train_clip_overlap_ratio: float = 0.7,
        val_clip_overlap_ratio: float = 0.35,
        temporal_clip_filter_enabled: bool = True,
        temporal_clip_min_frame_std: float = 6.0,
        temporal_clip_min_motion_delta: float = 2.0,
        progress_callback: Optional[
            Callable[[float, str, Optional[Dict[str, Any]]], None]
        ] = None,
    ) -> Tuple[DataLoader, DataLoader, Optional[List[float]], Dict[str, Any]]:
        """Build train/validation dataloaders for frame sequence clips."""
        return self._build_temporal_dataloaders(
            dataset_path,
            batch_size,
            validation_split,
            sequence_length,
            frame_stride,
            class_weight_strategy=class_weight_strategy,
            use_official_test_as_validation=use_official_test_as_validation,
            face_roi_policy=face_roi_policy,
            train_clip_overlap_ratio=train_clip_overlap_ratio,
            val_clip_overlap_ratio=val_clip_overlap_ratio,
            temporal_clip_filter_enabled=temporal_clip_filter_enabled,
            temporal_clip_min_frame_std=temporal_clip_min_frame_std,
            temporal_clip_min_motion_delta=temporal_clip_min_motion_delta,
            progress_callback=progress_callback,
        )

    def _build_temporal_dataloaders(
        self,
        dataset_path: str,
        batch_size: int,
        validation_split: float,
        sequence_length: int,
        frame_stride: int,
        class_weight_strategy: str = "balanced",
        use_official_test_as_validation: bool = False,
        face_roi_policy: Optional[Dict[str, Any]] = None,
        train_clip_overlap_ratio: float = 0.7,
        val_clip_overlap_ratio: float = 0.35,
        temporal_clip_filter_enabled: bool = True,
        temporal_clip_min_frame_std: float = 6.0,
        temporal_clip_min_motion_delta: float = 2.0,
        progress_callback: Optional[
            Callable[[float, str, Optional[Dict[str, Any]]], None]
        ] = None,
    ) -> Tuple[DataLoader, DataLoader, Optional[List[float]], Dict[str, Any]]:
        """Build temporal dataloaders from mixed image/video datasets."""
        root_path = Path(dataset_path).expanduser()
        if not root_path.exists() or not root_path.is_dir():
            raise ValueError(
                f"Dataset path does not exist or is not a directory: {dataset_path}"
            )

        if progress_callback is not None:
            progress_callback(
                0.05,
                "正在扫描时序数据集目录",
                {"stage": "scan_temporal_dataset", "unit": "phase"},
            )

        input_size = int(settings.MODEL_INPUT_SIZE)
        train_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        split_dirs = self._find_split_dirs(root_path)
        split_metadata: Dict[str, Any] = {
            "split_strategy": "explicit_train_val_dirs"
            if split_dirs
            else "internal_label_validation",
            "official_test_holdout_count": 0,
        }
        if split_dirs:
            if progress_callback is not None:
                progress_callback(
                    0.15,
                    "检测到 train/val 划分，正在收集视频源",
                    {"stage": "collect_train_temporal_sources", "unit": "phase"},
                )
            train_sources = self._collect_temporal_training_sources(
                split_dirs["train"], split_dirs["train"]
            )
            if progress_callback is not None:
                progress_callback(
                    0.28,
                    "正在收集验证视频源",
                    {"stage": "collect_val_temporal_sources", "unit": "phase"},
                )
            val_sources = self._collect_temporal_training_sources(
                split_dirs["val"], split_dirs["val"]
            )
            self._ensure_label_coverage(
                [source.label for source in train_sources],
                [source.label for source in val_sources],
                "Temporal source split",
            )
        else:
            if progress_callback is not None:
                progress_callback(
                    0.15,
                    "正在收集图像/视频时序源",
                    {"stage": "collect_temporal_sources", "unit": "phase"},
                )
            all_sources = self._collect_temporal_training_sources(root_path, root_path)
            if len(all_sources) < 2:
                raise ValueError(
                    "Dataset must contain at least 2 labeled image/video sources for train/validation split"
                )

            validation_split = max(0.0, min(0.9, validation_split))
            all_sources, official_test_sources = (
                self._partition_items_by_official_test_list(
                    all_sources,
                    root_path,
                    lambda source: (
                        source.path
                        or (source.frame_paths[0] if source.frame_paths else None)
                    ),
                    "temporal_sources",
                )
            )
            if len(all_sources) < 2:
                raise ValueError(
                    "Celeb-DF official test split leaves fewer than 2 temporal sources for train/validation"
                )
            split_metadata["official_test_holdout_count"] = len(official_test_sources)
            if official_test_sources and use_official_test_as_validation:
                train_sources = all_sources
                val_sources = official_test_sources
                split_metadata["split_strategy"] = "official_test_as_validation"
            else:
                try:
                    train_sources, val_sources = self._stratified_split_items_by_group(
                        all_sources,
                        validation_split,
                        lambda source: self._infer_validation_subgroup_from_path(
                            source.path
                            or (source.frame_paths[0] if source.frame_paths else None),
                            root_path,
                        ),
                        "Temporal subgroup split",
                    )
                    split_metadata["split_strategy"] = "internal_subgroup_validation"
                except ValueError:
                    train_sources, val_sources = self._stratified_split_items(
                        all_sources,
                        validation_split,
                        lambda source: int(source.label),
                        "Temporal source split",
                    )
                    split_metadata["split_strategy"] = "internal_label_validation"

            split_metadata["train_subgroup_counts"] = dict(
                Counter(
                    self._infer_validation_subgroup_from_path(
                        source.path
                        or (source.frame_paths[0] if source.frame_paths else None),
                        root_path,
                    )
                    for source in train_sources
                )
            )
            split_metadata["val_subgroup_counts"] = dict(
                Counter(
                    self._infer_validation_subgroup_from_path(
                        source.path
                        or (source.frame_paths[0] if source.frame_paths else None),
                        root_path,
                    )
                    for source in val_sources
                )
            )

        self._log_label_distribution(
            "temporal_train_sources", [source.label for source in train_sources]
        )
        self._log_label_distribution(
            "temporal_val_sources", [source.label for source in val_sources]
        )
        train_class_weights = self._compute_class_weights(
            [source.label for source in train_sources],
            strategy=class_weight_strategy,
        )

        train_clip_budget = self._resolve_temporal_clip_budget(
            sequence_length, training=True
        )
        val_clip_budget = self._resolve_temporal_clip_budget(
            sequence_length, training=False
        )

        train_clip_progress = self._chain_progress_reporter(
            progress_callback, 0.35, 0.72
        )
        val_clip_progress = self._chain_progress_reporter(progress_callback, 0.72, 0.93)
        train_samples = self._expand_temporal_sources_to_clips(
            train_sources,
            sequence_length,
            frame_stride,
            input_size,
            face_roi_policy=face_roi_policy,
            progress_callback=train_clip_progress,
            progress_label="正在生成训练集时序片段",
            max_clips_per_source=train_clip_budget,
            clip_overlap_ratio=train_clip_overlap_ratio,
        )
        val_samples = self._expand_temporal_sources_to_clips(
            val_sources,
            sequence_length,
            frame_stride,
            input_size,
            face_roi_policy=face_roi_policy,
            progress_callback=val_clip_progress,
            progress_label="正在生成验证集时序片段",
            max_clips_per_source=val_clip_budget,
            clip_overlap_ratio=val_clip_overlap_ratio,
        )

        clip_filter_metadata = {
            "kept": len(train_samples),
            "dropped": 0,
            "min_frame_std": float(temporal_clip_min_frame_std),
            "min_motion_delta": float(temporal_clip_min_motion_delta),
        }
        if temporal_clip_filter_enabled:
            filter_progress = self._chain_progress_reporter(
                progress_callback, 0.93, 0.96
            )
            train_samples, clip_filter_metadata = self._filter_temporal_clip_samples(
                train_samples,
                min_frame_std=temporal_clip_min_frame_std,
                min_motion_delta=temporal_clip_min_motion_delta,
                progress_callback=filter_progress,
            )
        split_metadata["train_clip_filter"] = clip_filter_metadata

        if not train_samples:
            raise ValueError("No labeled training clips found")
        if not val_samples:
            raise ValueError("No labeled validation clips found")

        self._ensure_label_coverage(
            [sample.label for sample in train_samples],
            [sample.label for sample in val_samples],
            "Temporal clip split",
        )
        self._log_label_distribution(
            "temporal_train_clips",
            [sample.label for sample in train_samples],
        )
        self._log_label_distribution(
            "temporal_val_clips",
            [sample.label for sample in val_samples],
        )

        train_dataset = SequenceClassificationDataset(
            train_samples,
            transform=train_transform,
            augment=True,
            roi_processor=self._face_roi_processor,
            roi_policy=face_roi_policy,
        )
        val_dataset = SequenceClassificationDataset(
            val_samples,
            transform=val_transform,
            include_source_id=True,
            roi_processor=self._face_roi_processor,
            roi_policy=face_roi_policy,
        )
        train_sampler = self._build_balanced_sampler(
            [sample.label for sample in train_samples],
            "temporal_train_clips",
        )

        if progress_callback is not None:
            progress_callback(
                0.96,
                "正在构建时序 DataLoader",
                {"stage": "build_temporal_dataloader", "unit": "phase"},
            )

        loader_kwargs = self._dataloader_kwargs(len(train_dataset))
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            **loader_kwargs,
        )
        val_loader_kwargs = self._dataloader_kwargs(len(val_dataset))
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            **val_loader_kwargs,
        )

        if progress_callback is not None:
            progress_callback(
                1.0,
                f"时序数据准备完成：训练片段 {len(train_dataset)}，验证片段 {len(val_dataset)}",
                {
                    "stage": "temporal_dataset_ready",
                    "current": len(train_dataset) + len(val_dataset),
                    "total": len(train_dataset) + len(val_dataset),
                    "unit": "clips",
                    "clip_filter_kept": clip_filter_metadata["kept"],
                    "clip_filter_dropped": clip_filter_metadata["dropped"],
                },
            )

        return train_loader, val_loader, train_class_weights, split_metadata

    def _dataset_has_temporal_media(self, dataset_path: str) -> bool:
        """Return True when the dataset contains video files or frame-group videos."""
        root_path = Path(dataset_path).expanduser()
        if not root_path.exists() or not root_path.is_dir():
            return False

        split_dirs = self._find_split_dirs(root_path)
        candidate_roots = list(split_dirs.values()) if split_dirs else [root_path]
        for candidate_root in candidate_roots:
            if self._collect_labeled_videos(candidate_root, candidate_root):
                return True
            if self._collect_labeled_video_groups(candidate_root, candidate_root):
                return True
        return False

    def _collect_temporal_training_sources(
        self, root_path: Path, label_root: Path
    ) -> List[TemporalMediaSource]:
        """Collect temporal training units without splitting a single video across sets."""
        grouped_videos = self._collect_labeled_video_groups(root_path, label_root)
        grouped_frame_paths = {
            frame_path
            for frame_paths, _label in grouped_videos
            for frame_path in frame_paths
        }

        sources: List[TemporalMediaSource] = [
            TemporalMediaSource(
                kind="frame_group",
                label=label,
                frame_paths=frame_paths,
                source_id=str(Path(frame_paths[0]).parent.resolve()),
            )
            for frame_paths, label in grouped_videos
        ]

        standalone_images = [
            sample
            for sample in self._collect_labeled_images(root_path, label_root)
            if sample.path not in grouped_frame_paths
        ]
        sources.extend(
            TemporalMediaSource(
                kind="image",
                label=sample.label,
                path=sample.path,
                source_id=sample.source_id or sample.path,
            )
            for sample in standalone_images
        )
        sources.extend(
            TemporalMediaSource(
                kind="video",
                label=label,
                path=video_path,
                source_id=str(Path(video_path).resolve()),
            )
            for video_path, label in self._collect_labeled_videos(root_path, label_root)
        )
        return sources

    def _expand_temporal_sources_to_clips(
        self,
        sources: List[TemporalMediaSource],
        sequence_length: int,
        frame_stride: int,
        input_size: int,
        face_roi_policy: Optional[Dict[str, Any]] = None,
        max_clips_per_source: Optional[int] = None,
        clip_overlap_ratio: float = 0.5,
        progress_callback: Optional[
            Callable[[float, str, Optional[Dict[str, Any]]], None]
        ] = None,
        progress_label: str = "正在生成时序片段",
    ) -> List[TemporalClipSample]:
        """Expand mixed temporal sources into fixed-length clips."""
        clips: List[TemporalClipSample] = []
        total_sources = len(sources)
        progress_interval = max(1, total_sources // 20) if total_sources else 1

        for index, source in enumerate(sources, start=1):
            source_path = source.path or (
                source.frame_paths[0] if source.frame_paths else ""
            )
            source_name = Path(source_path).name if source_path else f"source-{index}"
            source_id = source.source_id or source_path or f"source-{index}"
            if source.kind == "image" and source.path:
                clips.append(
                    TemporalClipSample(
                        frame_paths=[source.path] * sequence_length,
                        label=source.label,
                        source_id=source_id,
                        roi_materialized=False,
                    )
                )
            elif source.kind == "frame_group" and source.frame_paths:
                clips.extend(
                    self._build_clips_from_video_frames(
                        source.frame_paths,
                        source.label,
                        source_id,
                        sequence_length,
                        frame_stride,
                        max_clips=max_clips_per_source,
                        overlap_ratio=clip_overlap_ratio,
                    )
                )
            elif source.kind == "video" and source.path:
                source_progress = self._chain_progress_reporter(
                    progress_callback,
                    (index - 1) / max(total_sources, 1),
                    index / max(total_sources, 1),
                )
                clips.extend(
                    self._build_clips_from_raw_video(
                        source.path,
                        source.label,
                        source_id,
                        sequence_length,
                        frame_stride,
                        input_size,
                        face_roi_policy=face_roi_policy,
                        max_clips=max_clips_per_source,
                        overlap_ratio=clip_overlap_ratio,
                        progress_callback=source_progress,
                        progress_label=f"{progress_label}：{source_name}",
                    )
                )
                continue

            if progress_callback and (
                index == 1 or index % progress_interval == 0 or index == total_sources
            ):
                self._report_indexed_progress(
                    progress_callback,
                    index,
                    total_sources,
                    f"{progress_label}：{index}/{total_sources}（当前 {source_name}）",
                    stage="expand_temporal_sources",
                    unit="sources",
                )
        return clips

    def _collect_labeled_video_groups(
        self, root_path: Path, label_root: Path
    ) -> List[Tuple[List[str], int]]:
        """Collect frame paths grouped by video directory and label."""
        grouped_frames: Dict[str, Dict[str, Any]] = {}

        for file_path in root_path.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue

            label = self._infer_label_from_path(file_path, label_root)
            if label is None:
                continue

            if not self._is_valid_image(file_path):
                logger.warning("Skipping invalid image file", path=str(file_path))
                continue

            try:
                relative_parent = file_path.parent.relative_to(label_root)
            except ValueError:
                relative_parent = file_path.parent

            if len(relative_parent.parts) < 2:
                continue

            group_key = str(file_path.parent.resolve())
            if group_key not in grouped_frames:
                grouped_frames[group_key] = {"label": label, "frames": []}
            grouped_frames[group_key]["frames"].append(str(file_path))

        videos: List[Tuple[List[str], int]] = []
        for group_data in grouped_frames.values():
            sorted_frames = sorted(group_data["frames"], key=self._natural_sort_key)
            if sorted_frames:
                videos.append((sorted_frames, group_data["label"]))

        return videos

    def _build_clips_from_raw_video(
        self,
        video_path: str,
        label: int,
        source_id: str,
        sequence_length: int,
        frame_stride: int,
        input_size: int,
        face_roi_policy: Optional[Dict[str, Any]] = None,
        max_clips: Optional[int] = None,
        overlap_ratio: float = 0.5,
        progress_callback: Optional[
            Callable[[float, str, Optional[Dict[str, Any]]], None]
        ] = None,
        progress_label: str = "正在从原始视频取帧",
    ) -> List[TemporalClipSample]:
        """Build cached clip samples directly from a raw video file."""
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            logger.warning(
                "Skipping unreadable video file for temporal clips", path=video_path
            )
            return []

        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            capture.release()

        cache_root = self._video_frame_cache_root(input_size)
        cache_root.mkdir(parents=True, exist_ok=True)
        clips: List[TemporalClipSample] = []
        sampled_clips = self._sample_video_clip_indices(
            frame_count,
            sequence_length,
            frame_stride,
            max_clips=max_clips,
            overlap_ratio=overlap_ratio,
        )
        total_clips = len(sampled_clips)
        progress_interval = max(1, total_clips // 20) if total_clips else 1

        for clip_index, clip_indices in enumerate(sampled_clips, start=1):
            frame_paths: List[str] = []
            for frame_index in clip_indices:
                try:
                    frame_paths.append(
                        self._materialize_video_frame(
                            video_path,
                            frame_index,
                            cache_root,
                            input_size,
                            face_roi_policy=face_roi_policy,
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "Skipping broken temporal clip frame",
                        video_path=video_path,
                        frame_index=frame_index,
                        error=str(exc),
                    )
                    frame_paths = []
                    break

            if frame_paths:
                clips.append(
                    TemporalClipSample(
                        frame_paths=frame_paths,
                        label=label,
                        source_id=source_id,
                        roi_materialized=bool(face_roi_policy),
                    )
                )

            if progress_callback and (
                clip_index == 1
                or clip_index % progress_interval == 0
                or clip_index == total_clips
            ):
                self._report_indexed_progress(
                    progress_callback,
                    clip_index,
                    total_clips,
                    f"{progress_label}：片段 {clip_index}/{total_clips}",
                    stage="extract_video_clips",
                    unit="clips",
                )

        return clips

    def _sample_video_clip_indices(
        self,
        frame_count: int,
        sequence_length: int,
        frame_stride: int,
        max_clips: Optional[int] = None,
        overlap_ratio: float = 0.5,
    ) -> List[List[int]]:
        """Sample temporally ordered frame index clips from a raw video."""
        target_length = max(1, sequence_length)
        target_stride = max(1, frame_stride)
        clip_span = ((target_length - 1) * target_stride) + 1

        def build_clip(start_index: int) -> List[int]:
            max_index = max(frame_count - 1, 0)
            return [
                min(max_index, max(0, start_index + (idx * target_stride)))
                for idx in range(target_length)
            ]

        if frame_count <= 0:
            return [[0] * target_length]

        if frame_count <= clip_span:
            return [build_clip(0)]

        available_span = max(0, frame_count - clip_span)
        if available_span == 0:
            start_positions = [0]
        else:
            overlap_ratio = min(max(overlap_ratio, 0.0), 0.9)
            window_step = max(1, int(round(clip_span * (1.0 - overlap_ratio))))
            start_positions = list(range(0, available_span + 1, window_step))
            if not start_positions or start_positions[-1] != available_span:
                start_positions.append(available_span)
            start_positions = self._sample_evenly_spaced_items(
                sorted(set(start_positions)), max_clips
            )

        return [build_clip(start_index) for start_index in start_positions]

    def _build_clips_from_video_frames(
        self,
        frame_paths: List[str],
        label: int,
        source_id: str,
        sequence_length: int,
        frame_stride: int,
        max_clips: Optional[int] = None,
        overlap_ratio: float = 0.5,
    ) -> List[TemporalClipSample]:
        """Create fixed-length clips from sorted frame paths."""
        if not frame_paths:
            return []

        clip_span = ((sequence_length - 1) * frame_stride) + 1
        clips: List[TemporalClipSample] = []

        if len(frame_paths) < sequence_length:
            padded = frame_paths + [frame_paths[-1]] * (
                sequence_length - len(frame_paths)
            )
            return [
                TemporalClipSample(
                    frame_paths=padded,
                    label=label,
                    source_id=source_id,
                    roi_materialized=False,
                )
            ]

        if len(frame_paths) < clip_span:
            clip = [
                frame_paths[min(i * frame_stride, len(frame_paths) - 1)]
                for i in range(sequence_length)
            ]
            return [
                TemporalClipSample(
                    frame_paths=clip,
                    label=label,
                    source_id=source_id,
                    roi_materialized=False,
                )
            ]

        available_span = len(frame_paths) - clip_span
        overlap_ratio = min(max(overlap_ratio, 0.0), 0.9)
        window_step = max(1, int(round(clip_span * (1.0 - overlap_ratio))))
        start_positions = list(range(0, available_span + 1, window_step))
        if not start_positions or start_positions[-1] != available_span:
            start_positions.append(available_span)
        start_positions = self._sample_evenly_spaced_items(
            sorted(set(start_positions)), max_clips
        )

        for start in start_positions:
            clip = []
            for i in range(sequence_length):
                idx = start + (i * frame_stride)
                if idx >= len(frame_paths):
                    idx = len(frame_paths) - 1
                clip.append(frame_paths[idx])
            clips.append(
                TemporalClipSample(
                    frame_paths=clip,
                    label=label,
                    source_id=source_id,
                    roi_materialized=False,
                )
            )

        return clips

    def _natural_sort_key(self, file_path: str) -> List[Any]:
        """Sort helper that keeps numeric frame indexes in order."""
        name = Path(file_path).name.lower()
        return [
            int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)
        ]

    def _find_split_dirs(self, root_path: Path) -> Optional[Dict[str, Path]]:
        """Detect explicit train/val split directories."""
        child_dirs = {
            child.name.lower(): child for child in root_path.iterdir() if child.is_dir()
        }

        train_dir = child_dirs.get("train")
        val_dir = (
            child_dirs.get("val")
            or child_dirs.get("valid")
            or child_dirs.get("validation")
        )

        if train_dir and val_dir:
            return {"train": train_dir, "val": val_dir}
        return None

    def _collect_labeled_media_samples(
        self, root_path: Path, label_root: Path
    ) -> List[MediaClassificationSample]:
        """Collect labeled image samples and sampled raw-video frames."""
        samples = self._collect_labeled_images(root_path, label_root)
        video_sources = self._collect_labeled_videos(root_path, label_root)
        samples.extend(self._expand_video_sources(video_sources))
        return samples

    def _expand_media_sources(
        self, sources: List[Dict[str, Any]]
    ) -> List[MediaClassificationSample]:
        """Expand mixed image/video sources into trainable frame samples."""
        samples: List[MediaClassificationSample] = []
        for source in sources:
            if source.get("kind") == "image":
                sample = source.get("sample")
                if sample is not None:
                    samples.append(sample)
            elif source.get("kind") == "video":
                video_path = source.get("path")
                label = source.get("label")
                if video_path is not None and label is not None:
                    samples.extend(
                        self._expand_video_sources([(str(video_path), int(label))])
                    )
        return samples

    def _collect_labeled_images(
        self, root_path: Path, label_root: Path
    ) -> List[MediaClassificationSample]:
        """Collect valid image files and infer labels from directory names."""
        samples: List[MediaClassificationSample] = []

        for file_path in root_path.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue

            label = self._infer_label_from_path(file_path, label_root)
            if label is None:
                continue

            if not self._is_valid_image(file_path):
                logger.warning("Skipping invalid image file", path=str(file_path))
                continue

            resolved_path = str(file_path.resolve())
            samples.append(
                MediaClassificationSample(
                    path=resolved_path,
                    label=label,
                    source_id=resolved_path,
                )
            )

        return samples

    def _collect_labeled_videos(
        self, root_path: Path, label_root: Path
    ) -> List[Tuple[str, int]]:
        """Collect labeled raw video files for frame sampling."""
        videos: List[Tuple[str, int]] = []

        for file_path in root_path.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
                continue

            label = self._infer_label_from_path(file_path, label_root)
            if label is None:
                continue

            if not self._is_valid_video(file_path):
                logger.warning("Skipping invalid video file", path=str(file_path))
                continue

            videos.append((str(file_path.resolve()), label))

        return videos

    def _expand_video_sources(
        self, video_sources: List[Tuple[str, int]]
    ) -> List[MediaClassificationSample]:
        """Expand raw videos into sampled frame-level training items."""
        samples: List[MediaClassificationSample] = []
        max_frames = max(1, int(settings.MAX_FRAMES_PER_VIDEO))

        for video_path, label in video_sources:
            capture = cv2.VideoCapture(video_path)
            if not capture.isOpened():
                logger.warning("Skipping unreadable video file", path=video_path)
                continue

            try:
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if frame_count <= 0:
                    success, frame = capture.read()
                    if not success or frame is None:
                        logger.warning(
                            "Skipping video without readable frames", path=video_path
                        )
                        continue
                    frame_indices = [0]
                else:
                    frame_indices = self._sample_video_frame_indices(
                        frame_count, max_frames
                    )
            finally:
                capture.release()

            for frame_index in frame_indices:
                samples.append(
                    MediaClassificationSample(
                        path=video_path,
                        label=label,
                        frame_index=frame_index,
                        source_id=str(Path(video_path).resolve()),
                    )
                )

        return samples

    def _sample_video_frame_indices(
        self, frame_count: int, max_samples: int
    ) -> List[int]:
        """Evenly sample frame indices across a video."""
        if frame_count <= 1:
            return [0]

        sample_count = max(1, min(frame_count, max_samples))
        if sample_count == 1:
            return [frame_count // 2]

        step = (frame_count - 1) / float(sample_count - 1)
        indices = sorted(
            {
                min(frame_count - 1, max(0, int(round(step * idx))))
                for idx in range(sample_count)
            }
        )
        return indices or [0]

    def _infer_label_from_path(self, file_path: Path, root_path: Path) -> Optional[int]:
        """Infer class label from parent directory keywords."""
        try:
            relative_parents = list(file_path.relative_to(root_path).parents)
        except ValueError:
            relative_parents = list(file_path.parents)

        for parent in relative_parents:
            if str(parent) in {"", "."}:
                continue

            name = parent.name.lower()
            is_fake = self._matches_keyword(name, FAKE_LABEL_KEYWORDS)
            is_real = self._matches_keyword(name, REAL_LABEL_KEYWORDS)

            if is_fake and not is_real:
                return 0
            if is_real and not is_fake:
                return 1

        return None

    def _matches_keyword(self, value: str, keywords: set) -> bool:
        """Check if a directory name matches any label keyword."""
        normalized = "".join(ch if ch.isalnum() else " " for ch in value.lower())
        tokens = [token for token in normalized.split() if token]

        for keyword in keywords:
            if keyword in tokens:
                return True
            if keyword in normalized:
                return True
        return False

    def _is_valid_image(self, path: Path) -> bool:
        """Return True if PIL can open and verify the image."""
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except (UnidentifiedImageError, OSError, ValueError):
            return False

    def _is_valid_video(self, path: Path) -> bool:
        """Return True if OpenCV can open the video and read at least one frame."""
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return False

        try:
            success, frame = capture.read()
            return bool(success and frame is not None)
        finally:
            capture.release()

    def _run_train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        amp_enabled: bool = False,
        scaler: Optional[Any] = None,
        channels_last: bool = False,
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
    ) -> Tuple[float, float]:
        """Run one training epoch."""
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        total_batches = max(1, len(dataloader))
        progress_interval = max(1, total_batches // 20)

        for batch_index, (images, labels) in enumerate(dataloader, start=1):
            images = images.to(device, non_blocking=device.type == "cuda")
            if channels_last and images.ndim == 4:
                images = images.contiguous(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=device.type == "cuda")

            optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp_enabled
                else nullcontext()
            )
            with autocast_context:
                outputs = model(images)
                loss = criterion(outputs, labels)

            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            current_batch_size = labels.size(0)
            total_loss += loss.item() * current_batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += current_batch_size

            if progress_callback and (
                batch_index == 1
                or batch_index % progress_interval == 0
                or batch_index == total_batches
            ):
                running_loss = total_loss / max(total_samples, 1)
                running_accuracy = total_correct / max(total_samples, 1)
                progress_callback(
                    batch_index,
                    total_batches,
                    running_loss,
                    running_accuracy,
                )

        average_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        return average_loss, accuracy

    def _update_running_progress(
        self,
        job_id: int,
        progress: float,
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
        message: Optional[str] = None,
    ):
        """Persist in-epoch progress updates so UI does not stay at 0%."""
        clamped_progress = self._clamp_incomplete_progress(progress)
        with get_db_session() as db:
            job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if not job:
                return

            if job.status == JobStatus.CANCELLED.value:
                raise TrainingCancelledError(f"Training job {job_id} cancelled")

            current_progress = float(job.progress or 0.0)
            if clamped_progress > current_progress:
                job.progress = clamped_progress

        active_update: Dict[str, Any] = {"progress": clamped_progress}
        if epoch is not None:
            active_update["current_epoch"] = epoch
        if total_epochs is not None:
            active_update["total_epochs"] = total_epochs
        if message is not None:
            active_update["message"] = message
        self._set_active_job_metadata(job_id, active_update)

    def _clamp_incomplete_progress(self, progress: float) -> float:
        """Reserve 100% for jobs that are fully finalized."""
        return min(max(progress, 0.0), 99.0)

    def _summarize_logit_bucket(
        self,
        logits_list: List[torch.Tensor],
        labels_list: List[int],
        criterion: nn.Module,
        device: torch.device,
    ) -> Dict[str, float]:
        if not logits_list or not labels_list:
            return {}

        logits = torch.stack(logits_list, dim=0).to(device)
        labels = torch.tensor(labels_list, dtype=torch.long, device=device)
        loss = float(criterion(logits, labels).item())
        accuracy = float((logits.argmax(dim=1) == labels).float().mean().item())
        return {
            "count": int(labels.shape[0]),
            "loss": loss,
            "accuracy": accuracy,
        }

    def _run_eval_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        amp_enabled: bool = False,
        channels_last: bool = False,
    ) -> Dict[str, Any]:
        """Run one validation epoch."""
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        grouped_logits: Dict[str, List[torch.Tensor]] = {}
        grouped_labels: Dict[str, int] = {}
        sample_subgroup_logits: Dict[str, List[torch.Tensor]] = {}
        sample_subgroup_labels: Dict[str, List[int]] = {}
        aggregation_policy = getattr(model, "video_aggregation_policy", None) or {
            "video_aggregation_topk_ratio": settings.VIDEO_AGGREGATION_TOPK_RATIO,
            "video_aggregation_mean_weight": settings.VIDEO_AGGREGATION_MEAN_WEIGHT,
            "video_aggregation_peak_weight": settings.VIDEO_AGGREGATION_PEAK_WEIGHT,
            "video_aggregation_persistence_weight": settings.VIDEO_AGGREGATION_PERSISTENCE_WEIGHT,
        }

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    images, labels, source_ids = batch
                else:
                    images, labels = batch
                    source_ids = None

                images = images.to(device, non_blocking=device.type == "cuda")
                if channels_last and images.ndim == 4:
                    images = images.contiguous(memory_format=torch.channels_last)
                labels = labels.to(device, non_blocking=device.type == "cuda")

                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if amp_enabled
                    else nullcontext()
                )
                with autocast_context:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                current_batch_size = labels.size(0)
                total_loss += loss.item() * current_batch_size
                total_correct += (outputs.argmax(dim=1) == labels).sum().item()
                total_samples += current_batch_size

                if source_ids is not None:
                    detached_outputs = outputs.detach().float().cpu()
                    detached_labels = labels.detach().cpu()
                    for item_index, source_id in enumerate(source_ids):
                        source_key = str(source_id)
                        subgroup = self._infer_validation_subgroup_from_path(source_key)
                        grouped_logits.setdefault(source_key, []).append(
                            detached_outputs[item_index]
                        )
                        label_value = int(detached_labels[item_index].item())
                        sample_subgroup_logits.setdefault(subgroup, []).append(
                            detached_outputs[item_index]
                        )
                        sample_subgroup_labels.setdefault(subgroup, []).append(
                            label_value
                        )
                        if (
                            source_key in grouped_labels
                            and grouped_labels[source_key] != label_value
                        ):
                            raise ValueError(
                                f"Inconsistent validation labels for source {source_key}"
                            )
                        grouped_labels[source_key] = label_value

        sample_loss = total_loss / max(total_samples, 1)
        sample_accuracy = total_correct / max(total_samples, 1)

        subgroup_metrics: Dict[str, Dict[str, Any]] = {}
        for subgroup, logits_list in sample_subgroup_logits.items():
            summary = self._summarize_logit_bucket(
                logits_list,
                sample_subgroup_labels.get(subgroup, []),
                criterion,
                device,
            )
            if summary:
                subgroup_metrics[subgroup] = {
                    "sample_count": summary["count"],
                    "sample_loss": summary["loss"],
                    "sample_accuracy": summary["accuracy"],
                    "video_count": 0,
                    "video_loss": None,
                    "video_accuracy": None,
                }

        if grouped_logits:
            ordered_source_ids = sorted(grouped_logits)
            video_subgroup_logits: Dict[str, List[torch.Tensor]] = {}
            video_subgroup_labels: Dict[str, List[int]] = {}
            aggregated_logits = torch.stack(
                [
                    aggregate_logit_sequence(
                        grouped_logits[source_id],
                        confidence_threshold=settings.CONFIDENCE_THRESHOLD,
                        topk_ratio=aggregation_policy["video_aggregation_topk_ratio"],
                        mean_weight=aggregation_policy["video_aggregation_mean_weight"],
                        peak_weight=aggregation_policy["video_aggregation_peak_weight"],
                        persistence_weight=aggregation_policy[
                            "video_aggregation_persistence_weight"
                        ],
                        device=device,
                    )[0]
                    for source_id in ordered_source_ids
                ],
                dim=0,
            ).to(device)
            aggregated_labels = torch.tensor(
                [grouped_labels[source_id] for source_id in ordered_source_ids],
                dtype=torch.long,
                device=device,
            )
            video_loss = float(criterion(aggregated_logits, aggregated_labels).item())
            video_accuracy = float(
                (aggregated_logits.argmax(dim=1) == aggregated_labels)
                .float()
                .mean()
                .item()
            )

            for row_index, source_id in enumerate(ordered_source_ids):
                subgroup = self._infer_validation_subgroup_from_path(source_id)
                video_subgroup_logits.setdefault(subgroup, []).append(
                    aggregated_logits[row_index].detach().cpu()
                )
                video_subgroup_labels.setdefault(subgroup, []).append(
                    int(aggregated_labels[row_index].item())
                )

            for subgroup, logits_list in video_subgroup_logits.items():
                summary = self._summarize_logit_bucket(
                    logits_list,
                    video_subgroup_labels.get(subgroup, []),
                    criterion,
                    device,
                )
                if not summary:
                    continue
                subgroup_entry = subgroup_metrics.setdefault(
                    subgroup,
                    {
                        "sample_count": 0,
                        "sample_loss": None,
                        "sample_accuracy": None,
                    },
                )
                subgroup_entry["video_count"] = summary["count"]
                subgroup_entry["video_loss"] = summary["loss"]
                subgroup_entry["video_accuracy"] = summary["accuracy"]

            subgroup_macro_accuracy = self._compute_subgroup_macro_accuracy(
                subgroup_metrics
            )
            selection_score = (
                subgroup_macro_accuracy
                if subgroup_macro_accuracy is not None
                else video_accuracy
            )
            return {
                "loss": video_loss,
                "accuracy": video_accuracy,
                "sample_loss": sample_loss,
                "sample_accuracy": sample_accuracy,
                "video_count": len(ordered_source_ids),
                "subgroup_metrics": subgroup_metrics or None,
                "subgroup_macro_accuracy": subgroup_macro_accuracy,
                "selection_score": selection_score,
            }

        return {
            "loss": sample_loss,
            "accuracy": sample_accuracy,
            "sample_loss": sample_loss,
            "sample_accuracy": sample_accuracy,
            "video_count": total_samples,
            "subgroup_metrics": subgroup_metrics or None,
            "subgroup_macro_accuracy": self._compute_subgroup_macro_accuracy(
                subgroup_metrics
            ),
            "selection_score": self._compute_subgroup_macro_accuracy(subgroup_metrics)
            or sample_accuracy,
        }

    def _update_epoch_metrics(
        self,
        job_id: int,
        progress: float,
        accuracy: float,
        loss: float,
        sample_accuracy: Optional[float] = None,
        sample_loss: Optional[float] = None,
        video_count: Optional[int] = None,
        subgroup_metrics: Optional[Dict[str, Any]] = None,
        subgroup_macro_accuracy: Optional[float] = None,
        checkpoint_selection_score: Optional[float] = None,
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
        message: Optional[str] = None,
    ):
        """Persist epoch metrics to database and active metadata."""
        clamped_progress = self._clamp_incomplete_progress(progress)
        with get_db_session() as db:
            job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if not job:
                return

            if job.status == JobStatus.CANCELLED.value:
                raise TrainingCancelledError(f"Training job {job_id} cancelled")

            job.progress = clamped_progress
            job.accuracy = accuracy
            job.loss = loss

        active_update: Dict[str, Any] = {
            "progress": clamped_progress,
            "current_loss": loss,
            "current_accuracy": accuracy,
        }
        if sample_loss is not None:
            active_update["val_sample_loss"] = sample_loss
        if sample_accuracy is not None:
            active_update["val_sample_accuracy"] = sample_accuracy
        if video_count is not None:
            active_update["val_video_count"] = video_count
        if subgroup_metrics is not None:
            active_update["val_subgroup_metrics"] = subgroup_metrics
        if subgroup_macro_accuracy is not None:
            active_update["val_subgroup_macro_accuracy"] = subgroup_macro_accuracy
        if checkpoint_selection_score is not None:
            active_update["checkpoint_selection_score"] = checkpoint_selection_score
        if epoch is not None:
            active_update["current_epoch"] = epoch
        if total_epochs is not None:
            active_update["total_epochs"] = total_epochs
        if message is not None:
            active_update["message"] = message
        self._set_active_job_metadata(job_id, active_update)

    def _set_active_job_metadata(self, job_id: int, updates: Dict[str, Any]):
        """Merge metadata updates for a running training job."""
        if job_id not in self._active_jobs:
            self._active_jobs[job_id] = {}
        self._active_jobs[job_id].update(updates)

    def _is_job_cancelled(self, job_id: int) -> bool:
        """Check whether job status has been set to cancelled."""
        with get_db_session() as db:
            job = db.query(TrainingJob.status).filter(TrainingJob.id == job_id).first()
            if not job:
                return True
            return job[0] == JobStatus.CANCELLED.value

    async def _cancel_training(self, job_id: int):
        """Cancel running training job"""
        with get_db_session() as db:
            job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if job and job.status in [JobStatus.PENDING.value, JobStatus.RUNNING.value]:
                job.status = JobStatus.CANCELLED.value
                if not job.completed_at:
                    job.completed_at = datetime.now()

        if job_id in self._active_jobs:
            del self._active_jobs[job_id]

    def _db_to_response(self, job: TrainingJob) -> TrainingJobResponse:
        """Convert database model to response schema"""
        from app.schemas.training import TrainingParameters, TrainingResults

        active_meta = self._active_jobs.get(job.id, {})
        epoch_history = self._read_epoch_metrics_payload(job.id)
        latest_epoch_metric = None
        best_epoch_metric = None
        if epoch_history.get("metrics"):
            latest_epoch_metric = epoch_history["metrics"][-1]
        checkpoint_metadata = (
            self._read_checkpoint_metadata(job.model_path)
            if job.model_path and job.status == JobStatus.COMPLETED.value
            else {}
        )
        checkpoint_parameters = checkpoint_metadata.get("parameters", {})
        training_device = active_meta.get("requested_device", settings.TRAINING_DEVICE)
        current_epoch = active_meta.get("current_epoch")
        total_epochs = active_meta.get("total_epochs", job.epochs)
        progress_message = active_meta.get("message")
        validation_split = active_meta.get(
            "validation_split", checkpoint_parameters.get("validation_split", 0.2)
        )
        early_stopping = active_meta.get("early_stopping", True)
        patience = active_meta.get("patience", 10)
        early_stopping_min_delta = active_meta.get("early_stopping_min_delta", 0.002)
        weight_decay = active_meta.get("weight_decay", 0.0001)
        label_smoothing = active_meta.get("label_smoothing", 0.05)
        class_weight_strategy = active_meta.get(
            "class_weight_strategy",
            checkpoint_parameters.get("class_weight_strategy", "balanced"),
        )
        use_official_test_as_validation = active_meta.get(
            "use_official_test_as_validation",
            checkpoint_parameters.get("use_official_test_as_validation", False),
        )
        best_epoch = active_meta.get(
            "best_epoch", checkpoint_parameters.get("best_epoch")
        )
        if best_epoch is not None and epoch_history.get("metrics"):
            best_epoch_metric = next(
                (
                    metric
                    for metric in epoch_history["metrics"]
                    if int(metric.get("epoch", 0)) == int(best_epoch)
                ),
                None,
            )
        metric_snapshot = best_epoch_metric or latest_epoch_metric or {}
        epochs_trained = active_meta.get(
            "epochs_trained", checkpoint_parameters.get("epochs_trained")
        )
        val_sample_accuracy = active_meta.get(
            "val_sample_accuracy",
            metric_snapshot.get(
                "val_sample_accuracy", checkpoint_parameters.get("val_sample_accuracy")
            ),
        )
        val_sample_loss = active_meta.get(
            "val_sample_loss",
            metric_snapshot.get(
                "val_sample_loss", checkpoint_parameters.get("val_sample_loss")
            ),
        )
        val_video_count = active_meta.get(
            "val_video_count",
            metric_snapshot.get(
                "val_video_count", checkpoint_parameters.get("val_video_count")
            ),
        )
        val_subgroup_metrics = active_meta.get(
            "val_subgroup_metrics",
            metric_snapshot.get(
                "val_subgroup_metrics",
                checkpoint_parameters.get("val_subgroup_metrics"),
            ),
        )
        val_subgroup_macro_accuracy = active_meta.get(
            "val_subgroup_macro_accuracy",
            metric_snapshot.get(
                "val_subgroup_macro_accuracy",
                checkpoint_parameters.get("val_subgroup_macro_accuracy"),
            ),
        )
        checkpoint_selection_score = active_meta.get(
            "checkpoint_selection_score",
            metric_snapshot.get(
                "checkpoint_selection_score",
                checkpoint_parameters.get("checkpoint_selection_score"),
            ),
        )
        preprocessing_stage = active_meta.get("preprocessing_stage")
        preprocessing_progress = active_meta.get("preprocessing_progress")
        preprocessing_current = active_meta.get("preprocessing_current")
        preprocessing_total = active_meta.get("preprocessing_total")
        preprocessing_unit = active_meta.get("preprocessing_unit")

        if current_epoch is None and job.status == JobStatus.COMPLETED.value:
            current_epoch = job.epochs
        if epochs_trained is None and job.status == JobStatus.COMPLETED.value:
            epochs_trained = job.epochs

        if progress_message is None:
            if job.status == JobStatus.COMPLETED.value:
                progress_message = "训练完成，可决定是否保留模型文件"
            elif job.status == JobStatus.FAILED.value and job.error_message:
                progress_message = f"训练失败：{job.error_message}"
            elif job.status == JobStatus.CANCELLED.value:
                progress_message = "训练已取消"
            elif job.status == JobStatus.PENDING.value:
                progress_message = "等待开始训练"

        parameters = TrainingParameters(
            epochs=job.epochs if job.epochs is not None else settings.DEFAULT_EPOCHS,
            learning_rate=job.learning_rate
            if job.learning_rate is not None
            else settings.DEFAULT_LEARNING_RATE,
            batch_size=job.batch_size
            if job.batch_size is not None
            else settings.MODEL_BATCH_SIZE,
            validation_split=validation_split,
            early_stopping=early_stopping,
            patience=patience,
            early_stopping_min_delta=early_stopping_min_delta,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
            class_weight_strategy=class_weight_strategy,
            use_official_test_as_validation=use_official_test_as_validation,
            training_device=training_device,
            sequence_length=int(
                active_meta.get(
                    "sequence_length", checkpoint_parameters.get("sequence_length", 8)
                )
                or 8
            ),
            frame_stride=int(
                active_meta.get(
                    "frame_stride", checkpoint_parameters.get("frame_stride", 2)
                )
                or 2
            ),
            temporal_hidden_size=int(
                active_meta.get(
                    "temporal_hidden_size",
                    checkpoint_parameters.get("temporal_hidden_size", 256),
                )
                or 256
            ),
            temporal_num_layers=int(
                active_meta.get(
                    "temporal_num_layers",
                    checkpoint_parameters.get("temporal_num_layers", 2),
                )
                or 2
            ),
            feature_projection_size=int(
                active_meta.get(
                    "feature_projection_size",
                    checkpoint_parameters.get("feature_projection_size", 256),
                )
                or 256
            ),
            face_roi_enabled=bool(
                active_meta.get(
                    "face_roi_enabled",
                    checkpoint_parameters.get("face_roi_enabled", True),
                )
            ),
            face_roi_confidence_threshold=float(
                active_meta.get(
                    "face_roi_confidence_threshold",
                    checkpoint_parameters.get("face_roi_confidence_threshold", 0.35),
                )
                or 0.35
            ),
            face_roi_crop_padding=float(
                active_meta.get(
                    "face_roi_crop_padding",
                    checkpoint_parameters.get("face_roi_crop_padding", 0.18),
                )
                or 0.18
            ),
            temporal_bidirectional=bool(
                active_meta.get(
                    "temporal_bidirectional",
                    checkpoint_parameters.get(
                        "temporal_bidirectional", settings.TEMPORAL_BIDIRECTIONAL
                    ),
                )
            ),
            temporal_attention_pooling=bool(
                active_meta.get(
                    "temporal_attention_pooling",
                    checkpoint_parameters.get(
                        "temporal_attention_pooling",
                        settings.TEMPORAL_ATTENTION_POOLING,
                    ),
                )
            ),
            train_clip_overlap_ratio=float(
                active_meta.get(
                    "train_clip_overlap_ratio",
                    checkpoint_parameters.get(
                        "train_clip_overlap_ratio",
                        settings.TEMPORAL_TRAIN_CLIP_OVERLAP_RATIO,
                    ),
                )
            ),
            val_clip_overlap_ratio=float(
                active_meta.get(
                    "val_clip_overlap_ratio",
                    checkpoint_parameters.get(
                        "val_clip_overlap_ratio",
                        settings.TEMPORAL_VAL_CLIP_OVERLAP_RATIO,
                    ),
                )
            ),
            temporal_clip_filter_enabled=bool(
                active_meta.get(
                    "temporal_clip_filter_enabled",
                    checkpoint_parameters.get(
                        "temporal_clip_filter_enabled",
                        settings.TEMPORAL_CLIP_FILTER_ENABLED,
                    ),
                )
            ),
            temporal_clip_min_frame_std=float(
                active_meta.get(
                    "temporal_clip_min_frame_std",
                    checkpoint_parameters.get(
                        "temporal_clip_min_frame_std",
                        settings.TEMPORAL_CLIP_MIN_FRAME_STD,
                    ),
                )
            ),
            temporal_clip_min_motion_delta=float(
                active_meta.get(
                    "temporal_clip_min_motion_delta",
                    checkpoint_parameters.get(
                        "temporal_clip_min_motion_delta",
                        settings.TEMPORAL_CLIP_MIN_MOTION_DELTA,
                    ),
                )
            ),
        )

        results = None
        if job.accuracy is not None:
            results = TrainingResults(
                accuracy=job.accuracy,
                loss=job.loss,
                val_accuracy=job.accuracy,
                val_loss=job.loss,
                val_sample_accuracy=val_sample_accuracy,
                val_sample_loss=val_sample_loss,
                val_video_count=val_video_count,
                val_subgroup_metrics=val_subgroup_metrics,
                val_subgroup_macro_accuracy=val_subgroup_macro_accuracy,
                checkpoint_selection_score=checkpoint_selection_score,
                epochs_trained=epochs_trained,
                best_epoch=best_epoch,
                model_path=job.model_path,
            )

        return TrainingJobResponse(
            id=job.id,
            name=job.name,
            description=job.description,
            model_type=job.model_type,
            dataset_path=job.dataset_path,
            current_epoch=current_epoch,
            total_epochs=total_epochs,
            progress_message=progress_message,
            preprocessing_stage=preprocessing_stage,
            preprocessing_progress=preprocessing_progress,
            preprocessing_current=preprocessing_current,
            preprocessing_total=preprocessing_total,
            preprocessing_unit=preprocessing_unit,
            parameters=parameters,
            status=self._normalize_job_status(job.status),
            progress=job.progress if job.progress is not None else 0.0,
            results=results,
            created_at=job.created_at,
            updated_at=job.updated_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
        )

    def _normalize_job_status(self, raw_status: Optional[str]) -> JobStatus:
        """Normalize persisted status values to supported enum values."""
        status_value = raw_status or JobStatus.PENDING.value
        try:
            return JobStatus(status_value)
        except Exception:
            return JobStatus.PENDING

    def _register_retained_model(
        self, job: TrainingJob, model_path: Path, current_user: str
    ) -> ModelRegistry:
        """Create or update a model registry record for a retained checkpoint."""
        metadata = self._read_checkpoint_metadata(str(model_path))
        parameters = metadata.get("parameters", {})
        accuracy = metadata.get("val_accuracy", job.accuracy)
        existing = (
            self.db.query(ModelRegistry)
            .filter(ModelRegistry.training_job_id == job.id)
            .first()
        )

        payload = {
            "name": existing.name if existing else self._build_model_name(job),
            "model_type": metadata.get("model_type", job.model_type),
            "version": self._build_model_version(job),
            "file_path": str(model_path),
            "description": job.description
            or f"Retained model from training job {job.id}",
            "input_size": metadata.get("input_size", settings.MODEL_INPUT_SIZE),
            "num_classes": metadata.get("num_classes", 2),
            "parameters": parameters,
            "accuracy": accuracy,
            "status": ModelStatus.READY.value,
            "training_job_id": job.id,
            "del_flag": 0,
        }

        if existing:
            for field, value in payload.items():
                setattr(existing, field, value)
            registry_model = existing
        else:
            registry_model = ModelRegistry(**payload)
            self.db.add(registry_model)

        self.db.commit()
        self.db.refresh(registry_model)
        logger.info(
            "Model registry synchronized for retained checkpoint",
            job_id=job.id,
            model_registry_id=registry_model.id,
            user=current_user,
        )
        return registry_model

    def _build_model_name(self, job: TrainingJob) -> str:
        """Build a stable model name for a retained training job."""
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", (job.name or "model").strip()).strip("-")
        if not slug:
            slug = f"{job.model_type}-model"
        return f"{slug}-job-{job.id}"

    def _build_model_version(self, job: TrainingJob) -> str:
        """Build a deterministic version string from completion timestamp."""
        completed_at = job.completed_at or datetime.now()
        return completed_at.strftime("%Y.%m.%d.%H%M%S")

    def _read_checkpoint_metadata(self, model_path: Optional[str]) -> Dict[str, Any]:
        """Read checkpoint metadata without loading tensors into accelerators."""
        if not model_path:
            return {}

        resolved_path = Path(model_path).expanduser()
        if not resolved_path.is_absolute():
            resolved_path = (Path.cwd() / resolved_path).resolve()
        if not resolved_path.exists() or not resolved_path.is_file():
            return {}

        try:
            checkpoint = torch.load(str(resolved_path), map_location="cpu")
        except Exception as exc:
            logger.warning(
                "Failed to read checkpoint metadata",
                model_path=str(resolved_path),
                error=str(exc),
            )
            return {}

        parameters = {
            "epochs": checkpoint.get("epochs"),
            "learning_rate": checkpoint.get("learning_rate"),
            "batch_size": checkpoint.get("batch_size"),
            "validation_split": checkpoint.get("validation_split"),
            "weight_decay": checkpoint.get("weight_decay"),
            "label_smoothing": checkpoint.get("label_smoothing"),
            "class_weight_strategy": checkpoint.get("class_weight_strategy"),
            "class_weights": checkpoint.get("class_weights"),
            "majority_downsampling": checkpoint.get("majority_downsampling"),
            "use_official_test_as_validation": checkpoint.get(
                "use_official_test_as_validation"
            ),
            "split_strategy": checkpoint.get("split_strategy"),
            "official_test_holdout_count": checkpoint.get(
                "official_test_holdout_count"
            ),
            "train_subgroup_counts": checkpoint.get("train_subgroup_counts"),
            "val_subgroup_counts": checkpoint.get("val_subgroup_counts"),
            "train_clip_filter": checkpoint.get("train_clip_filter"),
            "early_stopping": checkpoint.get("early_stopping"),
            "patience": checkpoint.get("patience"),
            "early_stopping_min_delta": checkpoint.get("early_stopping_min_delta"),
            "training_device": checkpoint.get("training_device"),
            "sequence_length": checkpoint.get("sequence_length"),
            "frame_stride": checkpoint.get("frame_stride"),
            "hidden_size": checkpoint.get("hidden_size"),
            "num_layers": checkpoint.get("num_layers"),
            "video_temporal_enabled": checkpoint.get("video_temporal_enabled"),
            "training_mode": checkpoint.get("training_mode"),
            "temporal_hidden_size": checkpoint.get("temporal_hidden_size"),
            "temporal_num_layers": checkpoint.get("temporal_num_layers"),
            "feature_projection_size": checkpoint.get("feature_projection_size"),
            "temporal_bidirectional": checkpoint.get("temporal_bidirectional"),
            "temporal_attention_pooling": checkpoint.get("temporal_attention_pooling"),
            "face_roi_enabled": checkpoint.get("face_roi_enabled"),
            "yolo_face_model_path": checkpoint.get("yolo_face_model_path"),
            "face_roi_confidence_threshold": checkpoint.get(
                "face_roi_confidence_threshold"
            ),
            "face_roi_crop_padding": checkpoint.get("face_roi_crop_padding"),
            "face_roi_policy_version": checkpoint.get("face_roi_policy_version"),
            "face_roi_selection_policy": checkpoint.get("face_roi_selection_policy"),
            "video_aggregation_topk_ratio": checkpoint.get(
                "video_aggregation_topk_ratio"
            ),
            "video_aggregation_mean_weight": checkpoint.get(
                "video_aggregation_mean_weight"
            ),
            "video_aggregation_peak_weight": checkpoint.get(
                "video_aggregation_peak_weight"
            ),
            "video_aggregation_persistence_weight": checkpoint.get(
                "video_aggregation_persistence_weight"
            ),
            "train_clip_overlap_ratio": checkpoint.get("train_clip_overlap_ratio"),
            "val_clip_overlap_ratio": checkpoint.get("val_clip_overlap_ratio"),
            "temporal_clip_filter_enabled": checkpoint.get(
                "temporal_clip_filter_enabled"
            ),
            "temporal_clip_min_frame_std": checkpoint.get(
                "temporal_clip_min_frame_std"
            ),
            "temporal_clip_min_motion_delta": checkpoint.get(
                "temporal_clip_min_motion_delta"
            ),
            "frame_projection_size": checkpoint.get("frame_projection_size"),
            "best_epoch": checkpoint.get("best_epoch"),
            "epochs_trained": checkpoint.get("epochs_trained"),
            "val_sample_accuracy": checkpoint.get("val_sample_accuracy"),
            "val_sample_loss": checkpoint.get("val_sample_loss"),
            "val_video_count": checkpoint.get("val_video_count"),
            "val_subgroup_metrics": checkpoint.get("val_subgroup_metrics"),
            "val_subgroup_macro_accuracy": checkpoint.get(
                "val_subgroup_macro_accuracy"
            ),
            "checkpoint_selection_score": checkpoint.get("checkpoint_selection_score"),
        }
        filtered_parameters = {
            key: value for key, value in parameters.items() if value is not None
        }
        checkpoint["parameters"] = filtered_parameters
        return checkpoint
