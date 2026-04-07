from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from app.core.config import settings
from app.core.logging import logger


@dataclass(frozen=True)
class FaceCandidate:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height


class SingleFaceRoiProcessor:
    _detector_cache: Dict[str, Any] = {}

    def build_policy(
        self, overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        overrides = overrides or {}
        configured_model_path = overrides.get(
            "yolo_face_model_path", settings.YOLO_FACE_MODEL_PATH
        )
        model_path = (
            str(configured_model_path).strip() if configured_model_path else None
        )
        enabled = bool(overrides.get("face_roi_enabled", settings.YOLO_FACE_ENABLED))
        policy = {
            "face_roi_enabled": enabled,
            "yolo_face_model_path": model_path,
            "face_roi_confidence_threshold": float(
                overrides.get(
                    "face_roi_confidence_threshold",
                    settings.YOLO_FACE_CONFIDENCE_THRESHOLD,
                )
            ),
            "face_roi_crop_padding": float(
                overrides.get("face_roi_crop_padding", settings.YOLO_FACE_CROP_PADDING)
            ),
            "face_roi_policy_version": str(
                overrides.get(
                    "face_roi_policy_version", settings.YOLO_FACE_POLICY_VERSION
                )
            ),
            "face_roi_selection_policy": str(
                overrides.get(
                    "face_roi_selection_policy",
                    settings.YOLO_FACE_SELECTION_POLICY,
                )
            ),
        }
        policy["face_roi_effective_enabled"] = bool(enabled and model_path)
        policy["face_roi_policy_fingerprint"] = self.get_policy_fingerprint(policy)
        return policy

    def get_policy_fingerprint(self, policy: Optional[Dict[str, Any]]) -> str:
        if not policy:
            return "face-roi-disabled"
        relevant = {
            "face_roi_enabled": bool(policy.get("face_roi_enabled", False)),
            "yolo_face_model_path": policy.get("yolo_face_model_path") or "",
            "face_roi_confidence_threshold": float(
                policy.get(
                    "face_roi_confidence_threshold",
                    settings.YOLO_FACE_CONFIDENCE_THRESHOLD,
                )
            ),
            "face_roi_crop_padding": float(
                policy.get("face_roi_crop_padding", settings.YOLO_FACE_CROP_PADDING)
            ),
            "face_roi_policy_version": str(
                policy.get("face_roi_policy_version", settings.YOLO_FACE_POLICY_VERSION)
            ),
            "face_roi_selection_policy": str(
                policy.get(
                    "face_roi_selection_policy", settings.YOLO_FACE_SELECTION_POLICY
                )
            ),
        }
        digest = hashlib.sha1(
            json.dumps(relevant, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return digest[:12]

    def crop_pil(
        self, image: Image.Image, policy: Optional[Dict[str, Any]] = None
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        rgb_image = image.convert("RGB")
        effective_policy = self.build_policy(policy)
        metadata: Dict[str, Any] = {
            "face_roi_enabled": effective_policy["face_roi_enabled"],
            "face_roi_effective_enabled": effective_policy[
                "face_roi_effective_enabled"
            ],
            "face_roi_policy_version": effective_policy["face_roi_policy_version"],
            "face_roi_selection_policy": effective_policy["face_roi_selection_policy"],
            "face_roi_policy_fingerprint": effective_policy[
                "face_roi_policy_fingerprint"
            ],
            "face_roi_applied": False,
            "face_roi_fallback_used": True,
            "face_roi_confidence": None,
            "face_roi_bbox": None,
            "face_roi_detector_backend": "ultralytics_yolo",
            "face_roi_detector_status": "disabled",
            "face_roi_detector_version": self._ultralytics_version(),
        }

        if not effective_policy["face_roi_effective_enabled"]:
            metadata["face_roi_detector_status"] = (
                "weights_missing"
                if effective_policy["face_roi_enabled"]
                else "disabled"
            )
            return rgb_image, metadata

        candidates, detector_status = self._detect_candidates(
            rgb_image, effective_policy
        )
        metadata["face_roi_detector_status"] = detector_status
        if not candidates:
            return rgb_image, metadata

        candidate = self._select_candidate(candidates, rgb_image.size)
        crop_box = self._square_crop_box(
            candidate,
            rgb_image.size,
            effective_policy["face_roi_crop_padding"],
        )
        cropped = rgb_image.crop(crop_box)
        metadata.update(
            {
                "face_roi_applied": True,
                "face_roi_fallback_used": False,
                "face_roi_confidence": float(candidate.confidence),
                "face_roi_bbox": list(crop_box),
            }
        )
        return cropped, metadata

    def crop_frame(
        self, frame_bgr: np.ndarray, policy: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        rgb_image = Image.fromarray(frame_bgr[:, :, ::-1])
        cropped_image, metadata = self.crop_pil(rgb_image, policy)
        cropped_rgb = np.array(cropped_image, dtype=np.uint8)
        return cropped_rgb[:, :, ::-1], metadata

    def _ultralytics_version(self) -> Optional[str]:
        try:
            return importlib.metadata.version("ultralytics")
        except Exception:
            return None

    def _load_detector(self, policy: Dict[str, Any]) -> Tuple[Optional[Any], str]:
        model_path = policy.get("yolo_face_model_path")
        if not model_path:
            return None, "weights_missing"

        cache_key = str(model_path)
        if cache_key in self._detector_cache:
            cached = self._detector_cache[cache_key]
            if isinstance(cached, Exception):
                return None, type(cached).__name__.lower()
            return cached, "loaded"

        try:
            YOLO = importlib.import_module("ultralytics").YOLO
        except Exception as exc:
            self._detector_cache[cache_key] = exc
            logger.warning(
                "Ultralytics unavailable for face ROI preprocessing",
                model_path=model_path,
                error=str(exc),
            )
            return None, "import_error"

        try:
            detector = YOLO(model_path)
        except Exception as exc:
            self._detector_cache[cache_key] = exc
            logger.warning(
                "Failed to initialize YOLO face detector",
                model_path=model_path,
                error=str(exc),
            )
            return None, "load_error"

        self._detector_cache[cache_key] = detector
        return detector, "loaded"

    def _detect_candidates(
        self, image: Image.Image, policy: Dict[str, Any]
    ) -> Tuple[List[FaceCandidate], str]:
        detector, detector_status = self._load_detector(policy)
        if detector is None:
            return [], detector_status

        image_array = np.array(image, dtype=np.uint8)
        try:
            if callable(getattr(detector, "predict", None)):
                results = detector.predict(
                    source=image_array,
                    conf=policy["face_roi_confidence_threshold"],
                    verbose=False,
                )
            else:
                results = detector(
                    image_array,
                    conf=policy["face_roi_confidence_threshold"],
                    verbose=False,
                )
        except Exception as exc:
            logger.warning(
                "YOLO face detection failed during preprocessing",
                error=str(exc),
            )
            return [], "predict_error"

        candidates: List[FaceCandidate] = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = getattr(boxes, "xyxy", None)
            confidences = getattr(boxes, "conf", None)
            if xyxy is None:
                continue
            try:
                xyxy_values = xyxy.tolist()
            except Exception:
                xyxy_values = []
            try:
                confidence_values = (
                    confidences.tolist() if confidences is not None else []
                )
            except Exception:
                confidence_values = []

            for index, raw_box in enumerate(xyxy_values):
                if len(raw_box) < 4:
                    continue
                confidence = float(
                    confidence_values[index] if index < len(confidence_values) else 1.0
                )
                if confidence < policy["face_roi_confidence_threshold"]:
                    continue
                x1, y1, x2, y2 = [int(round(value)) for value in raw_box[:4]]
                if x2 <= x1 or y2 <= y1:
                    continue
                candidates.append(
                    FaceCandidate(
                        x1=max(0, x1),
                        y1=max(0, y1),
                        x2=max(0, x2),
                        y2=max(0, y2),
                        confidence=confidence,
                    )
                )

        return candidates, detector_status

    def _select_candidate(
        self, candidates: List[FaceCandidate], image_size: Tuple[int, int]
    ) -> FaceCandidate:
        width, height = image_size
        image_area = max(1, width * height)

        def rank_key(candidate: FaceCandidate) -> Tuple[float, float]:
            area_ratio = candidate.area / float(image_area)
            return (candidate.confidence, area_ratio)

        return max(candidates, key=rank_key)

    def _square_crop_box(
        self,
        candidate: FaceCandidate,
        image_size: Tuple[int, int],
        padding_ratio: float,
    ) -> Tuple[int, int, int, int]:
        width, height = image_size
        if width <= 0 or height <= 0:
            return (0, 0, max(1, width), max(1, height))

        center_x = (candidate.x1 + candidate.x2) / 2.0
        center_y = (candidate.y1 + candidate.y2) / 2.0
        side = max(candidate.width, candidate.height)
        side = max(1.0, side * (1.0 + max(0.0, padding_ratio) * 2.0))
        side = min(side, float(min(width, height)))
        half_side = side / 2.0

        left = int(round(center_x - half_side))
        top = int(round(center_y - half_side))
        right = int(round(center_x + half_side))
        bottom = int(round(center_y + half_side))

        if left < 0:
            right += -left
            left = 0
        if top < 0:
            bottom += -top
            top = 0
        if right > width:
            left -= right - width
            right = width
        if bottom > height:
            top -= bottom - height
            bottom = height

        left = max(0, left)
        top = max(0, top)
        right = min(width, right)
        bottom = min(height, bottom)

        if right <= left or bottom <= top:
            return (0, 0, width, height)
        return (left, top, right, bottom)
