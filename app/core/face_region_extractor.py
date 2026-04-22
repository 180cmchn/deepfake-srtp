from __future__ import annotations

import importlib
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from app.core.config import settings
from app.core.logging import logger


EYE_LEFT_INDICES = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
]
EYE_RIGHT_INDICES = [
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
]
MOUTH_INDICES = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
    308,
]


@dataclass(frozen=True)
class FaceRegions:
    face: Image.Image
    eyes: Image.Image
    mouth: Image.Image
    metadata: Dict[str, Any]


class FaceRegionExtractor:
    _mesh_cache: Optional[Any] = None

    def __init__(self):
        pass

    def build_policy(
        self, overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        overrides = overrides or {}
        policy = {
            "face_region_mode": str(
                overrides.get("face_region_mode", settings.FACE_REGION_MODE)
            ),
            "face_region_policy_version": str(
                overrides.get(
                    "face_region_policy_version", settings.FACE_REGION_POLICY_VERSION
                )
            ),
            "face_region_eye_padding": float(
                overrides.get(
                    "face_region_eye_padding", settings.FACE_REGION_EYE_PADDING
                )
            ),
            "face_region_mouth_padding": float(
                overrides.get(
                    "face_region_mouth_padding", settings.FACE_REGION_MOUTH_PADDING
                )
            ),
            "face_region_require_landmarks": bool(
                overrides.get(
                    "face_region_require_landmarks",
                    settings.FACE_REGION_REQUIRE_LANDMARKS,
                )
            ),
            "face_region_eye_weight": float(
                overrides.get("face_region_eye_weight", settings.FACE_REGION_EYE_WEIGHT)
            ),
            "face_region_mouth_weight": float(
                overrides.get(
                    "face_region_mouth_weight", settings.FACE_REGION_MOUTH_WEIGHT
                )
            ),
        }
        policy["face_region_effective_enabled"] = (
            policy["face_region_mode"] == "face_eyes_mouth_fusion"
        )
        policy["face_region_policy_fingerprint"] = self.get_policy_fingerprint(policy)
        return policy

    def get_policy_fingerprint(self, policy: Optional[Dict[str, Any]]) -> str:
        if not policy:
            return "face-region-disabled"
        relevant = {
            "face_region_mode": policy.get("face_region_mode", "face_only"),
            "face_region_policy_version": policy.get(
                "face_region_policy_version", settings.FACE_REGION_POLICY_VERSION
            ),
            "face_region_eye_padding": float(
                policy.get("face_region_eye_padding", settings.FACE_REGION_EYE_PADDING)
            ),
            "face_region_mouth_padding": float(
                policy.get(
                    "face_region_mouth_padding", settings.FACE_REGION_MOUTH_PADDING
                )
            ),
            "face_region_require_landmarks": bool(
                policy.get(
                    "face_region_require_landmarks",
                    settings.FACE_REGION_REQUIRE_LANDMARKS,
                )
            ),
        }
        digest = hashlib.sha1(
            json.dumps(relevant, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return digest[:12]

    def extract_regions(
        self,
        face_image: Image.Image,
        target_size: int = 224,
        policy: Optional[Dict[str, Any]] = None,
    ) -> FaceRegions:
        rgb = face_image.convert("RGB")
        w, h = rgb.size
        effective_policy = self.build_policy(policy)

        landmarks = self._detect_landmarks(rgb)

        if landmarks is None:
            if effective_policy["face_region_require_landmarks"]:
                metadata = {
                    "face_region_mode": effective_policy["face_region_mode"],
                    "face_region_policy_version": effective_policy[
                        "face_region_policy_version"
                    ],
                    "face_region_policy_fingerprint": effective_policy[
                        "face_region_policy_fingerprint"
                    ],
                    "face_region_landmark_detected": False,
                    "face_region_fallback_used": False,
                    "face_region_detector_status": "landmarks_missing",
                    "eye_bbox": None,
                    "mouth_bbox": None,
                }
                return FaceRegions(
                    face=rgb.resize((target_size, target_size), Image.BILINEAR),
                    eyes=rgb.resize((target_size, target_size), Image.BILINEAR),
                    mouth=rgb.resize((target_size, target_size), Image.BILINEAR),
                    metadata=metadata,
                )
            eye_crop = self._fallback_eye_region(rgb, w, h)
            mouth_crop = self._fallback_mouth_region(rgb, w, h)
            metadata = {
                "face_region_mode": effective_policy["face_region_mode"],
                "face_region_policy_version": effective_policy[
                    "face_region_policy_version"
                ],
                "face_region_policy_fingerprint": effective_policy[
                    "face_region_policy_fingerprint"
                ],
                "face_region_landmark_detected": False,
                "eye_bbox": None,
                "mouth_bbox": None,
                "face_region_fallback_used": True,
                "face_region_detector_status": "fallback_boxes",
            }
        else:
            eye_bbox = self._landmarks_to_bbox(
                landmarks,
                EYE_LEFT_INDICES + EYE_RIGHT_INDICES,
                w,
                h,
                effective_policy["face_region_eye_padding"],
            )
            mouth_bbox = self._landmarks_to_bbox(
                landmarks,
                MOUTH_INDICES,
                w,
                h,
                effective_policy["face_region_mouth_padding"],
            )
            eye_crop = rgb.crop(eye_bbox)
            mouth_crop = rgb.crop(mouth_bbox)
            metadata = {
                "face_region_mode": effective_policy["face_region_mode"],
                "face_region_policy_version": effective_policy[
                    "face_region_policy_version"
                ],
                "face_region_policy_fingerprint": effective_policy[
                    "face_region_policy_fingerprint"
                ],
                "face_region_landmark_detected": True,
                "eye_bbox": list(eye_bbox),
                "mouth_bbox": list(mouth_bbox),
                "face_region_fallback_used": False,
                "face_region_detector_status": "landmarks_detected",
            }

        face_resized = rgb.resize((target_size, target_size), Image.BILINEAR)
        eye_resized = eye_crop.resize((target_size, target_size), Image.BILINEAR)
        mouth_resized = mouth_crop.resize((target_size, target_size), Image.BILINEAR)

        return FaceRegions(
            face=face_resized,
            eyes=eye_resized,
            mouth=mouth_resized,
            metadata=metadata,
        )

    def _detect_landmarks(
        self, image: Image.Image
    ) -> Optional[List[Tuple[float, float]]]:
        mesh = self._get_or_create_mesh()
        if mesh is None:
            return None

        image_array = np.array(image, dtype=np.uint8)
        try:
            results = mesh.process(image_array)
        except Exception as exc:
            logger.warning(
                "MediaPipe face mesh processing failed",
                error=str(exc),
            )
            return None

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]
        return landmarks

    def _get_or_create_mesh(self) -> Optional[Any]:
        if FaceRegionExtractor._mesh_cache is not None:
            if isinstance(FaceRegionExtractor._mesh_cache, Exception):
                return None
            return FaceRegionExtractor._mesh_cache

        try:
            mp = importlib.import_module("mediapipe")
            mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.3,
            )
            FaceRegionExtractor._mesh_cache = mesh
            return mesh
        except Exception as exc:
            logger.warning(
                "MediaPipe unavailable for face region extraction",
                error=str(exc),
            )
            FaceRegionExtractor._mesh_cache = exc
            return None

    def _landmarks_to_bbox(
        self,
        landmarks: List[Tuple[float, float]],
        indices: List[int],
        img_width: int,
        img_height: int,
        padding: float,
    ) -> Tuple[int, int, int, int]:
        valid_indices = [i for i in indices if 0 <= i < len(landmarks)]
        if not valid_indices:
            return (0, 0, img_width, img_height)

        xs = [landmarks[i][0] * img_width for i in valid_indices]
        ys = [landmarks[i][1] * img_height for i in valid_indices]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        box_w = max(1.0, max_x - min_x)
        box_h = max(1.0, max_y - min_y)
        pad_x = box_w * padding
        pad_y = box_h * padding

        left = max(0, int(min_x - pad_x))
        top = max(0, int(min_y - pad_y))
        right = min(img_width, int(max_x + pad_x))
        bottom = min(img_height, int(max_y + pad_y))

        if right <= left or bottom <= top:
            return (0, 0, img_width, img_height)
        return (left, top, right, bottom)

    def _fallback_eye_region(self, image: Image.Image, w: int, h: int) -> Image.Image:
        top = int(h * 0.15)
        bottom = int(h * 0.45)
        left = int(w * 0.08)
        right = int(w * 0.92)
        return image.crop((left, top, right, bottom))

    def _fallback_mouth_region(self, image: Image.Image, w: int, h: int) -> Image.Image:
        top = int(h * 0.55)
        bottom = int(h * 0.90)
        left = int(w * 0.15)
        right = int(w * 0.85)
        return image.crop((left, top, right, bottom))
