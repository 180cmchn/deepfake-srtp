import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from app.core.video_face_roi import SingleFaceRoiProcessor
from app.services.detection_service import DetectionService
from app.services.training_service import (
    SequenceClassificationDataset,
    TemporalClipSample,
    TrainingService,
)


class _MockRoiProcessor(SingleFaceRoiProcessor):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def crop_pil(self, image, _policy):
        self.calls += 1
        return image, {"face_roi_applied": True}


class FaceRoiPipelineTests(unittest.TestCase):
    def test_face_roi_processor_falls_back_when_detector_is_disabled_or_unconfigured(
        self,
    ):
        processor = SingleFaceRoiProcessor()
        image = Image.new("RGB", (80, 60), color=(120, 80, 40))

        cropped_disabled, disabled_metadata = processor.crop_pil(
            image,
            {
                "face_roi_enabled": False,
                "yolo_face_model_path": None,
            },
        )
        cropped_missing, missing_metadata = processor.crop_pil(
            image,
            {
                "face_roi_enabled": True,
                "yolo_face_model_path": None,
            },
        )

        self.assertEqual(cropped_disabled.size, image.size)
        self.assertEqual(cropped_missing.size, image.size)
        self.assertFalse(disabled_metadata["face_roi_applied"])
        self.assertFalse(missing_metadata["face_roi_applied"])
        self.assertEqual(disabled_metadata["face_roi_detector_status"], "disabled")
        self.assertEqual(
            missing_metadata["face_roi_detector_status"], "weights_missing"
        )

    def test_sequence_dataset_skips_double_roi_crop_for_materialized_clips(self):
        mock_processor = _MockRoiProcessor()
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_path = Path(temp_dir) / "frame.jpg"
            Image.new("RGB", (32, 32), color=(0, 0, 0)).save(frame_path)

            dataset = SequenceClassificationDataset(
                [
                    TemporalClipSample(
                        frame_paths=[str(frame_path), str(frame_path)],
                        label=0,
                        source_id="source-a",
                        roi_materialized=True,
                    )
                ],
                transform=lambda image: torch.from_numpy(
                    np.array(image, dtype=np.float32)
                ).permute(2, 0, 1),
                roi_processor=mock_processor,
                roi_policy={"face_roi_enabled": True},
            )

            result = dataset[0]

        clip = result[0]
        label = result[1]
        self.assertEqual(label, 0)
        self.assertEqual(tuple(clip.shape), (2, 3, 32, 32))
        self.assertEqual(mock_processor.calls, 0)

    def test_materialized_video_frame_cache_key_changes_with_roi_policy(self):
        service = TrainingService(db=None)
        face_image = Image.new("RGB", (64, 64), color=(255, 255, 255))

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "sample.mp4"
            video_path.write_bytes(b"video-bytes")
            cache_root = Path(temp_dir) / "cache"

            with (
                patch(
                    "app.services.training_service.ImageClassificationDataset._load_video_frame",
                    return_value=face_image,
                ),
                patch.object(
                    service._face_roi_processor,
                    "crop_pil",
                    return_value=(face_image, {"face_roi_applied": False}),
                ),
            ):
                disabled_path = service._materialize_video_frame(
                    str(video_path),
                    0,
                    cache_root,
                    224,
                    face_roi_policy={"face_roi_enabled": False},
                )
                enabled_path = service._materialize_video_frame(
                    str(video_path),
                    0,
                    cache_root,
                    224,
                    face_roi_policy={
                        "face_roi_enabled": True,
                        "yolo_face_model_path": "weights.pt",
                    },
                )

        self.assertNotEqual(disabled_path, enabled_path)

    def test_video_probability_aggregation_surfaces_topk_and_persistence_details(self):
        service = DetectionService(db=None)
        probability_sequence = [
            {"fake": 0.05, "real": 0.95},
            {"fake": 0.95, "real": 0.05},
            {"fake": 0.96, "real": 0.04},
            {"fake": 0.10, "real": 0.90},
        ]

        prediction, confidence, probabilities, decision_metrics = (
            service._aggregate_probability_sequence(
                probability_sequence,
                confidence_threshold=0.5,
                aggregation_policy={
                    "topk_ratio": 0.5,
                    "mean_weight": 0.2,
                    "peak_weight": 0.5,
                    "persistence_weight": 0.3,
                },
            )
        )

        self.assertEqual(prediction, "fake")
        self.assertAlmostEqual(confidence, probabilities["fake"])
        self.assertGreater(decision_metrics.topk_fake_probability, 0.9)
        self.assertGreater(decision_metrics.longest_positive_run_ratio, 0.4)
        self.assertGreater(decision_metrics.aggregated_fake_probability, 0.5)

    def test_temporal_clip_filter_drops_low_information_training_clips(self):
        service = TrainingService(db=None)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            static_frame_paths = []
            motion_frame_paths = []
            for index in range(3):
                static_path = temp_root / f"static-{index}.png"
                Image.new("RGB", (48, 48), color=(64, 64, 64)).save(static_path)
                static_frame_paths.append(str(static_path))

                motion_path = temp_root / f"motion-{index}.png"
                motion_image = Image.new("RGB", (48, 48), color=(16, 16, 16))
                for x in range(24):
                    for y in range(48):
                        motion_image.putpixel(
                            (x, y),
                            (
                                min(255, 40 + (index * 50) + (x * 2)),
                                min(255, 20 + (index * 40) + y),
                                10,
                            ),
                        )
                motion_image.save(motion_path)
                motion_frame_paths.append(str(motion_path))

            kept_samples, metadata = service._filter_temporal_clip_samples(
                [
                    TemporalClipSample(
                        frame_paths=static_frame_paths,
                        label=0,
                        source_id="static-source",
                    ),
                    TemporalClipSample(
                        frame_paths=motion_frame_paths,
                        label=1,
                        source_id="motion-source",
                    ),
                ],
                min_frame_std=1.0,
                min_motion_delta=5.0,
            )

        self.assertEqual(len(kept_samples), 1)
        self.assertEqual(kept_samples[0].source_id, "motion-source")
        self.assertEqual(metadata["dropped"], 1)

    def test_checkpoint_metadata_preserves_face_roi_and_temporal_policy(self):
        service = TrainingService(db=None)
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as temp_file:
            checkpoint_path = temp_file.name

        try:
            torch.save(
                {
                    "model_state_dict": {},
                    "face_roi_enabled": True,
                    "yolo_face_model_path": "/models/yolo-face.pt",
                    "face_roi_confidence_threshold": 0.42,
                    "face_roi_crop_padding": 0.2,
                    "face_roi_policy_version": "single_face_v1",
                    "face_roi_selection_policy": "confidence_area",
                    "temporal_bidirectional": True,
                    "temporal_attention_pooling": True,
                    "video_aggregation_topk_ratio": 0.25,
                    "video_aggregation_mean_weight": 0.45,
                    "video_aggregation_peak_weight": 0.35,
                    "video_aggregation_persistence_weight": 0.2,
                    "train_clip_overlap_ratio": 0.7,
                    "val_clip_overlap_ratio": 0.35,
                    "temporal_clip_filter_enabled": True,
                    "temporal_clip_min_frame_std": 6.0,
                    "temporal_clip_min_motion_delta": 2.0,
                },
                checkpoint_path,
            )

            metadata = service._read_checkpoint_metadata(checkpoint_path)
        finally:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

        parameters = metadata["parameters"]
        self.assertTrue(parameters["face_roi_enabled"])
        self.assertEqual(parameters["yolo_face_model_path"], "/models/yolo-face.pt")
        self.assertAlmostEqual(parameters["face_roi_confidence_threshold"], 0.42)
        self.assertAlmostEqual(parameters["face_roi_crop_padding"], 0.2)
        self.assertTrue(parameters["temporal_bidirectional"])
        self.assertTrue(parameters["temporal_attention_pooling"])
        self.assertAlmostEqual(parameters["video_aggregation_topk_ratio"], 0.25)
        self.assertTrue(parameters["temporal_clip_filter_enabled"])


if __name__ == "__main__":
    unittest.main()
