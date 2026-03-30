import os
import tempfile
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.schemas.detection import VideoDetectionRequest
from app.services.detection_service import DetectionService


class DummyBackgroundTasks:
    def add_task(self, *_args, **_kwargs):
        return None


class SavedDetectionRecord(SimpleNamespace):
    pass


class DetectionVideoMetadataTruthfulnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_detect_video_exposes_source_and_sampled_metadata_separately(self):
        service = DetectionService(db=None)
        loaded_model = {
            "model": object(),
            "model_type": "vit",
            "input_size": 224,
            "source": "registry",
        }
        saved_record = SavedDetectionRecord(id=88, created_at=datetime.now())

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(b"video-bytes")
            file_path = temp_file.name

        try:
            with (
                patch.object(
                    service,
                    "_get_video_source_metadata",
                    return_value={
                        "source_total_frames": 300,
                        "source_fps": 30.0,
                        "source_duration_seconds": 10.0,
                    },
                ),
                patch.object(
                    service,
                    "_extract_frames",
                    new=AsyncMock(return_value=[(object(), 0.0), (object(), 4.0)]),
                ),
                patch.object(
                    service,
                    "_load_model",
                    new=AsyncMock(return_value=loaded_model),
                ),
                patch.object(service, "_preprocess_frame", return_value=object()),
                patch.object(
                    service,
                    "_predict_tensor",
                    new=AsyncMock(
                        side_effect=[
                            {"probabilities": {"fake": 0.8, "real": 0.2}},
                            {"probabilities": {"fake": 0.8, "real": 0.2}},
                        ]
                    ),
                ),
                patch.object(
                    service,
                    "_save_detection_result",
                    new=AsyncMock(return_value=saved_record),
                ) as save_mock,
            ):
                response = await service.detect_video(
                    video_path=file_path,
                    request=VideoDetectionRequest(
                        model_type="vit",
                        confidence_threshold=0.9,
                    ),
                    background_tasks=DummyBackgroundTasks(),
                )
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

        self.assertTrue(response.success)
        self.assertEqual(response.video_info["source_total_frames"], 300)
        self.assertEqual(response.video_info["source_fps"], 30.0)
        self.assertEqual(response.video_info["source_duration_seconds"], 10.0)
        self.assertEqual(response.video_info["sampled_frame_count"], 2)
        self.assertEqual(response.video_info["analyzed_frame_count"], 2)
        self.assertEqual(response.video_info["sampled_duration_seconds"], 4.0)
        self.assertEqual(response.video_info["total_frames"], 2)
        self.assertEqual(response.video_info["processed_frames"], 2)
        self.assertEqual(response.video_info["duration"], 4.0)
        self.assertEqual(
            response.summary["aggregation_strategy"], "smoothed_frame_sequence"
        )
        self.assertEqual(response.summary["sampled_frame_count"], 2)
        self.assertEqual(response.summary["analyzed_frame_count"], 2)
        self.assertEqual(response.summary["source_total_frames"], 300)
        await_args = save_mock.await_args
        if await_args is None:
            self.fail("_save_detection_result should be awaited with video metadata")
        self.assertEqual(
            await_args.kwargs["video_metadata"]["source_total_frames"],
            300,
        )
        self.assertEqual(
            await_args.kwargs["video_metadata"]["sampled_frame_count"],
            2,
        )
        self.assertEqual(
            await_args.kwargs["video_metadata"]["analyzed_frame_count"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
