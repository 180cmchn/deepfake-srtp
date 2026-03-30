import unittest
from datetime import datetime
from types import SimpleNamespace

from app.services.detection_service import DetectionService


class FakeHistoryQuery:
    def __init__(self, records):
        self._records = list(records)
        self._offset = 0
        self._limit = None

    def filter(self, *_args, **_kwargs):
        return self

    def outerjoin(self, *_args, **_kwargs):
        return self

    def order_by(self, *_args, **_kwargs):
        return self

    def count(self):
        return len(self._records)

    def offset(self, value):
        self._offset = value
        return self

    def limit(self, value):
        self._limit = value
        return self

    def all(self):
        records = self._records[self._offset :]
        if self._limit is not None:
            records = records[: self._limit]
        return list(records)


class FakeHistoryDB:
    def __init__(self, records):
        self._records = list(records)

    def query(self, _model):
        return FakeHistoryQuery(self._records)


class DetectionHistoryFailedDiscoverabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_history_keeps_failed_rows_with_explicit_status_error_and_file_path(
        self,
    ):
        created_at = datetime(2026, 3, 30, 12, 0, 0)
        records = [
            SimpleNamespace(
                id=2,
                file_path="uploads/20260330/broken-upload.png",
                file_name="broken-upload.png",
                file_type="image",
                prediction="failed",
                confidence=0.0,
                processing_time=0.07,
                model_name="registry-vit",
                model_type="vit",
                status="failed",
                error_message="prediction exploded",
                created_at=created_at,
                source_total_frames=None,
                source_fps=None,
                source_duration_seconds=None,
                sampled_frame_count=None,
                analyzed_frame_count=None,
                sampled_duration_seconds=None,
                model=None,
            ),
            SimpleNamespace(
                id=1,
                file_path="uploads/20260330/original-upload.png",
                file_name="original-upload.png",
                file_type="image",
                prediction="real",
                confidence=0.91,
                processing_time=0.21,
                model_name="registry-vit",
                model_type="vit",
                status="completed",
                error_message=None,
                created_at=created_at,
                source_total_frames=None,
                source_fps=None,
                source_duration_seconds=None,
                sampled_frame_count=None,
                analyzed_frame_count=None,
                sampled_duration_seconds=None,
                model=None,
            ),
        ]

        history = await DetectionService(FakeHistoryDB(records)).get_history(limit=10)

        self.assertEqual(history.total, 2)
        self.assertEqual(len(history.detections), 2)
        self.assertCountEqual([item.id for item in history.detections], [1, 2])

        failed_item = next(item for item in history.detections if item.id == 2)
        self.assertEqual(failed_item.status.value, "failed")
        self.assertEqual(failed_item.error_message, "prediction exploded")
        self.assertEqual(failed_item.file_path, "uploads/20260330/broken-upload.png")
        self.assertIsNone(failed_item.prediction)
        self.assertIsNone(failed_item.confidence)
        self.assertEqual(failed_item.processing_time, 0.07)
        self.assertIsNotNone(failed_item.error_message)
        self.assertGreater(len(failed_item.error_message or ""), 0)

        completed_item = next(item for item in history.detections if item.id == 1)
        self.assertEqual(completed_item.status.value, "completed")
        completed_prediction = completed_item.prediction
        self.assertIsNotNone(completed_prediction)
        if completed_prediction is None:
            self.fail("Completed history rows should keep their prediction value")
        self.assertEqual(completed_prediction.value, "real")
        self.assertEqual(completed_item.confidence, 0.91)
        self.assertIsNone(completed_item.error_message)

    async def test_get_history_uses_video_metadata_and_honest_missing_model_label(self):
        created_at = datetime(2026, 3, 30, 12, 0, 0)
        records = [
            SimpleNamespace(
                id=3,
                file_path="uploads/20260330/broken-video.mp4",
                file_name="broken-video.mp4",
                file_type="video",
                prediction="failed",
                confidence=0.0,
                processing_time=0.17,
                model_name=None,
                model_type=None,
                status="failed",
                error_message="No usable ready/deployed registry model is available for detection",
                created_at=created_at,
                source_total_frames=300,
                source_fps=30.0,
                source_duration_seconds=10.0,
                sampled_frame_count=24,
                analyzed_frame_count=12,
                sampled_duration_seconds=4.0,
                model=None,
            )
        ]

        history = await DetectionService(FakeHistoryDB(records)).get_history(limit=10)

        self.assertEqual(history.total, 1)
        item = history.detections[0]
        self.assertEqual(item.model_name, "No model loaded")
        self.assertEqual(item.source_total_frames, 300)
        self.assertEqual(item.source_fps, 30.0)
        self.assertEqual(item.source_duration_seconds, 10.0)
        self.assertEqual(item.sampled_frame_count, 24)
        self.assertEqual(item.analyzed_frame_count, 12)
        self.assertEqual(item.sampled_duration_seconds, 4.0)


if __name__ == "__main__":
    unittest.main()
