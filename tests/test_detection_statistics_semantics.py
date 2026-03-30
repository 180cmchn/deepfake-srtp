import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sqlalchemy import text
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.database_models import DetectionResult as DetectionResultModel
from app.schemas.detection import (
    BatchDetectionRequest,
    DetectionResponse,
    DetectionResult,
)
from app.services.detection_service import DetectionService


class FakeStatisticsQuery:
    def __init__(self, records):
        self._records = list(records)

    def filter(self, *_args, **_kwargs):
        return self

    def outerjoin(self, *_args, **_kwargs):
        return self

    def all(self):
        return list(self._records)


class FakeStatisticsDB:
    def __init__(self, records):
        self._records = list(records)

    def query(self, _model):
        return FakeStatisticsQuery(self._records)


class DummyBackgroundTasks:
    def add_task(self, *_args, **_kwargs):
        return None


class DetectionStatisticsSemanticsTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_statistics_is_null_safe_and_status_aware(self):
        created_at = datetime(2026, 3, 30, 12, 0, 0)
        records = [
            SimpleNamespace(
                id=1,
                prediction="real",
                confidence=0.9,
                processing_time=0.2,
                model_type="vit",
                model_name="registry-vit",
                file_type="image",
                status="completed",
                error_message=None,
                created_at=created_at,
                model=None,
            ),
            SimpleNamespace(
                id=2,
                prediction="fake",
                confidence=0.6,
                processing_time=None,
                model_type="resnet",
                model_name="registry-resnet",
                file_type="video",
                status="completed",
                error_message=None,
                created_at=created_at,
                model=None,
            ),
            SimpleNamespace(
                id=3,
                prediction="failed",
                confidence=0.0,
                processing_time=0.5,
                model_type="vit",
                model_name="registry-vit",
                file_type="image",
                status="failed",
                error_message="prediction exploded",
                created_at=created_at,
                model=None,
            ),
        ]

        stats = await DetectionService(
            FakeStatisticsDB(records)
        ).get_statistics_filtered()

        self.assertEqual(stats.total_detections, 3)
        self.assertEqual(stats.real_detections, 1)
        self.assertEqual(stats.fake_detections, 1)
        self.assertEqual(stats.failed_detections, 1)
        self.assertAlmostEqual(stats.average_confidence, 0.75)
        self.assertAlmostEqual(stats.average_processing_time, 0.35)
        self.assertEqual(stats.detections_by_status["completed"], 2)
        self.assertEqual(stats.detections_by_status["failed"], 1)
        self.assertEqual(stats.detections_by_file_type["image"], 2)
        self.assertEqual(stats.detections_by_file_type["video"], 1)
        self.assertEqual(stats.confidence_distribution["0.6-0.8"], 1)
        self.assertEqual(stats.confidence_distribution["0.8-1.0"], 1)
        self.assertEqual(sum(stats.confidence_distribution.values()), 2)
        self.assertEqual(stats.daily_detections["2026-03-30"], 3)

    async def test_history_and_statistics_share_filter_scope_for_failed_rows(self):
        engine = create_engine("sqlite:///:memory:")
        TestingSessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=engine
        )
        Base.metadata.create_all(bind=engine)
        session = TestingSessionLocal()

        try:
            created_at = datetime(2026, 3, 30, 12, 30, 0)
            session.add_all(
                [
                    DetectionResultModel(
                        file_path="uploads/real.png",
                        file_name="real.png",
                        file_type="image",
                        prediction="real",
                        confidence=0.91,
                        processing_time=0.21,
                        model_name="registry-vit",
                        model_type="vit",
                        status="completed",
                        created_at=created_at,
                    ),
                    DetectionResultModel(
                        file_path="uploads/failed.png",
                        file_name="failed.png",
                        file_type="image",
                        prediction="failed",
                        confidence=0.0,
                        processing_time=0.17,
                        model_name="registry-vit",
                        model_type="vit",
                        status="failed",
                        error_message="prediction exploded",
                        created_at=created_at,
                    ),
                ]
            )
            session.commit()

            service = DetectionService(session)
            history = await service.get_history(status="failed", limit=10)
            stats = await service.get_statistics_filtered(status="failed")

            self.assertEqual(history.total, 1)
            self.assertEqual(len(history.detections), 1)
            self.assertEqual(history.total, stats.total_detections)
            self.assertEqual(stats.failed_detections, 1)
            self.assertEqual(stats.real_detections, 0)
            self.assertEqual(stats.fake_detections, 0)
        finally:
            session.close()
            Base.metadata.drop_all(bind=engine)

    async def test_legacy_null_status_row_is_still_filterable_as_failed(self):
        engine = create_engine("sqlite:///:memory:")
        TestingSessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=engine
        )
        Base.metadata.create_all(bind=engine)
        session = TestingSessionLocal()

        try:
            created_at = datetime(2026, 3, 30, 13, 0, 0)
            legacy_row = DetectionResultModel(
                file_path="uploads/legacy-failed.png",
                file_name="legacy-failed.png",
                file_type="image",
                prediction="failed",
                confidence=0.0,
                processing_time=0.15,
                model_name=None,
                model_type=None,
                status="completed",
                error_message="legacy failure",
                created_at=created_at,
            )
            session.add(legacy_row)
            session.commit()
            session.execute(
                text("UPDATE detection_results SET status = NULL WHERE id = :id"),
                {"id": legacy_row.id},
            )
            session.commit()

            service = DetectionService(session)
            history = await service.get_history(status="failed", limit=10)
            stats = await service.get_statistics_filtered(status="failed")

            self.assertEqual(history.total, 1)
            self.assertEqual(len(history.detections), 1)
            self.assertEqual(history.detections[0].status.value, "failed")
            self.assertEqual(stats.total_detections, 1)
            self.assertEqual(stats.failed_detections, 1)
        finally:
            session.close()
            Base.metadata.drop_all(bind=engine)

    async def test_legacy_null_status_row_with_blank_error_stays_completed(self):
        engine = create_engine("sqlite:///:memory:")
        TestingSessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=engine
        )
        Base.metadata.create_all(bind=engine)
        session = TestingSessionLocal()

        try:
            created_at = datetime(2026, 3, 30, 13, 30, 0)
            legacy_row = DetectionResultModel(
                file_path="uploads/legacy-completed.png",
                file_name="legacy-completed.png",
                file_type="image",
                prediction="real",
                confidence=0.82,
                processing_time=0.2,
                model_name="registry-vit",
                model_type="vit",
                status="completed",
                error_message="   ",
                created_at=created_at,
            )
            session.add(legacy_row)
            session.commit()
            session.execute(
                text("UPDATE detection_results SET status = NULL WHERE id = :id"),
                {"id": legacy_row.id},
            )
            session.commit()

            service = DetectionService(session)
            failed_history = await service.get_history(status="failed", limit=10)
            completed_history = await service.get_history(status="completed", limit=10)
            failed_stats = await service.get_statistics_filtered(status="failed")
            completed_stats = await service.get_statistics_filtered(status="completed")

            self.assertEqual(failed_history.total, 0)
            self.assertEqual(failed_stats.total_detections, 0)
            self.assertEqual(completed_history.total, 1)
            self.assertEqual(completed_stats.total_detections, 1)
        finally:
            session.close()
            Base.metadata.drop_all(bind=engine)

    async def test_failed_status_row_does_not_count_as_real_or_fake(self):
        created_at = datetime(2026, 3, 30, 14, 0, 0)
        records = [
            SimpleNamespace(
                id=4,
                prediction="real",
                confidence=0.99,
                processing_time=0.11,
                model_type="vit",
                model_name="registry-vit",
                file_type="image",
                status="failed",
                error_message="inconsistent legacy failure",
                created_at=created_at,
                model=None,
            )
        ]

        stats = await DetectionService(
            FakeStatisticsDB(records)
        ).get_statistics_filtered()

        self.assertEqual(stats.total_detections, 1)
        self.assertEqual(stats.real_detections, 0)
        self.assertEqual(stats.fake_detections, 0)
        self.assertEqual(stats.failed_detections, 1)

    async def test_detect_batch_normalizes_raw_exceptions_into_detection_responses(
        self,
    ):
        service = DetectionService(FakeStatisticsDB([]))
        success_response = DetectionResponse(
            success=True,
            record_id=1,
            file_info={"name": "ok.png", "type": "image", "size": 12},
            result=DetectionResult(
                prediction="real", confidence=0.88, processing_time=0.1
            ),
            processing_time=0.1,
            created_at=datetime(2026, 3, 30, 14, 30, 0),
        )

        with patch.object(
            service,
            "_detect_batch_file",
            new=AsyncMock(
                side_effect=[RuntimeError("unsupported file type"), success_response]
            ),
        ):
            batch_result = await service.detect_batch(
                file_paths=["/tmp/bad.xyz", "/tmp/good.png"],
                request=BatchDetectionRequest(
                    model_type="vit", parallel_processing=False
                ),
                background_tasks=DummyBackgroundTasks(),
                original_file_names={
                    "/tmp/bad.xyz": "bad.xyz",
                    "/tmp/good.png": "good.png",
                },
            )

        self.assertEqual(batch_result.total_files, 2)
        self.assertEqual(batch_result.processed_files, 1)
        self.assertEqual(batch_result.failed_files, 1)
        self.assertTrue(
            all(isinstance(item, DetectionResponse) for item in batch_result.results)
        )
        failed_item = batch_result.results[0]
        self.assertFalse(failed_item.success)
        self.assertEqual(failed_item.file_info["name"], "bad.xyz")
        self.assertEqual(failed_item.file_info["type"], "unknown")
        self.assertEqual(failed_item.error_message, "unsupported file type")


if __name__ == "__main__":
    unittest.main()
