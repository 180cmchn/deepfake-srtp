import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, patch

from app.models.database_models import AuditLog, DetectionResult as DetectionResultModel
from app.schemas.detection import DetectionRequest, VideoDetectionRequest
from app.services.detection_service import DetectionService


class DummyBackgroundTasks:
    def __init__(self):
        self.tasks = []

    def add_task(self, func, *args, **kwargs):
        self.tasks.append((func, args, kwargs))


class RecordingDB:
    def __init__(self, query_result=None):
        self.added = []
        self.commit_count = 0
        self.rollback_count = 0
        self.refresh_count = 0
        self.query_result = query_result

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.commit_count += 1

    def rollback(self):
        self.rollback_count += 1

    def query(self, _model):
        return QueryStub(self.query_result)

    def refresh(self, obj):
        self.refresh_count += 1
        if getattr(obj, "id", None) is None:
            obj.id = len(self.added)
        if getattr(obj, "created_at", None) is None:
            obj.created_at = datetime.now()


class QueryStub:
    def __init__(self, result):
        self.result = result

    def filter(self, *_args, **_kwargs):
        return self

    def first(self):
        return self.result


class AuditCommitFailingDB(RecordingDB):
    def commit(self):
        self.commit_count += 1
        if self.commit_count == 2:
            raise RuntimeError("audit commit failed")


def build_loaded_model(*, model_id=7, model_name="registry-vit", model_type="vit"):
    return {
        "model": object(),
        "model_id": model_id,
        "model_name": model_name,
        "model_type": model_type,
        "input_size": 224,
        "source": "registry",
        "status": "ready",
        "weight_state": "checkpoint_loaded",
    }


class DetectionServicePersistenceAuditTests(unittest.IsolatedAsyncioTestCase):
    def build_audit_context(self, *, user_id="audit-user"):
        return {
            "user_id": user_id,
            "ip_address": "203.0.113.10",
            "user_agent": "unit-test-agent",
        }

    def assert_audit_log(
        self,
        audit_log,
        *,
        action,
        resource_id,
        user_id,
        status,
        file_name,
        file_type,
        prediction=None,
        error_message=None,
    ):
        self.assertIsInstance(audit_log, AuditLog)
        self.assertEqual(audit_log.action, action)
        self.assertEqual(audit_log.resource_type, "detection_result")
        self.assertEqual(audit_log.resource_id, str(resource_id))
        self.assertEqual(audit_log.user_id, user_id)
        self.assertEqual(audit_log.ip_address, "203.0.113.10")
        self.assertEqual(audit_log.user_agent, "unit-test-agent")
        self.assertEqual(audit_log.details["status"], status)
        self.assertEqual(audit_log.details["file_name"], file_name)
        self.assertEqual(audit_log.details["file_type"], file_type)
        if prediction is not None:
            self.assertEqual(audit_log.details["prediction"], prediction)
        if error_message is not None:
            self.assertEqual(audit_log.details["error_message"], error_message)

    async def test_detect_file_persists_original_filename_stored_path_and_model_linkage_on_success(
        self,
    ):
        db = RecordingDB()
        service = DetectionService(db)
        background_tasks = DummyBackgroundTasks()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(b"image-bytes-for-success")
            file_path = temp_file.name

        try:
            with (
                patch.object(
                    service,
                    "_load_model",
                    new=AsyncMock(return_value=build_loaded_model()),
                ),
                patch.object(service, "_preprocess_image", return_value=object()),
                patch.object(
                    service,
                    "_predict_tensor",
                    new=AsyncMock(
                        return_value={"probabilities": {"fake": 0.9, "real": 0.1}}
                    ),
                ),
            ):
                response = await service.detect_file(
                    file_path=file_path,
                    request=DetectionRequest(
                        model_type="vit", return_probabilities=True
                    ),
                    background_tasks=DummyBackgroundTasks(),
                    original_file_name="original-upload.png",
                    audit_context=self.build_audit_context(),
                )
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

        self.assertTrue(response.success)
        self.assertEqual(response.record_id, 1)
        self.assertEqual(len(db.added), 2)

        record = db.added[0]
        audit_log = db.added[1]
        self.assertEqual(record.file_path, file_path)
        self.assertEqual(record.file_name, "original-upload.png")
        self.assertEqual(record.file_type, "image")
        self.assertEqual(record.status, "completed")
        self.assertIsNone(record.error_message)
        self.assertEqual(record.file_size, len(b"image-bytes-for-success"))
        self.assertEqual(record.model_id, 7)
        self.assertEqual(record.model_name, "registry-vit")
        self.assertEqual(record.model_type, "vit")
        self.assertEqual(record.prediction, "fake")
        self.assertAlmostEqual(record.confidence, 0.9)
        self.assert_audit_log(
            audit_log,
            action="detect",
            resource_id=record.id,
            user_id="audit-user",
            status="completed",
            file_name="original-upload.png",
            file_type="image",
            prediction="fake",
        )

    async def test_detect_file_persists_failed_status_error_and_loaded_model_linkage(
        self,
    ):
        db = RecordingDB()
        service = DetectionService(db)
        background_tasks = DummyBackgroundTasks()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(b"image-bytes-for-failure")
            file_path = temp_file.name

        try:
            with (
                patch.object(
                    service,
                    "_load_model",
                    new=AsyncMock(return_value=build_loaded_model(model_id=11)),
                ),
                patch.object(service, "_preprocess_image", return_value=object()),
                patch.object(
                    service,
                    "_predict_tensor",
                    new=AsyncMock(side_effect=RuntimeError("prediction exploded")),
                ),
            ):
                response = await service.detect_file(
                    file_path=file_path,
                    request=DetectionRequest(model_type="vit"),
                    background_tasks=background_tasks,
                    original_file_name="broken-upload.png",
                    audit_context=self.build_audit_context(),
                )
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

        self.assertFalse(response.success)
        self.assertIn("prediction exploded", response.error_message or "")
        self.assertIsNone(response.error_code)
        self.assertEqual(len(db.added), 2)

        record = db.added[0]
        audit_log = db.added[1]
        self.assertEqual(response.record_id, record.id)
        self.assertEqual(len(background_tasks.tasks), 1)
        self.assertEqual(background_tasks.tasks[0][1][0], file_path)
        self.assertEqual(record.file_path, file_path)
        self.assertEqual(record.file_name, "broken-upload.png")
        self.assertEqual(record.file_type, "image")
        self.assertEqual(record.status, "failed")
        self.assertEqual(record.error_message, "prediction exploded")
        self.assertEqual(record.file_size, len(b"image-bytes-for-failure"))
        self.assertEqual(record.model_id, 11)
        self.assertEqual(record.model_name, "registry-vit")
        self.assertEqual(record.model_type, "vit")
        self.assertEqual(record.prediction, "failed")
        self.assertEqual(record.confidence, 0.0)
        self.assertGreaterEqual(record.processing_time, 0.0)
        self.assertEqual(response.created_at, record.created_at)
        self.assert_audit_log(
            audit_log,
            action="detect",
            resource_id=record.id,
            user_id="audit-user",
            status="failed",
            file_name="broken-upload.png",
            file_type="image",
            prediction="failed",
            error_message="prediction exploded",
        )

    async def test_detect_video_persists_failed_status_error_and_original_filename(
        self,
    ):
        db = RecordingDB()
        service = DetectionService(db)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(b"video-bytes-for-failure")
            file_path = temp_file.name

        try:
            with (
                patch.object(
                    service,
                    "_extract_frames",
                    new=AsyncMock(return_value=[(object(), 0.0), (object(), 1.0)]),
                ),
                patch.object(
                    service,
                    "_load_model",
                    new=AsyncMock(
                        return_value=build_loaded_model(
                            model_id=21,
                            model_name="registry-video-vit",
                        )
                    ),
                ),
                patch.object(service, "_preprocess_frame", return_value=object()),
                patch.object(
                    service,
                    "_predict_tensor",
                    new=AsyncMock(
                        side_effect=[
                            RuntimeError("frame failure"),
                            RuntimeError("frame failure"),
                        ]
                    ),
                ),
            ):
                response = await service.detect_video(
                    video_path=file_path,
                    request=VideoDetectionRequest(model_type="vit"),
                    background_tasks=DummyBackgroundTasks(),
                    original_file_name="broken-video.mp4",
                    audit_context=self.build_audit_context(),
                )
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

        self.assertFalse(response.success)
        self.assertEqual(len(db.added), 2)

        record = db.added[0]
        audit_log = db.added[1]
        self.assertEqual(response.record_id, record.id)
        self.assertEqual(record.file_path, file_path)
        self.assertEqual(record.file_name, "broken-video.mp4")
        self.assertEqual(record.file_type, "video")
        self.assertEqual(record.status, "failed")
        self.assertEqual(
            record.error_message, "No frames could be processed successfully"
        )
        self.assertEqual(record.file_size, len(b"video-bytes-for-failure"))
        self.assertEqual(record.model_id, 21)
        self.assertEqual(record.model_name, "registry-video-vit")
        self.assertEqual(record.model_type, "vit")
        self.assertEqual(record.prediction, "failed")
        self.assertEqual(record.confidence, 0.0)
        self.assertEqual(response.created_at, record.created_at)
        self.assert_audit_log(
            audit_log,
            action="detect_video",
            resource_id=record.id,
            user_id="audit-user",
            status="failed",
            file_name="broken-video.mp4",
            file_type="video",
            prediction="failed",
            error_message="No frames could be processed successfully",
        )

    async def test_delete_detection_record_writes_audit_log(self):
        existing_record = DetectionResultModel(
            id=99,
            file_path="uploads/existing.png",
            file_name="existing.png",
            file_type="image",
            prediction="real",
            confidence=0.82,
            processing_time=0.31,
            file_size=1234,
            model_id=8,
            model_name="registry-vit",
            model_type="vit",
            status="completed",
            del_flag=0,
        )
        db = RecordingDB(query_result=existing_record)
        service = DetectionService(db)

        success = await service.delete_detection_record(
            99,
            audit_context=self.build_audit_context(user_id="delete-user"),
        )

        self.assertTrue(success)
        self.assertEqual(existing_record.del_flag, 1)
        self.assertEqual(len(db.added), 1)
        self.assertEqual(db.commit_count, 2)
        self.assert_audit_log(
            db.added[0],
            action="delete_detection_result",
            resource_id=99,
            user_id="delete-user",
            status="deleted",
            file_name="existing.png",
            file_type="image",
            prediction="real",
        )

    async def test_detect_file_audit_failure_does_not_break_success_response(self):
        db = AuditCommitFailingDB()
        service = DetectionService(db)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(b"image-bytes-for-audit-failure")
            file_path = temp_file.name

        try:
            with (
                patch.object(
                    service,
                    "_load_model",
                    new=AsyncMock(return_value=build_loaded_model()),
                ),
                patch.object(service, "_preprocess_image", return_value=object()),
                patch.object(
                    service,
                    "_predict_tensor",
                    new=AsyncMock(
                        return_value={"probabilities": {"fake": 0.9, "real": 0.1}}
                    ),
                ),
            ):
                response = await service.detect_file(
                    file_path=file_path,
                    request=DetectionRequest(model_type="vit"),
                    background_tasks=DummyBackgroundTasks(),
                    original_file_name="audit-failure.png",
                    audit_context=self.build_audit_context(),
                )
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

        self.assertTrue(response.success)
        self.assertEqual(response.record_id, 1)
        self.assertEqual(len(db.added), 2)
        self.assertEqual(db.rollback_count, 1)


if __name__ == "__main__":
    unittest.main()
