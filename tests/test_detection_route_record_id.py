import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, mock_open, patch

from app.api.routes import detection as detection_routes
from app.schemas.detection import (
    BatchDetectionRequest,
    BatchDetectionResponse,
    DetectionRequest,
    DetectionResponse,
    DetectionResult,
    VideoDetectionRequest,
    VideoDetectionResponse,
)


class FakeUploadFile:
    def __init__(self, filename, content):
        self.filename = filename
        self._content = content

    async def read(self):
        return self._content


class DummyBackgroundTasks:
    def add_task(self, *_args, **_kwargs):
        return None


class DetectionRouteRecordIdTests(unittest.IsolatedAsyncioTestCase):
    def build_detection_result(self):
        return DetectionResult(
            prediction="fake",
            confidence=0.97,
            processing_time=0.42,
        )

    async def test_detect_route_image_response_preserves_record_id(self):
        created_at = datetime(2026, 3, 30, 10, 0, 0)
        service_result = DetectionResponse(
            success=True,
            record_id=101,
            file_info={"name": "stored-image.png", "type": "image", "size": 12},
            result=self.build_detection_result(),
            processing_time=0.42,
            created_at=created_at,
        )

        with (
            patch("app.api.routes.detection.os.makedirs"),
            patch("app.api.routes.detection.uuid.uuid4", return_value="image-upload"),
            patch("builtins.open", mock_open()),
            patch("app.services.detection_service.DetectionService") as service_cls,
        ):
            service_cls.return_value.detect_file = AsyncMock(
                return_value=service_result
            )

            response = await detection_routes.detect_deepfake(
                background_tasks=DummyBackgroundTasks(),
                file=FakeUploadFile("sample.png", b"image-bytes"),
                request=DetectionRequest(model_type="vit"),
                db=object(),
                current_user="route-test-user",
            )

        self.assertTrue(response.success)
        self.assertEqual(response.record_id, 101)
        result = response.result
        if result is None:
            self.fail("Image /detect response should include a result on success")
        self.assertEqual(result.prediction, "fake")
        service_cls.return_value.detect_file.assert_awaited_once()
        await_args = service_cls.return_value.detect_file.await_args
        if await_args is None:
            self.fail("detect_file should have been awaited with keyword arguments")
        self.assertEqual(
            await_args.kwargs["original_file_name"],
            "sample.png",
        )

    async def test_detect_route_video_branch_maps_record_id_from_video_response(self):
        created_at = datetime(2026, 3, 30, 10, 5, 0)
        aggregated_result = self.build_detection_result()
        service_result = VideoDetectionResponse(
            success=True,
            record_id=202,
            video_info={
                "source_total_frames": 240,
                "source_fps": 24.0,
                "source_duration_seconds": 10.0,
                "sampled_frame_count": 24,
                "analyzed_frame_count": 12,
                "sampled_duration_seconds": 3.5,
                "total_frames": 24,
                "processed_frames": 12,
                "duration": 3.5,
            },
            aggregated_result=aggregated_result,
            summary={
                "source_total_frames": 240,
                "sampled_frame_count": 24,
                "analyzed_frame_count": 12,
            },
            processing_time=1.5,
            created_at=created_at,
        )

        with (
            patch("app.api.routes.detection.os.makedirs"),
            patch("app.api.routes.detection.os.path.getsize", return_value=4096),
            patch("app.api.routes.detection.uuid.uuid4", return_value="video-upload"),
            patch("builtins.open", mock_open()),
            patch("app.services.detection_service.DetectionService") as service_cls,
        ):
            service_cls.return_value.detect_video = AsyncMock(
                return_value=service_result
            )

            response = await detection_routes.detect_deepfake(
                background_tasks=DummyBackgroundTasks(),
                file=FakeUploadFile("clip.mp4", b"video-bytes"),
                request=DetectionRequest(model_type="vit"),
                db=object(),
                current_user="route-test-user",
            )

        self.assertTrue(response.success)
        self.assertEqual(response.record_id, 202)
        self.assertEqual(response.result, aggregated_result)
        self.assertEqual(response.file_info["name"], "clip.mp4")
        self.assertEqual(response.file_info["type"], "video")
        self.assertEqual(response.file_info["source_total_frames"], 240)
        self.assertEqual(response.file_info["source_fps"], 24.0)
        self.assertEqual(response.file_info["source_duration_seconds"], 10.0)
        self.assertEqual(response.file_info["sampled_frame_count"], 24)
        self.assertEqual(response.file_info["analyzed_frame_count"], 12)
        self.assertEqual(response.file_info["sampled_duration_seconds"], 3.5)
        self.assertEqual(response.file_info["total_frames"], 24)
        self.assertEqual(response.file_info["processed_frames"], 12)
        self.assertEqual(response.file_info["duration"], 3.5)
        self.assertIsNone(response.error_message)
        service_cls.return_value.detect_video.assert_awaited_once()
        await_args = service_cls.return_value.detect_video.await_args
        if await_args is None:
            self.fail("detect_video should have been awaited with keyword arguments")
        self.assertEqual(
            await_args.kwargs["original_file_name"],
            "clip.mp4",
        )

    async def test_detect_route_video_branch_keeps_explicit_failure_shape(self):
        created_at = datetime(2026, 3, 30, 10, 10, 0)
        service_result = VideoDetectionResponse(
            success=False,
            record_id=404,
            video_info={
                "source_total_frames": 120,
                "source_fps": 30.0,
                "source_duration_seconds": 4.0,
                "total_frames": 0,
                "processed_frames": 0,
                "duration": 0.0,
            },
            summary={},
            error_message="video processing failed",
            processing_time=0.75,
            created_at=created_at,
        )

        with (
            patch("app.api.routes.detection.os.makedirs"),
            patch("app.api.routes.detection.os.path.getsize", return_value=512),
            patch("app.api.routes.detection.uuid.uuid4", return_value="video-failure"),
            patch("builtins.open", mock_open()),
            patch("app.services.detection_service.DetectionService") as service_cls,
        ):
            service_cls.return_value.detect_video = AsyncMock(
                return_value=service_result
            )

            response = await detection_routes.detect_deepfake(
                background_tasks=DummyBackgroundTasks(),
                file=FakeUploadFile("broken.mp4", b"video-bytes"),
                request=DetectionRequest(model_type="vit"),
                db=object(),
                current_user="route-test-user",
            )

        self.assertFalse(response.success)
        self.assertEqual(response.record_id, 404)
        self.assertIsNone(response.result)
        self.assertEqual(response.error_message, "video processing failed")
        self.assertEqual(response.file_info["name"], "broken.mp4")
        self.assertEqual(response.file_info["size"], 512)
        self.assertEqual(response.file_info["source_total_frames"], 120)
        self.assertEqual(response.file_info["source_duration_seconds"], 4.0)

    async def test_detect_route_image_raises_409_for_model_unavailable(self):
        created_at = datetime(2026, 3, 30, 10, 12, 0)
        service_result = DetectionResponse(
            success=False,
            record_id=515,
            file_info={"name": "stored-image.png", "type": "image", "size": 12},
            error_message="No usable ready/deployed registry model is available for detection (model_type='vit').",
            error_code="model_unavailable",
            processing_time=0.11,
            created_at=created_at,
        )

        with (
            patch("app.api.routes.detection.os.makedirs"),
            patch(
                "app.api.routes.detection.uuid.uuid4", return_value="image-unavailable"
            ),
            patch("builtins.open", mock_open()),
            patch("app.api.routes.detection.os.path.exists", return_value=True),
            patch("app.api.routes.detection.os.remove") as remove_mock,
            patch("app.services.detection_service.DetectionService") as service_cls,
        ):
            service_cls.return_value.detect_file = AsyncMock(
                return_value=service_result
            )

            with self.assertRaises(detection_routes.HTTPException) as exc_info:
                await detection_routes.detect_deepfake(
                    background_tasks=DummyBackgroundTasks(),
                    file=FakeUploadFile("sample.png", b"image-bytes"),
                    request=DetectionRequest(model_type="vit"),
                    db=object(),
                    current_user="route-test-user",
                )

        self.assertEqual(exc_info.exception.status_code, 409)
        self.assertEqual(exc_info.exception.detail["error_code"], "model_unavailable")
        self.assertEqual(exc_info.exception.detail["record_id"], 515)
        remove_mock.assert_called_once()

    async def test_detect_video_route_raises_409_for_model_unavailable(self):
        created_at = datetime(2026, 3, 30, 10, 13, 0)
        service_result = VideoDetectionResponse(
            success=False,
            record_id=616,
            video_info={"name": "stored-video.mp4", "size": 12},
            summary={},
            error_message="No usable ready/deployed registry model is available for detection (model_type='vit').",
            error_code="model_unavailable",
            processing_time=0.25,
            created_at=created_at,
        )

        with (
            patch("app.api.routes.detection.os.makedirs"),
            patch(
                "app.api.routes.detection.uuid.uuid4", return_value="video-unavailable"
            ),
            patch("builtins.open", mock_open()),
            patch("app.api.routes.detection.os.path.exists", return_value=True),
            patch("app.api.routes.detection.os.remove") as remove_mock,
            patch("app.services.detection_service.DetectionService") as service_cls,
        ):
            service_cls.return_value.detect_video = AsyncMock(
                return_value=service_result
            )

            with self.assertRaises(detection_routes.HTTPException) as exc_info:
                await detection_routes.detect_deepfake_video(
                    background_tasks=DummyBackgroundTasks(),
                    file=FakeUploadFile("video.mp4", b"video-bytes"),
                    request=VideoDetectionRequest(model_type="vit"),
                    db=object(),
                )

        self.assertEqual(exc_info.exception.status_code, 409)
        self.assertEqual(exc_info.exception.detail["error_code"], "model_unavailable")
        self.assertEqual(exc_info.exception.detail["record_id"], 616)
        remove_mock.assert_called_once()

    async def test_detect_video_route_preserves_record_id(self):
        created_at = datetime(2026, 3, 30, 10, 15, 0)
        service_result = VideoDetectionResponse(
            success=True,
            record_id=303,
            video_info={"name": "stored-video.mp4", "processed_frames": 8},
            aggregated_result=self.build_detection_result(),
            summary={"processed_frames": 8},
            processing_time=1.2,
            created_at=created_at,
        )

        with (
            patch("app.api.routes.detection.os.makedirs"),
            patch(
                "app.api.routes.detection.uuid.uuid4", return_value="dedicated-video"
            ),
            patch("builtins.open", mock_open()),
            patch("app.services.detection_service.DetectionService") as service_cls,
        ):
            service_cls.return_value.detect_video = AsyncMock(
                return_value=service_result
            )

            response = await detection_routes.detect_deepfake_video(
                background_tasks=DummyBackgroundTasks(),
                file=FakeUploadFile("dedicated.mp4", b"video-bytes"),
                request=VideoDetectionRequest(model_type="vit"),
                db=object(),
            )

        self.assertTrue(response.success)
        self.assertEqual(response.record_id, 303)
        service_cls.return_value.detect_video.assert_awaited_once()
        await_args = service_cls.return_value.detect_video.await_args
        if await_args is None:
            self.fail("detect_video should have been awaited with keyword arguments")
        self.assertEqual(
            await_args.kwargs["original_file_name"],
            "dedicated.mp4",
        )

    async def test_detect_batch_route_cleans_up_on_model_unavailable(self):
        created_at = datetime(2026, 3, 30, 10, 20, 0)
        batch_result = BatchDetectionResponse(
            success=False,
            total_files=2,
            processed_files=0,
            failed_files=2,
            results=[
                DetectionResponse(
                    success=False,
                    record_id=701,
                    file_info={"name": "stored-1.png", "type": "image", "size": 10},
                    error_message="No usable ready/deployed registry model is available for detection (model_type='vit').",
                    error_code="model_unavailable",
                    processing_time=0.1,
                    created_at=created_at,
                ),
                DetectionResponse(
                    success=False,
                    record_id=702,
                    file_info={"name": "stored-2.png", "type": "image", "size": 10},
                    error_message="No usable ready/deployed registry model is available for detection (model_type='vit').",
                    error_code="model_unavailable",
                    processing_time=0.1,
                    created_at=created_at,
                ),
            ],
            summary={},
            processing_time=0.2,
            created_at=created_at,
        )

        with (
            patch("app.api.routes.detection.os.makedirs"),
            patch(
                "app.api.routes.detection.uuid.uuid4",
                side_effect=["batch-image-1", "batch-image-2"],
            ),
            patch("builtins.open", mock_open()),
            patch("app.api.routes.detection.os.path.exists", return_value=True),
            patch("app.api.routes.detection.os.remove") as remove_mock,
            patch("app.services.detection_service.DetectionService") as service_cls,
        ):
            service_cls.return_value.detect_batch = AsyncMock(return_value=batch_result)

            with self.assertRaises(detection_routes.HTTPException) as exc_info:
                await detection_routes.detect_deepfake_batch(
                    background_tasks=DummyBackgroundTasks(),
                    files=[
                        FakeUploadFile("one.png", b"img-1"),
                        FakeUploadFile("two.png", b"img-2"),
                    ],
                    request=BatchDetectionRequest(model_type="vit"),
                    db=object(),
                    current_user=None,
                )

        self.assertEqual(exc_info.exception.status_code, 409)
        self.assertEqual(remove_mock.call_count, 2)

    async def test_detect_batch_route_does_not_mask_mixed_failures_as_model_unavailable(
        self,
    ):
        created_at = datetime(2026, 3, 30, 10, 25, 0)
        mixed_result = SimpleNamespace(
            success=False,
            total_files=2,
            processed_files=0,
            failed_files=2,
            results=[
                RuntimeError("unsupported file type"),
                DetectionResponse(
                    success=False,
                    record_id=801,
                    file_info={"name": "stored-2.png", "type": "image", "size": 10},
                    error_message="No usable ready/deployed registry model is available for detection (model_type='vit').",
                    error_code="model_unavailable",
                    processing_time=0.1,
                    created_at=created_at,
                ),
            ],
            summary={},
            processing_time=0.2,
            created_at=created_at,
        )

        with (
            patch("app.api.routes.detection.os.makedirs"),
            patch(
                "app.api.routes.detection.uuid.uuid4",
                side_effect=["batch-mixed-1", "batch-mixed-2"],
            ),
            patch("builtins.open", mock_open()),
            patch("app.services.detection_service.DetectionService") as service_cls,
        ):
            service_cls.return_value.detect_batch = AsyncMock(return_value=mixed_result)

            response = await detection_routes.detect_deepfake_batch(
                background_tasks=DummyBackgroundTasks(),
                files=[
                    FakeUploadFile("one.png", b"img-1"),
                    FakeUploadFile("two.png", b"img-2"),
                ],
                request=BatchDetectionRequest(model_type="vit"),
                db=object(),
                current_user=None,
            )

        self.assertIs(response, mixed_result)


if __name__ == "__main__":
    unittest.main()
