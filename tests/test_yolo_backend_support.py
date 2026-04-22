import unittest
from unittest.mock import patch
from datetime import datetime, timedelta

import torch
import torch.nn as nn

from app.core.config import Settings, get_supported_model_types
from app.models.ml_models import (
    ModelRegistry,
    create_model,
)
from app.schemas.detection import (
    BatchDetectionRequest,
    DetectionConfig,
    DetectionRequest,
    VideoDetectionRequest,
)
from app.schemas.models import ModelCreate
from app.schemas.training import TrainingJobCreate, TrainingParameters
from app.services.detection_service import DetectionService
from app.services.training_service import TrainingService


class _DummyUltralyticsClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(3 * 224 * 224, 16)
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x):
        flattened = self.flatten(x)
        hidden = torch.relu(self.hidden(flattened))
        return self.classifier(hidden)


class _FakeQuery:
    def filter(self, *_args, **_kwargs):
        return self

    def count(self):
        return 0


class _FakeDB:
    def __init__(self):
        self.added = None

    def query(self, _model):
        return _FakeQuery()

    def add(self, obj):
        self.added = obj

    def commit(self):
        return None

    def refresh(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = 123

    def rollback(self):
        return None


class _FakeModelRecord:
    def __init__(self, num_classes=2, parameters=None):
        self.num_classes = num_classes
        self.parameters = parameters or {}


class YoloSchemaSupportTests(unittest.TestCase):
    def test_default_supported_models_include_yolo(self):
        settings = Settings(_env_file=None)

        self.assertIn("yolo", settings.SUPPORTED_MODELS)
        self.assertIn("yolo", get_supported_model_types())

    def test_training_and_model_schemas_accept_yolo(self):
        training_job = TrainingJobCreate(
            name="yolo-training-job",
            model_type="yolo",
            dataset_path="/tmp/dataset",
        )
        model = ModelCreate(
            name="yolo-model",
            model_type="yolo",
            version="1.0.0",
            file_path="/tmp/yolo-model.pth",
        )

        self.assertEqual(training_job.model_type, "yolo")
        self.assertEqual(model.model_type, "yolo")

    def test_detection_schemas_accept_yolo(self):
        detection_request = DetectionRequest(model_type="yolo")
        batch_request = BatchDetectionRequest(model_type="yolo")
        video_request = VideoDetectionRequest(model_type="yolo")
        detection_config = DetectionConfig(default_model_type="yolo")

        self.assertEqual(detection_request.model_type, "yolo")
        self.assertEqual(batch_request.model_type, "yolo")
        self.assertEqual(video_request.model_type, "yolo")
        self.assertEqual(detection_config.default_model_type, "yolo")


class YoloModelFactoryTests(unittest.TestCase):
    def test_model_registry_lists_yolo_and_factory_creates_it(self):
        self.assertIn("yolo", ModelRegistry.list_models())
        dummy_model = _DummyUltralyticsClassifier(num_classes=2)

        with patch(
            "app.models.ml_models._build_ultralytics_yolo_backbone",
            return_value=(
                dummy_model,
                dummy_model.classifier,
                16,
                {"status": "builtin_random_init", "weight_state": "random_init"},
            ),
        ):
            model = create_model("yolo", num_classes=2, pretrained=False)
            model.eval()
            with torch.no_grad():
                logits = model(torch.zeros(2, 3, 224, 224))

        self.assertEqual(model.__class__.__name__, "YOLOClassifier")
        self.assertEqual(tuple(logits.shape), (2, 2))

    def test_factory_creates_temporal_hybrid_yolo(self):
        dummy_model = _DummyUltralyticsClassifier(num_classes=2)

        with patch(
            "app.models.ml_models._build_ultralytics_yolo_backbone",
            return_value=(
                dummy_model,
                dummy_model.classifier,
                16,
                {"status": "builtin_random_init", "weight_state": "random_init"},
            ),
        ) as builder_mock:
            model = create_model(
                "yolo",
                num_classes=2,
                pretrained=False,
                video_temporal_enabled=True,
                yolo_model_variant="custom-yolo-cls",
            )

        self.assertEqual(model.__class__.__name__, "VideoTemporalHybridModel")
        self.assertEqual(model.backbone_type, "yolo")
        self.assertEqual(
            builder_mock.call_args.kwargs["model_variant"], "custom-yolo-cls"
        )

    def test_factory_creates_multi_region_temporal_hybrid_yolo(self):
        dummy_model = _DummyUltralyticsClassifier(num_classes=2)

        with patch(
            "app.models.ml_models._build_ultralytics_yolo_backbone",
            return_value=(
                dummy_model,
                dummy_model.classifier,
                16,
                {"status": "builtin_random_init", "weight_state": "random_init"},
            ),
        ):
            model = create_model(
                "yolo",
                num_classes=2,
                pretrained=False,
                video_temporal_enabled=True,
                face_region_mode="face_eyes_mouth_fusion",
            )
            model.eval()
            with torch.no_grad():
                logits = model(torch.zeros(2, 4, 3, 3, 224, 224))

        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertTrue(model.face_region_enabled)


class YoloTrainingModeSupportTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_job_allows_yolo_for_temporal_dataset(self):
        service = TrainingService(db=_FakeDB())
        fake_background_tasks = type(
            "FakeBackgroundTasks",
            (),
            {"add_task": lambda self, *args, **kwargs: None},
        )()

        with patch.object(
            service,
            "_db_to_response",
            return_value={"job_id": 123, "model_type": "yolo"},
        ):
            response = await service.create_job(
                TrainingJobCreate(
                    name="yolo-video-job",
                    model_type="yolo",
                    dataset_path="/tmp/video-dataset",
                    parameters=TrainingParameters(yolo_model_variant="custom-yolo-cls"),
                ),
                fake_background_tasks,
                auto_start=False,
            )

        self.assertEqual(response["model_type"], "yolo")
        self.assertEqual(service.db.added.model_type, "yolo")


class YoloCheckpointCompatibilityTests(unittest.TestCase):
    def test_detection_service_rebuilds_temporal_yolo_checkpoint(self):
        service = DetectionService(db=None)
        dummy_model = _DummyUltralyticsClassifier(num_classes=2)
        checkpoint = {
            "model_type": "yolo",
            "num_classes": 2,
            "video_temporal_enabled": True,
            "temporal_hidden_size": 128,
            "temporal_num_layers": 2,
            "feature_projection_size": 64,
            "temporal_bidirectional": True,
            "temporal_attention_pooling": True,
            "yolo_model_variant": "custom-yolo-cls",
        }
        model_record = _FakeModelRecord(
            num_classes=2,
            parameters={
                "yolo_model_variant": "custom-yolo-cls",
                "temporal_hidden_size": 128,
                "temporal_num_layers": 2,
                "feature_projection_size": 64,
            },
        )

        with patch(
            "app.models.ml_models._build_ultralytics_yolo_backbone",
            return_value=(
                dummy_model,
                dummy_model.classifier,
                16,
                {"status": "builtin_random_init", "weight_state": "random_init"},
            ),
        ):
            model = service._build_model_from_checkpoint(
                "yolo",
                checkpoint,
                model_record,
            )

        self.assertEqual(model.__class__.__name__, "VideoTemporalHybridModel")
        self.assertEqual(model.backbone_type, "yolo")


class TrainingEtaEstimationTests(unittest.TestCase):
    def test_estimate_time_remaining_uses_progress_and_elapsed_time(self):
        service = TrainingService(db=None)
        started_at = datetime.now() - timedelta(seconds=30)

        eta = service._estimate_time_remaining(
            status="running",
            progress=50.0,
            started_at=started_at,
        )

        self.assertIsNotNone(eta)
        self.assertGreaterEqual(eta, 25)
        self.assertLessEqual(eta, 35)

    def test_estimate_time_remaining_returns_none_for_non_running_jobs(self):
        service = TrainingService(db=None)

        eta = service._estimate_time_remaining(
            status="completed",
            progress=75.0,
            started_at=datetime.now() - timedelta(seconds=30),
        )

        self.assertIsNone(eta)


if __name__ == "__main__":
    unittest.main()
