import os
import inspect
import tempfile
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.api.routes import detection as detection_routes
from app.schemas.detection import (
    BatchDetectionRequest,
    DetectionRequest,
    VideoDetectionRequest,
)
from app.schemas.models import ModelMetrics
from app.schemas.training import TrainingResults
from app.services.detection_service import DetectionService


class DummyBackgroundTasks:
    def __init__(self):
        self.tasks = []

    def add_task(self, func, *args, **kwargs):
        self.tasks.append((func, args, kwargs))


class ReportingContractTruthfulnessTests(unittest.TestCase):
    def test_detection_request_preprocess_fields_are_marked_deprecated(self):
        self.assertIn(
            "Deprecated",
            DetectionRequest.model_fields["preprocess"].description,
        )
        self.assertIn(
            "always applies preprocessing",
            DetectionRequest.model_fields["preprocess"].description,
        )
        self.assertIn(
            "Deprecated",
            BatchDetectionRequest.model_fields["preprocess"].description,
        )
        self.assertIn(
            "Deprecated",
            VideoDetectionRequest.model_fields["preprocess"].description,
        )

    def test_detection_route_form_preprocess_params_are_marked_deprecated(self):
        detection_preprocess = (
            inspect.signature(detection_routes.parse_detection_request)
            .parameters["preprocess"]
            .default
        )
        batch_preprocess = (
            inspect.signature(detection_routes.parse_batch_detection_request)
            .parameters["preprocess"]
            .default
        )
        video_preprocess = (
            inspect.signature(detection_routes.parse_video_detection_request)
            .parameters["preprocess"]
            .default
        )

        self.assertIn("Deprecated", detection_preprocess.description)
        self.assertIn("Deprecated", batch_preprocess.description)
        self.assertIn("Deprecated", video_preprocess.description)

    def test_training_results_descriptions_clarify_legacy_aliases(self):
        self.assertIn(
            "Deprecated legacy alias",
            TrainingResults.model_fields["accuracy"].description,
        )
        self.assertIn(
            "Deprecated legacy alias",
            TrainingResults.model_fields["loss"].description,
        )
        self.assertIn(
            "Video-level validation accuracy summary",
            TrainingResults.model_fields["val_accuracy"].description,
        )
        self.assertIn(
            "Video-level validation loss summary",
            TrainingResults.model_fields["val_loss"].description,
        )

    def test_model_metrics_descriptions_clarify_optional_advanced_fields(self):
        self.assertIn(
            "Primary accuracy metric",
            ModelMetrics.model_fields["accuracy"].description,
        )
        self.assertIn(
            "Optional advanced metric",
            ModelMetrics.model_fields["precision"].description,
        )
        self.assertIn(
            "Optional advanced metric",
            ModelMetrics.model_fields["recall"].description,
        )
        self.assertIn(
            "Optional advanced metric",
            ModelMetrics.model_fields["f1_score"].description,
        )
        self.assertIn(
            "Optional advanced report artifact",
            ModelMetrics.model_fields["classification_report"].description,
        )
        self.assertIn(
            "Optional advanced report artifact",
            ModelMetrics.model_fields["confusion_matrix"].description,
        )


class ReportingContractRuntimeBehaviorTests(unittest.IsolatedAsyncioTestCase):
    async def test_detect_file_still_applies_preprocessing_when_request_flag_is_false(
        self,
    ):
        service = DetectionService(db=None)
        background_tasks = DummyBackgroundTasks()
        persisted_result = SimpleNamespace(
            id=1,
            created_at=datetime(2026, 3, 31, 10, 30, 0),
        )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(b"reporting-contract-image")
            file_path = temp_file.name

        try:
            with (
                patch.object(
                    service,
                    "_load_model",
                    new=AsyncMock(
                        return_value={
                            "model": object(),
                            "model_id": 7,
                            "model_name": "registry-vit",
                            "model_type": "vit",
                            "input_size": 224,
                        }
                    ),
                ),
                patch.object(
                    service,
                    "_preprocess_image",
                    return_value=object(),
                ) as preprocess_image_mock,
                patch.object(
                    service,
                    "_predict_tensor",
                    new=AsyncMock(
                        return_value={"probabilities": {"fake": 0.8, "real": 0.2}}
                    ),
                ),
                patch.object(
                    service,
                    "_save_detection_result",
                    new=AsyncMock(return_value=persisted_result),
                ),
                patch.object(service, "_write_detection_audit_log"),
            ):
                response = await service.detect_file(
                    file_path=file_path,
                    request=DetectionRequest(model_type="vit", preprocess=False),
                    background_tasks=background_tasks,
                    original_file_name="preprocess-false.png",
                )
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

        preprocess_image_mock.assert_called_once()
        self.assertEqual(preprocess_image_mock.call_args.args[0], file_path)
        self.assertEqual(preprocess_image_mock.call_args.kwargs["input_size"], 224)
        self.assertIn("face_roi_policy", preprocess_image_mock.call_args.kwargs)
        self.assertTrue(response.success)
        self.assertEqual(response.record_id, 1)
        self.assertEqual(response.file_info["resolution"], "224x224")
        self.assertEqual(len(background_tasks.tasks), 1)


if __name__ == "__main__":
    unittest.main()
