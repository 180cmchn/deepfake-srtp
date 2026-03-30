import inspect
import unittest

from app.api.routes import detection as detection_routes
from app.schemas.detection import (
    BatchDetectionRequest,
    DetectionRequest,
    VideoDetectionRequest,
)
from app.schemas.models import ModelMetrics
from app.schemas.training import TrainingResults


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


if __name__ == "__main__":
    unittest.main()
