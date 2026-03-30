import unittest
from datetime import datetime

from app.models.database_models import ModelRegistry
from app.services.model_service import ModelService


class ModelMetricsContractTests(unittest.TestCase):
    def test_db_to_response_keeps_metrics_when_accuracy_is_missing(self):
        model = ModelRegistry(
            id=9,
            name="registry-vit",
            model_type="vit",
            version="1.0",
            description=None,
            input_size=224,
            num_classes=2,
            parameters={},
            file_path="/tmp/model.pt",
            status="ready",
            accuracy=None,
            precision=0.81,
            recall=0.77,
            f1_score=0.79,
            is_default=False,
            deployment_info=None,
            created_at=datetime(2026, 3, 30, 12, 0, 0),
            updated_at=None,
            training_job_id=None,
        )

        response = ModelService(db=None)._db_to_response(model)

        self.assertIsNotNone(response.metrics)
        if response.metrics is None:
            self.fail("Model metrics should be present when precision/recall/F1 exist")
        self.assertIsNone(response.metrics.accuracy)
        self.assertEqual(response.metrics.precision, 0.81)
        self.assertEqual(response.metrics.recall, 0.77)
        self.assertEqual(response.metrics.f1_score, 0.79)


if __name__ == "__main__":
    unittest.main()
