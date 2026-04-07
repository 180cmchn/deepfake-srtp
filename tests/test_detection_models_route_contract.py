import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

from app.api.routes import detection as detection_routes
from app.core.config import settings


class FakeModelsQuery:
    def __init__(self, rows):
        self._rows = list(rows)

    def filter(self, *_args, **_kwargs):
        return self

    def order_by(self, *_args, **_kwargs):
        return self

    def all(self):
        return list(self._rows)


class FakeModelsDB:
    def __init__(self, rows):
        self._rows = list(rows)

    def query(self, _model):
        return FakeModelsQuery(self._rows)


class DetectionModelsRouteContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_models_route_requires_explicit_selection_when_only_fallback_models_exist(
        self,
    ):
        with patch(
            "app.models.ml_models.ModelRegistry.list_models",
            return_value=["vit", "resnet", "yolo"],
        ):
            response = await detection_routes.get_available_models(db=FakeModelsDB([]))

        self.assertEqual(response["default"]["model_id"], None)
        self.assertEqual(response["default"]["model_type"], None)
        self.assertEqual(response["default"]["source"], None)
        self.assertEqual(
            response["default"]["selection_policy"],
            "explicit_ready_model_required",
        )
        expected_fallback_type = (
            settings.DEFAULT_MODEL_TYPE
            if settings.DEFAULT_MODEL_TYPE in {"vit", "resnet", "yolo"}
            else "resnet"
        )
        self.assertEqual(
            response["default"]["fallback_model_type"],
            expected_fallback_type,
        )
        self.assertIn("yolo", response["model_types"])
        self.assertIn(
            "yolo",
            {model["model_type"] for model in response["models"]},
        )
        self.assertTrue(
            all(model["source"] == "builtin" for model in response["models"])
        )
        self.assertTrue(all(model["is_ready"] is False for model in response["models"]))

    async def test_models_route_prefers_ready_registry_default(self):
        ready_model = SimpleNamespace(
            id=7,
            name="registry-vit",
            model_type="vit",
            status="ready",
            is_default=True,
            created_at=datetime(2026, 3, 30, 12, 0, 0),
        )

        with patch(
            "app.models.ml_models.ModelRegistry.list_models",
            return_value=["vit", "resnet", "yolo"],
        ):
            response = await detection_routes.get_available_models(
                db=FakeModelsDB([ready_model])
            )

        self.assertEqual(response["default"]["model_id"], 7)
        self.assertEqual(response["default"]["model_type"], "vit")
        self.assertEqual(response["default"]["source"], "registry")
        self.assertTrue(response["default"]["is_ready"])
        self.assertEqual(response["default"]["selection_policy"], "primary")
        self.assertIn("yolo", response["model_types"])


if __name__ == "__main__":
    unittest.main()
