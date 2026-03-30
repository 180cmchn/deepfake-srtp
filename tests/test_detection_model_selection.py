import os
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime
from unittest.mock import AsyncMock, patch

from app.models.database_models import ModelStatus
from app.schemas.detection import DetectionRequest
from app.services.detection_service import DetectionService


@dataclass
class FakeRegistryModel:
    id: int
    name: str
    status: str
    model_type: str = "vit"
    is_default: bool = False


class FakeQuery:
    def __init__(self, requested_model=None, ready_registry_models=None):
        self._requested_model = requested_model
        self._ready_registry_models = ready_registry_models or []

    def filter(self, *_args, **_kwargs):
        return self

    def order_by(self, *_args, **_kwargs):
        return self

    def first(self):
        return self._requested_model

    def all(self):
        return list(self._ready_registry_models)


class FakeDB:
    def __init__(self, requested_model=None, ready_registry_models=None):
        self._requested_model = requested_model
        self._ready_registry_models = ready_registry_models or []
        self._added = []

    def query(self, _model):
        return FakeQuery(
            requested_model=self._requested_model,
            ready_registry_models=self._ready_registry_models,
        )

    def add(self, obj):
        self._added.append(obj)

    def commit(self):
        return None

    def rollback(self):
        return None

    def refresh(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = len(self._added)
        if getattr(obj, "created_at", None) is None:
            obj.created_at = datetime.now()


class DummyBackgroundTasks:
    def __init__(self):
        self.tasks = []

    def add_task(self, func, *args, **kwargs):
        self.tasks.append((func, args, kwargs))


def build_loaded_registry_model(model_record: FakeRegistryModel):
    return {
        "model": object(),
        "model_id": model_record.id,
        "model_name": model_record.name,
        "model_type": model_record.model_type,
        "input_size": 224,
        "source": "registry",
        "status": model_record.status,
        "weight_state": "checkpoint_loaded",
    }


class DetectionServiceModelSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_load_model_rejects_missing_requested_model_without_registry_fallback(
        self,
    ):
        service = DetectionService(
            FakeDB(requested_model=None, ready_registry_models=[])
        )

        with patch("app.services.detection_service.create_model") as create_model_mock:
            with patch.object(service, "_load_registry_model_record") as load_mock:
                with self.assertRaises(ValueError) as exc_info:
                    await service._load_model(model_id=404, model_type="vit")

        message = str(exc_info.exception)
        self.assertIn("Requested registry model id=404 was not found", message)
        self.assertIn(
            "No ready/deployed registry fallback candidate could be selected.",
            message,
        )
        self.assertIn("Built-in fallback inference is disabled", message)
        create_model_mock.assert_not_called()
        load_mock.assert_not_called()

    async def test_load_model_rejects_unready_requested_model_without_registry_fallback(
        self,
    ):
        requested_model = FakeRegistryModel(
            id=12,
            name="training-vit",
            status=ModelStatus.TRAINING.value,
            model_type="vit",
        )
        service = DetectionService(
            FakeDB(requested_model=requested_model, ready_registry_models=[])
        )

        with patch("app.services.detection_service.create_model") as create_model_mock:
            with patch.object(service, "_load_registry_model_record") as load_mock:
                with self.assertRaises(ValueError) as exc_info:
                    await service._load_model(
                        model_id=requested_model.id,
                        model_type=requested_model.model_type,
                    )

        message = str(exc_info.exception)
        self.assertIn(
            "Requested registry model id=12 (training-vit) is training, not ready/deployed",
            message,
        )
        self.assertIn("Built-in fallback inference is disabled", message)
        create_model_mock.assert_not_called()
        load_mock.assert_not_called()

    async def test_detect_file_returns_failure_without_prediction_when_only_builtin_path_exists(
        self,
    ):
        service = DetectionService(FakeDB())
        no_model_error = (
            "No usable ready/deployed registry model is available for detection "
            "(model_type='vit'). Requested model decision: Requested built-in "
            "fallback model_type='vit' cannot run inference without a "
            "ready/deployed registry model. No ready/deployed registry fallback "
            "candidate could be selected. Built-in fallback inference is disabled "
            "until a vetted registry model is ready or deployed."
        )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(b"not-a-real-image")
            file_path = temp_file.name

        try:
            with patch.object(
                service,
                "_load_model",
                new=AsyncMock(side_effect=ValueError(no_model_error)),
            ):
                response = await service.detect_file(
                    file_path=file_path,
                    request=DetectionRequest(model_type="vit"),
                    background_tasks=DummyBackgroundTasks(),
                )
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

        self.assertFalse(response.success)
        self.assertEqual(response.record_id, 1)
        self.assertIsNone(response.result)
        self.assertIsNotNone(response.error_message)
        self.assertEqual(response.error_code, "model_unavailable")
        self.assertIn(
            "Built-in fallback inference is disabled", response.error_message or ""
        )

    def test_get_history_no_model_fallback_text_does_not_claim_builtin(self):
        service = DetectionService(FakeDB())
        self.assertEqual(
            service._resolve_error_code(
                "No usable ready/deployed registry model is available for detection (no explicit model request)."
            ),
            "model_unavailable",
        )

    async def test_load_model_falls_back_to_ready_registry_model_with_explicit_provenance(
        self,
    ):
        requested_model = FakeRegistryModel(
            id=21,
            name="staging-vit",
            status=ModelStatus.TRAINING.value,
            model_type="vit",
        )
        fallback_model = FakeRegistryModel(
            id=77,
            name="deployed-resnet",
            status=ModelStatus.DEPLOYED.value,
            model_type="resnet",
            is_default=True,
        )
        service = DetectionService(
            FakeDB(
                requested_model=requested_model,
                ready_registry_models=[fallback_model],
            )
        )

        with patch.object(
            service,
            "_load_registry_model_record",
            return_value=build_loaded_registry_model(fallback_model),
        ) as load_mock:
            loaded_model = await service._load_model(
                model_id=requested_model.id,
                model_type=requested_model.model_type,
            )

        self.assertEqual(loaded_model["model_id"], fallback_model.id)
        self.assertEqual(loaded_model["model_name"], fallback_model.name)
        self.assertEqual(loaded_model["model_type"], fallback_model.model_type)
        self.assertEqual(loaded_model["source"], "registry")
        self.assertEqual(loaded_model["status"], fallback_model.status)
        self.assertEqual(loaded_model["weight_state"], "checkpoint_loaded")
        self.assertEqual(loaded_model["requested_model_id"], requested_model.id)
        self.assertEqual(
            loaded_model["requested_model_type"], requested_model.model_type
        )
        self.assertEqual(loaded_model["requested_model_status"], requested_model.status)
        self.assertEqual(loaded_model["readiness"], "ready")
        self.assertEqual(loaded_model["selection_policy"], "fallback_default")
        self.assertIn("not ready/deployed", loaded_model["fallback_reason"])
        load_mock.assert_called_once_with(fallback_model)

    async def test_partial_checkpoint_load_is_rejected(self):
        service = DetectionService(FakeDB())

        class PartialModel:
            def load_state_dict(self, _state_dict):
                raise RuntimeError("Missing key(s) in state_dict: 'head.weight'.")

        with self.assertRaises(ValueError) as exc_info:
            service._load_model_state(PartialModel(), {"weights": 1})

        self.assertIn("partial weight loading is rejected", str(exc_info.exception))


if __name__ == "__main__":
    unittest.main()
