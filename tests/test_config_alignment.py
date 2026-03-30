import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.api.routes import datasets as dataset_routes
from app.api.routes import detection as detection_routes
from app.core.config import Settings
from app.schemas.models import ModelStatus
from app.services import model_service as model_service_module
from app.services.model_service import ModelService


class QueryStub:
    def __init__(self, result):
        self.result = result

    def filter(self, *_args, **_kwargs):
        return self

    def first(self):
        return self.result

    def update(self, *_args, **_kwargs):
        return 1


class FakeDB:
    def __init__(self, result):
        self.result = result
        self.commit_count = 0
        self.rollback_count = 0

    def query(self, _model):
        return QueryStub(self.result)

    def commit(self):
        self.commit_count += 1

    def rollback(self):
        self.rollback_count += 1


class ConfigAlignmentTests(unittest.TestCase):
    def test_settings_accept_legacy_api_v1_prefix_env_key(self):
        with patch.dict(os.environ, {"API_V1_PREFIX": "/legacy-api"}, clear=True):
            settings = Settings(_env_file=None)

        self.assertEqual(settings.API_V1_STR, "/legacy-api")

    def test_settings_prefer_api_v1_str_over_legacy_prefix(self):
        with patch.dict(
            os.environ,
            {"API_V1_STR": "/canonical-api", "API_V1_PREFIX": "/legacy-api"},
            clear=True,
        ):
            settings = Settings(_env_file=None)

        self.assertEqual(settings.API_V1_STR, "/canonical-api")

    def test_detection_upload_path_uses_configured_upload_dir(self):
        with (
            patch.object(detection_routes.settings, "UPLOAD_DIR", "custom_uploads"),
            patch("app.api.routes.detection.os.makedirs") as makedirs,
            patch("app.api.routes.detection.uuid.uuid4", return_value="upload-id"),
        ):
            upload_path = detection_routes._build_upload_path("sample.mp4")

        self.assertEqual(upload_path, os.path.join("custom_uploads", "upload-id.mp4"))
        makedirs.assert_called_once_with("custom_uploads", exist_ok=True)

    def test_dataset_upload_path_uses_configured_data_dir(self):
        with (
            patch.object(dataset_routes.settings, "DATA_DIR", "custom_data"),
            patch("app.api.routes.datasets.os.makedirs") as makedirs,
            patch("app.api.routes.datasets.uuid.uuid4", return_value="dataset-id"),
        ):
            file_path = dataset_routes._build_dataset_upload_path(
                file_type="image",
                file_extension=".png",
                dataset_id=12,
            )

        self.assertEqual(
            file_path,
            os.path.join("custom_data", "dataset_12_img_dataset-id.png"),
        )
        makedirs.assert_called_once_with("custom_data", exist_ok=True)

    def test_model_deploy_urls_follow_configured_api_prefix(self):
        db_model = SimpleNamespace(
            status=ModelStatus.READY.value,
            deployment_info=None,
            is_default=False,
        )
        db = FakeDB(db_model)
        service = ModelService(db)

        with patch.object(model_service_module.settings, "API_V1_STR", "/custom-api"):
            deployment = asyncio.run(service.deploy_model(7, {"is_default": False}))

        self.assertEqual(deployment.endpoint_url, "/custom-api/models/7/predict")
        self.assertEqual(deployment.health_check_url, "/custom-api/models/7/health")
        self.assertEqual(db.commit_count, 1)


if __name__ == "__main__":
    unittest.main()
