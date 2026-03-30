import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from app.api.routes import health as health_routes
from app.core.config import settings
from app import main as app_main


class FakeScalarResult:
    def __init__(self, value):
        self._value = value

    def scalar(self):
        return self._value


class RecordingDB:
    def __init__(self, counts=None, errors=None):
        self.counts = counts or {}
        self.errors = errors or {}
        self.queries = []

    def execute(self, statement):
        sql = str(statement)
        self.queries.append(sql)
        for table, error in self.errors.items():
            if table in sql:
                raise error
        for table, count in self.counts.items():
            if table in sql:
                return FakeScalarResult(count)
        raise AssertionError(f"Unexpected query: {sql}")


class HealthTruthfulnessTests(unittest.IsolatedAsyncioTestCase):
    def build_memory(self, *, total=8, available=4, used=4, percent=50):
        gib = 1024**3
        return SimpleNamespace(
            total=total * gib,
            available=available * gib,
            used=used * gib,
            percent=percent,
        )

    def build_disk(self, *, total=100, free=60, used=40):
        gib = 1024**3
        return SimpleNamespace(total=total * gib, free=free * gib, used=used * gib)

    async def test_top_level_health_returns_503_when_database_is_unhealthy(self):
        with patch(
            "app.main.get_database_health_snapshot",
            return_value={
                "healthy": False,
                "status": "unhealthy",
                "response_time_ms": 12.5,
                "error": "database unavailable",
            },
        ):
            response = await app_main.health_check()

        payload = json.loads(response.body)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(payload["status"], "unhealthy")
        self.assertEqual(payload["database"], "unhealthy")
        self.assertEqual(payload["checks"]["database"]["error"], "database unavailable")
        self.assertEqual(payload["version"], settings.APP_VERSION)

    async def test_api_health_returns_503_and_structured_database_failure(self):
        with (
            patch(
                "app.api.routes.health.get_database_health_snapshot",
                return_value={
                    "healthy": False,
                    "status": "unhealthy",
                    "response_time_ms": 7.5,
                    "error": "db down",
                },
            ),
            patch(
                "app.api.routes.health.psutil.disk_usage",
                return_value=self.build_disk(),
            ),
            patch(
                "app.api.routes.health.psutil.virtual_memory",
                return_value=self.build_memory(),
            ),
        ):
            response = await health_routes.health_check()

        payload = json.loads(response.body)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(payload["status"], "unhealthy")
        self.assertEqual(payload["checks"]["database"]["status"], "unhealthy")
        self.assertEqual(payload["checks"]["database"]["error"], "db down")
        self.assertEqual(payload["version"], settings.APP_VERSION)

    async def test_system_status_queries_real_table_names(self):
        db = RecordingDB(
            counts={
                "detection_results": 10,
                "training_jobs": 3,
                "model_registry": 2,
                "dataset_info": 5,
            }
        )
        with (
            patch(
                "app.api.routes.health.get_database_health_snapshot",
                return_value={
                    "healthy": True,
                    "status": "healthy",
                    "response_time_ms": 2.0,
                },
            ),
            patch("app.api.routes.health.psutil.cpu_percent", return_value=12.5),
            patch(
                "app.api.routes.health.psutil.virtual_memory",
                return_value=self.build_memory(),
            ),
            patch(
                "app.api.routes.health.psutil.disk_usage",
                return_value=self.build_disk(),
            ),
        ):
            response = await health_routes.get_system_status(
                db=db,
                current_user="health-user",
            )

        payload = json.loads(response.body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "healthy")
        self.assertEqual(
            payload["database"]["tables"]["detection_results"]["row_count"], 10
        )
        self.assertEqual(
            payload["database"]["tables"]["model_registry"]["row_count"], 2
        )
        executed_sql = "\n".join(db.queries)
        self.assertIn("SELECT COUNT(*) FROM detection_results", executed_sql)
        self.assertIn("SELECT COUNT(*) FROM training_jobs", executed_sql)
        self.assertIn("SELECT COUNT(*) FROM model_registry", executed_sql)
        self.assertIn("SELECT COUNT(*) FROM dataset_info", executed_sql)
        self.assertNotIn("detection_records", executed_sql)
        self.assertNotIn("models", executed_sql)
        self.assertNotIn("datasets", executed_sql)

    async def test_system_status_returns_503_when_database_dependency_is_down(self):
        with (
            patch(
                "app.api.routes.health.get_database_health_snapshot",
                return_value={
                    "healthy": False,
                    "status": "unhealthy",
                    "response_time_ms": 33.0,
                    "error": "cannot connect",
                },
            ),
            patch("app.api.routes.health.psutil.cpu_percent", return_value=1.0),
            patch(
                "app.api.routes.health.psutil.virtual_memory",
                return_value=self.build_memory(),
            ),
            patch(
                "app.api.routes.health.psutil.disk_usage",
                return_value=self.build_disk(),
            ),
        ):
            response = await health_routes.get_system_status(
                db=RecordingDB(),
                current_user="health-user",
            )

        payload = json.loads(response.body)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(payload["status"], "unhealthy")
        self.assertEqual(payload["database"]["error"], "cannot connect")

    async def test_metrics_failure_raises_structured_http_exception(self):
        with patch(
            "app.api.routes.health.psutil.cpu_percent",
            side_effect=RuntimeError("metrics boom"),
        ):
            with self.assertRaises(HTTPException) as context:
                await health_routes.get_system_metrics(
                    db=RecordingDB(),
                    current_user="metrics-user",
                )

        self.assertEqual(context.exception.status_code, 500)
        self.assertEqual(
            context.exception.detail["message"],
            "Failed to get system metrics",
        )
        self.assertEqual(context.exception.detail["error"], "metrics boom")


if __name__ == "__main__":
    unittest.main()
