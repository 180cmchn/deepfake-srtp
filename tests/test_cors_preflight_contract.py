import unittest

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from app.core.config import settings


class CorsPreflightContractTests(unittest.TestCase):
    def build_client(self):
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.BACKEND_CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.post("/probe")
        def probe():
            return {"ok": True}

        return TestClient(app)

    def test_preflight_allows_configured_origin(self):
        client = self.build_client()
        allowed_origin = settings.BACKEND_CORS_ORIGINS[0]

        response = client.options(
            "/probe",
            headers={
                "Origin": allowed_origin,
                "Access-Control-Request-Method": "POST",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            allowed_origin,
        )
        self.assertIn(
            "POST",
            response.headers.get("access-control-allow-methods", ""),
        )

    def test_preflight_rejects_disallowed_origin(self):
        client = self.build_client()
        blocked_origin = "https://evil.example.com"

        response = client.options(
            "/probe",
            headers={
                "Origin": blocked_origin,
                "Access-Control-Request-Method": "POST",
            },
        )

        self.assertNotEqual(
            response.headers.get("access-control-allow-origin"),
            blocked_origin,
        )
        self.assertIn(response.status_code, {400, 405})


if __name__ == "__main__":
    unittest.main()
