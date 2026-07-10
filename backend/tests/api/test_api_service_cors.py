import os
from unittest.mock import patch
from fastapi.testclient import TestClient
from starlette.requests import Request

# Import the FastAPI app and CORS functions
from retrieval import api_service
from retrieval.api_service import app


def test_cors_origins_local_development_defaults() -> None:
    # Test that when tenant is 'local', all requested local origins are included
    with patch.dict(os.environ, {"CODESEEK_TENANT_ID": "local", "CODESEEK_CORS_ORIGINS": "http://localhost:5173"}):
        origins = api_service._cors_origins()
        assert "http://localhost:5173" in origins
        assert "http://127.0.0.1:5173" in origins
        assert "http://0.0.0.0:5173" in origins
        assert "http://localhost:5174" in origins
        assert "http://127.0.0.1:5174" in origins


def test_cors_origin_regex_tenant_isolation() -> None:
    # Test that regex is only set when tenant is 'local'
    with patch.dict(os.environ, {"CODESEEK_TENANT_ID": "local"}):
        regex = api_service._cors_origin_regex()
        assert regex is not None
        assert regex == r"^http://(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?$"

    with patch.dict(os.environ, {"CODESEEK_TENANT_ID": "prod"}):
        regex = api_service._cors_origin_regex()
        assert regex is None


def test_enforce_https_middleware_bypasses_options() -> None:
    # Test that preflight OPTIONS requests bypass enforce_https_middleware even if ENFORCE_HTTPS is True
    client = TestClient(app)

    with patch.object(api_service, "ENFORCE_HTTPS", True), \
         patch.object(api_service, "TRUST_X_FORWARDED_PROTO", False), \
         patch.dict(os.environ, {"CODESEEK_CORS_ORIGINS": "http://localhost:5173"}):
        
        # Make an OPTIONS request from an allowed origin
        headers = {
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "x-app-encryption-key",
        }
        
        # Test an endpoint that usually enforces HTTPS (like /api/v1/sessions)
        response = client.options("/api/v1/sessions", headers=headers)
        
        # CORS preflight should succeed with 200 OK and correct headers
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
        assert response.headers.get("access-control-allow-credentials") == "true"


def test_cors_preflight_succeeds_for_local_origins() -> None:
    client = TestClient(app)

    # When requesting from local origins, it should match the allowed CORS origins and return 200
    with patch.object(api_service, "ENFORCE_HTTPS", False), \
         patch.dict(os.environ, {"CODESEEK_CORS_ORIGINS": "http://localhost:5173", "CODESEEK_TENANT_ID": "local"}):
        
        local_origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://0.0.0.0:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
            "http://localhost:3000",  # matched by regex
            "http://127.0.0.1:8080",  # matched by regex
            "http://0.0.0.0:9000",    # matched by regex
        ]
        
        for origin in local_origins:
            for path in ["/api/v1/sessions", "/api/v1/health"]:
                headers = {
                    "Origin": origin,
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "content-type,x-codeseek-api-key,authorization,x-requested-with",
                }
                response = client.options(path, headers=headers)
                assert response.status_code == 200
                assert response.headers.get("access-control-allow-origin") == origin
                assert response.headers.get("access-control-allow-credentials") == "true"
                # Access-Control-Allow-Headers should echo back the requested headers or allow them
                allowed_headers = response.headers.get("access-control-allow-headers", "").lower()
                assert "content-type" in allowed_headers
                assert "x-codeseek-api-key" in allowed_headers
                assert "authorization" in allowed_headers
                assert "x-requested-with" in allowed_headers
