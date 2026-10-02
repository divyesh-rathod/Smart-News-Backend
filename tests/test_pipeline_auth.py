import pytest
from fastapi.testclient import TestClient

from app.api.V1 import scripts_route
from app.config import settings
from app.main import app

ADMIN_KEY = "test-admin-key"


@pytest.fixture
def pipeline_runs(monkeypatch):
    """Configure an admin key and record pipeline starts instead of running the subprocesses."""
    runs = []
    monkeypatch.setattr(scripts_route, "start_full_pipeline_subprocesses", lambda: runs.append("run"))
    monkeypatch.setattr(settings, "ADMIN_API_KEY", ADMIN_KEY)
    return runs


def trigger(headers: dict[str, str]):
    return TestClient(app).post("/api/V1/scripts/run_pipeline", headers=headers)


def test_the_admin_token_starts_the_pipeline(pipeline_runs):
    response = trigger({"X-Admin-Token": ADMIN_KEY})

    assert response.status_code == 202
    assert pipeline_runs == ["run"]


@pytest.mark.parametrize(
    "headers",
    [{}, {"X-Admin-Token": "wrong"}, {"Authorization": f"Bearer {ADMIN_KEY}"}],
    ids=["no token", "wrong token", "token in the wrong header"],
)
def test_requests_without_the_admin_token_are_rejected(pipeline_runs, headers):
    response = trigger(headers)

    assert response.status_code == 401
    assert pipeline_runs == []


@pytest.mark.parametrize("configured_key", [None, ""])
def test_the_pipeline_cannot_be_triggered_when_no_admin_key_is_configured(pipeline_runs, monkeypatch, configured_key):
    monkeypatch.setattr(settings, "ADMIN_API_KEY", configured_key)

    response = trigger({"X-Admin-Token": ""})

    assert response.status_code == 503
    assert pipeline_runs == []
