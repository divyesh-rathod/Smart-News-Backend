import logging

from fastapi.testclient import TestClient

from app.main import app
from app.ml_models import rerank


def test_startup_loads_the_cross_encoder_before_serving(monkeypatch):
    loads = []
    monkeypatch.setattr(rerank, "get_model", lambda: loads.append("loaded"))

    with TestClient(app):
        assert loads == ["loaded"]


def test_startup_lets_app_info_logs_through(monkeypatch):
    monkeypatch.setattr(rerank, "get_model", lambda: None)
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    level = root.level
    root.setLevel(logging.WARNING)  # Python's default, which uvicorn leaves in place
    try:
        with TestClient(app):
            assert logging.getLogger("app.services.news_services").isEnabledFor(logging.INFO)
            assert root.handlers
    finally:
        root.setLevel(level)
