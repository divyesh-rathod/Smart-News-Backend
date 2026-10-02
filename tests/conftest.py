import asyncio
import os

import pytest

# app.config requires DATABASE_URL at import time. Tests must never reach the dev database, so it is
# always overridden: tests marked "db" get TEST_DATABASE_URL, and the rest never connect.
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
os.environ["DATABASE_URL"] = TEST_DATABASE_URL or "postgresql+asyncpg://test:test@localhost:5432/test"

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402

from app.db.base import Base  # noqa: E402
import app.db.models  # noqa: E402, F401
from app.db.session import AsyncSessionLocal  # noqa: E402
from app.services import recommendation_cache  # noqa: E402


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    # Mark every test that uses the db fixture before -m filtering runs, so `-m "not db"` works.
    db_items = [item for item in items if "db" in item.fixturenames]
    for item in db_items:
        item.add_marker(pytest.mark.db)
    if TEST_DATABASE_URL or not db_items:
        return
    if os.environ.get("CI"):
        raise pytest.UsageError("Tests that use the db fixture need TEST_DATABASE_URL, and CI must not skip them")
    for item in db_items:
        item.add_marker(pytest.mark.skip(reason="TEST_DATABASE_URL is not set"))


@pytest.fixture(autouse=True)
def empty_recommendation_cache():
    recommendation_cache.clear()


@pytest.fixture
def db():
    """Empty every table before the test. The schema comes from `alembic upgrade head`."""
    # Each test drives the app with asyncio.run, i.e. a new event loop. asyncpg connections can't move
    # between loops, so the app's sessions get an engine that opens a fresh connection every time.
    AsyncSessionLocal.configure(bind=create_async_engine(TEST_DATABASE_URL, poolclass=NullPool))
    tables = ", ".join(f'"{table.name}"' for table in Base.metadata.sorted_tables)

    async def truncate():
        async with AsyncSessionLocal() as session:
            await session.execute(text(f"TRUNCATE {tables} CASCADE"))
            await session.commit()

    asyncio.run(truncate())


@pytest.fixture
def user(db):
    """A saved user that the API treats as logged in."""
    from app.main import app
    from app.utils.auth import get_current_user
    from tests.factories import add_user

    user = asyncio.run(add_user())
    app.dependency_overrides[get_current_user] = lambda: user
    yield user
    app.dependency_overrides.clear()


@pytest.fixture
def client(user):
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


@pytest.fixture
def fake_rerank(monkeypatch):
    """Stand-in for the cross-encoder: records its inputs and keeps the stage-1 order."""
    from app.ml_models import retrieve

    calls = []

    async def rerank_top_k(query, candidates, top_n=5):
        calls.append({"query": query, "candidates": candidates})
        return [{**c, "score": 1.0} for c in candidates[:top_n]]

    monkeypatch.setattr(retrieve, "rerank_top_k", rerank_top_k)
    return calls
