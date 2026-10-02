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
