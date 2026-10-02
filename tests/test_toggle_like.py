import asyncio
import logging
import uuid

import pytest
from fastapi.testclient import TestClient

from app.db.models import Like
from app.db.session import AsyncSessionLocal
from app.main import app
from app.ml_models import retrieve
from app.utils.auth import get_current_user
from tests.factories import add_article, add_user, vector


@pytest.fixture
def user(db):
    user = asyncio.run(add_user())
    app.dependency_overrides[get_current_user] = lambda: user
    yield user
    app.dependency_overrides.clear()


@pytest.fixture
def client(user):
    return TestClient(app)


def toggle_like(client, article_id):
    return client.post(f"/api/V1/news/toggle-like/{article_id}")


def stored_like(user, article_id) -> int | None:
    async def get():
        async with AsyncSessionLocal() as session:
            like = await session.get(Like, (user.id, article_id))
            return like.is_liked if like else None

    return asyncio.run(get())


def test_like_is_saved_and_returns_recommendations(client, user, fake_rerank):
    source = asyncio.run(add_article("source", vector(1, 0)))
    neighbour = asyncio.run(add_article("neighbour", vector(1, 1)))

    response = toggle_like(client, source)

    assert response.status_code == 200
    body = response.json()
    assert body["liked"] is True
    assert [a["article_id"] for a in body["top5"]] == [str(neighbour)]
    assert [a["article_id"] for a in body["similar"]] == [str(neighbour)]
    assert stored_like(user, source) == 1


def test_liking_an_article_without_an_embedding_saves_the_like_with_no_recommendations(client, user, fake_rerank):
    source = asyncio.run(add_article("source", embedding=None))
    asyncio.run(add_article("neighbour", vector(1, 1)))

    response = toggle_like(client, source)

    assert response.status_code == 200
    assert response.json() == {"message": "Article liked", "liked": True, "top5": [], "similar": []}
    assert stored_like(user, source) == 1


def test_a_ranking_failure_keeps_the_like_and_is_logged(client, user, monkeypatch, caplog):
    async def broken_rerank(query, candidates, top_n=5):
        raise RuntimeError("model crashed")

    monkeypatch.setattr(retrieve, "rerank_top_k", broken_rerank)
    source = asyncio.run(add_article("source", vector(1, 0)))
    asyncio.run(add_article("neighbour", vector(1, 1)))

    response = toggle_like(client, source)

    assert response.status_code == 200
    assert response.json() == {"message": "Article liked", "liked": True, "top5": [], "similar": []}
    assert stored_like(user, source) == 1
    [record] = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert str(source) in record.getMessage()
    assert record.exc_info[1].args == ("model crashed",)


def test_toggling_a_liked_article_removes_the_like(client, user, fake_rerank):
    source = asyncio.run(add_article("source", vector(1, 0)))
    toggle_like(client, source)

    response = toggle_like(client, source)

    assert response.status_code == 200
    assert response.json() == {"message": "Like removed", "liked": False, "top5": [], "similar": []}
    assert stored_like(user, source) == 0


def test_liking_an_unknown_article_is_rejected_without_saving_anything(client, user, fake_rerank):
    missing = uuid.uuid4()

    response = toggle_like(client, missing)

    assert response.status_code == 400
    assert response.json() == {"detail": "Article not found"}
    assert stored_like(user, missing) is None
