import asyncio
import logging
import uuid

from app.db.models import Like
from app.db.session import AsyncSessionLocal
from app.ml_models import retrieve
from app.services import news_services
from tests.factories import add_article, vector


def set_like(client, article_id, liked: bool = True):
    return client.put(f"/api/V1/news/like/{article_id}", json={"liked": liked})


def stored_like(user, article_id) -> int | None:
    async def get():
        async with AsyncSessionLocal() as session:
            like = await session.get(Like, (user.id, article_id))
            return like.is_liked if like else None

    return asyncio.run(get())


def test_like_is_saved_and_returns_recommendations(client, user, fake_rerank):
    source = asyncio.run(add_article("source", vector(1, 0)))
    neighbour = asyncio.run(add_article("neighbour", vector(1, 1)))

    response = set_like(client, source)

    assert response.status_code == 200
    body = response.json()
    assert body["liked"] is True
    assert [a["article_id"] for a in body["top5"]] == [str(neighbour)]
    assert [a["article_id"] for a in body["similar"]] == [str(neighbour)]
    assert stored_like(user, source) == 1


def test_repeating_a_like_keeps_it_liked(client, user, fake_rerank):
    source = asyncio.run(add_article("source", vector(1, 0)))
    asyncio.run(add_article("neighbour", vector(1, 1)))

    first, second = set_like(client, source), set_like(client, source)

    assert first.json() == second.json()
    assert second.json()["liked"] is True
    assert stored_like(user, source) == 1


def test_unliking_removes_the_like_and_repeating_it_changes_nothing(client, user, fake_rerank):
    source = asyncio.run(add_article("source", vector(1, 0)))
    set_like(client, source)

    responses = [set_like(client, source, liked=False) for _ in range(2)]

    for response in responses:
        assert response.status_code == 200
        assert response.json() == {"message": "Like removed", "liked": False, "top5": [], "similar": []}
    assert stored_like(user, source) == 0


def test_concurrent_likes_of_the_same_article_both_succeed(db, user, fake_rerank):
    source = asyncio.run(add_article("source", vector(1, 0)))

    async def like_twice_at_once():
        return await asyncio.gather(*(news_services.set_article_like(source, user, True) for _ in range(2)))

    results = asyncio.run(like_twice_at_once())

    assert [liked for _, liked, _, _ in results] == [True, True]
    assert stored_like(user, source) == 1


def test_liking_an_article_without_an_embedding_saves_the_like_with_no_recommendations(client, user, fake_rerank):
    source = asyncio.run(add_article("source", embedding=None))
    asyncio.run(add_article("neighbour", vector(1, 1)))

    response = set_like(client, source)

    assert response.status_code == 200
    assert response.json() == {"message": "Article liked", "liked": True, "top5": [], "similar": []}
    assert stored_like(user, source) == 1


def test_a_ranking_failure_keeps_the_like_and_is_logged(client, user, monkeypatch, caplog):
    async def broken_rerank(query, candidates, top_n=5):
        raise RuntimeError("model crashed")

    monkeypatch.setattr(retrieve, "rerank_top_k", broken_rerank)
    source = asyncio.run(add_article("source", vector(1, 0)))
    asyncio.run(add_article("neighbour", vector(1, 1)))

    response = set_like(client, source)

    assert response.status_code == 200
    assert response.json() == {"message": "Article liked", "liked": True, "top5": [], "similar": []}
    assert stored_like(user, source) == 1
    [record] = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert str(source) in record.getMessage()
    assert record.exc_info[1].args == ("model crashed",)


def test_liking_an_unknown_article_is_rejected_without_saving_anything(client, user, fake_rerank):
    missing = uuid.uuid4()

    response = set_like(client, missing)

    assert response.status_code == 400
    assert response.json() == {"detail": "Article not found"}
    assert stored_like(user, missing) is None


def test_a_like_request_without_the_desired_state_is_rejected(client, user):
    response = client.put(f"/api/V1/news/like/{uuid.uuid4()}", json={})

    assert response.status_code == 422


def test_the_feed_says_which_articles_the_user_has_liked(client, user, fake_rerank):
    liked = asyncio.run(add_article("liked", vector(1, 0)))
    unliked = asyncio.run(add_article("unliked later", vector(1, 1)))
    untouched = asyncio.run(add_article("untouched", vector(0, 1)))
    set_like(client, liked)
    set_like(client, unliked)
    set_like(client, unliked, liked=False)

    feed = client.get("/api/V1/news/unseen-articles", params={"limit": 10}).json()["results"]

    assert {item["article_id"]: item["liked"] for item in feed} == {
        str(liked): True,
        str(unliked): False,
        str(untouched): False,
    }
