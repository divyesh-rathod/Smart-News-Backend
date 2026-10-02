import asyncio
from datetime import datetime, timedelta, timezone

from tests.factories import add_article

FEED = "/api/V1/news/unseen-articles"


def add_articles(*hours_ago: int) -> list[str]:
    """One article per entry, published that many hours before a fixed time. Returns ids, as strings."""
    base = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)

    async def add():
        return [str(await add_article(f"article {i}", pub_date=base - timedelta(hours=h))) for i, h in enumerate(hours_ago)]

    return asyncio.run(add())


def page(client, limit: int, cursor: str | None = None) -> tuple[list[str], str | None]:
    params = {"limit": limit} | ({"cursor": cursor} if cursor else {})
    response = client.get(FEED, params=params)
    assert response.status_code == 200, response.text
    body = response.json()
    return [item["article_id"] for item in body["results"]], body["next_cursor"]


def test_following_next_cursor_visits_every_article_once_newest_first(client, user):
    # Two pairs share a publication time, one pair right at a page boundary.
    ids = add_articles(0, 1, 2, 2, 3, 4, 4)

    seen, cursor, pages = [], None, 0
    while True:
        page_ids, cursor = page(client, limit=3, cursor=cursor)
        seen += page_ids
        pages += 1
        if cursor is None:
            break

    assert sorted(seen) == sorted(ids)
    assert len(seen) == len(set(seen))
    assert seen[:2] == ids[:2] and seen[-3] == ids[4]
    assert pages == 3


def test_a_reload_starts_again_from_the_newest_unread_article(client, user):
    ids = add_articles(0, 1, 2, 3, 4)
    first_page, _ = page(client, limit=3)
    client.post(f"/api/V1/news/mark-as-read/{first_page[0]}")

    after_reload, _ = page(client, limit=3)

    assert first_page == ids[:3]
    assert after_reload == ids[1:4]


def test_the_last_page_has_no_next_cursor(client, user):
    add_articles(0, 1)

    ids, cursor = page(client, limit=5)

    assert len(ids) == 2
    assert cursor is None


def test_a_cursor_that_was_not_issued_by_the_api_is_rejected(client, user):
    response = client.get(FEED, params={"cursor": "not-a-cursor"})

    assert response.status_code == 400
