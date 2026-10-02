import asyncio
from datetime import datetime, timedelta, timezone

from app.db.models import UserFeedPosition
from app.db.session import AsyncSessionLocal


def stored_last_read_date(user) -> datetime | None:
    async def get():
        async with AsyncSessionLocal() as session:
            row = await session.get(UserFeedPosition, user.id)
            return row.last_read_date if row else None

    return asyncio.run(get())


def test_set_date_stores_the_given_date(client, user):
    response = client.post("/api/V1/news/set-date", params={"last_read_date": "2026-09-01T12:00:00Z"})

    assert response.status_code == 200
    assert stored_last_read_date(user) == datetime(2026, 9, 1, 12, tzinfo=timezone.utc)


def test_set_date_overwrites_an_earlier_date(client, user):
    client.post("/api/V1/news/set-date", params={"last_read_date": "2026-09-01T12:00:00Z"})

    response = client.post("/api/V1/news/set-date", params={"last_read_date": "2026-09-02T08:30:00Z"})

    assert response.status_code == 200
    assert stored_last_read_date(user) == datetime(2026, 9, 2, 8, 30, tzinfo=timezone.utc)


def test_set_date_defaults_to_now(client, user):
    response = client.post("/api/V1/news/set-date")

    assert response.status_code == 200
    assert abs(stored_last_read_date(user) - datetime.now(timezone.utc)) < timedelta(minutes=1)
