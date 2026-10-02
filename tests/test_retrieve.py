import asyncio
import math
import os
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.db.session import AsyncSessionLocal
from app.ml_models import retrieve
from tests.factories import add_article, vector


@pytest.fixture
def index_scans_only(db):
    """Make the planner use the HNSW index on a test-sized table, as it would on a large one."""
    AsyncSessionLocal.configure(bind=create_async_engine(
        os.environ["TEST_DATABASE_URL"],
        poolclass=NullPool,
        connect_args={"server_settings": {"enable_seqscan": "off", "enable_sort": "off"}},
    ))


async def stage1_plan(source_id: uuid.UUID, embedding: list[float]) -> str:
    stmt = retrieve.build_stage1_query(source_id, embedding)
    sql = stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    async with AsyncSessionLocal() as session:
        rows = await session.execute(text(f"EXPLAIN {sql}"))
        return "\n".join(row[0] for row in rows)


def ids(rows: list[dict]) -> list[uuid.UUID]:
    return [row["article_id"] for row in rows]


def test_candidates_are_ranked_by_cosine_distance_not_euclidean(db, fake_rerank):
    # Not unit length on purpose: Euclidean distance would rank `nearby` first.
    source = asyncio.run(add_article("source", vector(1, 0)))
    same_direction = asyncio.run(add_article("same direction", vector(10, 1)))
    nearby = asyncio.run(add_article("nearby", vector(0.5, 0.5)))

    _, similar = asyncio.run(retrieve.main(str(source)))

    assert ids(similar) == [same_direction, nearby]
    assert similar[0]["score"] == pytest.approx(1 - 10 / math.sqrt(101))
    assert similar[1]["score"] == pytest.approx(1 - 1 / math.sqrt(2))


def test_source_and_unembedded_articles_are_never_candidates(db, fake_rerank):
    source = asyncio.run(add_article("source", vector(1, 0)))
    embedded = asyncio.run(add_article("embedded", vector(1, 1)))
    asyncio.run(add_article("not embedded yet", embedding=None))

    top5, similar = asyncio.run(retrieve.main(str(source)))

    assert ids(similar) == [embedded]
    assert ids(top5) == [embedded]


def test_stage2_reranks_the_stage1_candidates_against_the_source_text(db, fake_rerank):
    source = asyncio.run(add_article("rates rise", vector(1, 0)))
    candidate = asyncio.run(add_article("rates rise again", vector(1, 1)))

    top5, _ = asyncio.run(retrieve.main(str(source)))

    [call] = fake_rerank
    assert call["query"] == "rates rise"
    assert ids(call["candidates"]) == [candidate]
    assert call["candidates"][0]["title"] == "Title: rates rise again"
    assert top5 == [{**call["candidates"][0], "score": 1.0}]


def test_stage1_keeps_the_50_nearest_and_stage2_returns_5(db, fake_rerank):
    async def seed():
        source = await add_article("source", vector(1, 0))
        for i in range(55):
            await add_article(f"candidate {i}", vector(1, i / 10))
        return source

    source = asyncio.run(seed())

    top5, similar = asyncio.run(retrieve.main(str(source)))

    assert len(similar) == 50
    assert [row["cleaned_text"] for row in similar] == [f"candidate {i}" for i in range(50)]
    assert len(top5) == 5


def test_the_hnsw_index_serves_stage1_and_still_returns_50_neighbours(index_scans_only, fake_rerank):
    async def seed():
        source = await add_article("source", vector(1, 0))
        for i in range(55):
            await add_article(f"candidate {i}", vector(1, i / 10))
        return source

    source = asyncio.run(seed())

    assert "ix_processed_articles_embedding_hnsw" in asyncio.run(stage1_plan(source, vector(1, 0)))
    _, similar = asyncio.run(retrieve.main(str(source)))
    # pgvector's default hnsw.ef_search (40) would cap an index scan at 40 rows.
    assert [row["cleaned_text"] for row in similar] == [f"candidate {i}" for i in range(50)]


@pytest.mark.parametrize("setup", ["no embedding yet", "not processed yet"])
def test_article_that_cannot_be_compared_gets_no_recommendations(db, fake_rerank, setup):
    asyncio.run(add_article("neighbour", vector(1, 0)))
    if setup == "no embedding yet":
        source = asyncio.run(add_article("source", embedding=None))
    else:
        source = asyncio.run(add_article("source", processed=False))

    assert asyncio.run(retrieve.main(str(source))) == ([], [])
    assert fake_rerank == []


def test_no_embedded_neighbours_means_no_recommendations(db, fake_rerank):
    source = asyncio.run(add_article("source", vector(1, 0)))
    asyncio.run(add_article("not embedded yet", embedding=None))

    assert asyncio.run(retrieve.main(str(source))) == ([], [])
    assert fake_rerank == []


def test_stage1_query_does_not_load_embeddings():
    stmt = retrieve.build_stage1_query(uuid.uuid4(), vector(1, 0))

    assert "embedding" not in [column.name for column in stmt.selected_columns]
