import asyncio

import numpy as np
import pytest
from sqlalchemy import update

from app.db.models import ProcessedArticle
from app.db.session import AsyncSessionLocal
from app.ml_models import generate_embeddings
from app.utils import sbert_helper
from tests.factories import add_article, vector


async def set_vector_input(article_id, category_2: str | None) -> None:
    async with AsyncSessionLocal() as session:
        await session.execute(
            update(ProcessedArticle)
            .where(ProcessedArticle.article_id == article_id)
            .values(category_2=category_2)
        )
        await session.commit()


async def stored_embedding(article_id):
    async with AsyncSessionLocal() as session:
        return (await session.get(ProcessedArticle, article_id)).embedding


@pytest.mark.model
def test_generate_embedding_is_unit_length():
    embedding = sbert_helper.generate_embedding("the central bank raised interest rates")

    assert len(embedding) == 384
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-5)


def test_generate_embedding_skips_empty_text():
    assert sbert_helper.generate_embedding("") is None
    assert sbert_helper.generate_embedding(None) is None


def test_embedding_run_skips_articles_without_text_and_finishes(db, monkeypatch):
    monkeypatch.setattr(sbert_helper, "generate_embedding", lambda text: vector(1, 0))

    async def seed():
        with_text = await add_article("rates rise")
        empty = await add_article("empty")
        missing = await add_article("missing")
        await set_vector_input(empty, "")
        await set_vector_input(missing, None)
        return with_text, empty, missing

    with_text, empty, missing = asyncio.run(seed())

    # Before the fix, the two articles without text were re-selected every batch and the run never ended.
    total = asyncio.run(asyncio.wait_for(generate_embeddings.main(), timeout=30))

    assert total == 1
    assert list(asyncio.run(stored_embedding(with_text))) == vector(1, 0)
    assert asyncio.run(stored_embedding(empty)) is None
    assert asyncio.run(stored_embedding(missing)) is None
