"""The local corpus (every article with an embedding), loaded into memory for offline evaluation."""

import asyncio
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import select

from app.db.models import Article, ProcessedArticle
from app.db.session import AsyncSessionLocal, engine


@dataclass(frozen=True)
class Doc:
    link: str
    title: str
    text: str  # cleaned_text: what the cross-encoder reads
    vector_input: str  # category_2: what SBERT embedded (description, then categories)
    categories: tuple[str, ...]


@dataclass
class Corpus:
    docs: list[Doc]
    embeddings: np.ndarray  # (n, 384), unit length; row i is docs[i]
    row: dict[str, int] = field(init=False)  # link -> row

    def __post_init__(self):
        self.row = {doc.link: i for i, doc in enumerate(self.docs)}


async def _load() -> Corpus:
    stmt = (
        select(
            Article.link,
            Article.title,
            Article.categories,
            ProcessedArticle.cleaned_text,
            ProcessedArticle.category_2,
            ProcessedArticle.embedding,
        )
        .join(ProcessedArticle, ProcessedArticle.article_id == Article.id)
        .where(ProcessedArticle.embedding.is_not(None))
        .order_by(Article.link)
    )
    try:
        async with AsyncSessionLocal() as session:
            rows = (await session.execute(stmt)).all()
    finally:
        # Callers run more event loops later; pooled asyncpg connections can't move between loops.
        await engine.dispose()
    docs = [
        Doc(link, title, text or "", vector_input or "", tuple(categories or ()))
        for link, title, categories, text, vector_input, _ in rows
    ]
    embeddings = np.array([row.embedding for row in rows], dtype=np.float32)
    return Corpus(docs, embeddings)


def load_corpus() -> Corpus:
    return asyncio.run(_load())
