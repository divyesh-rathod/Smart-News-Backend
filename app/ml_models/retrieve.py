# app/ml_models/retrieve.py

import logging
from uuid import UUID

from sqlalchemy import Select, select, text

from app.db.session import AsyncSessionLocal
from app.db.models import Article, ProcessedArticle
from app.ml_models.rerank import rerank_top_k

logger = logging.getLogger(__name__)

STAGE1_LIMIT = 50
# An HNSW index scan returns at most hnsw.ef_search rows (pgvector default 40), so this must exceed
# STAGE1_LIMIT plus the excluded source row. Higher also means better recall and slower queries.
HNSW_EF_SEARCH = 100


def build_stage1_query(source_article_id: UUID, source_embedding) -> Select:
    """The STAGE1_LIMIT nearest embedded articles to source_embedding by cosine distance, excluding the source."""
    # Only a vector_cosine_ops index can serve ORDER BY <=>; Postgres ignores a vector_l2_ops one.
    distance = ProcessedArticle.embedding.cosine_distance(source_embedding).label("distance")
    return (
        select(
            ProcessedArticle.article_id,
            ProcessedArticle.cleaned_text,
            ProcessedArticle.category_1,
            ProcessedArticle.category_2,
            Article.title,
            Article.link,
            distance,
        )
        .join(Article, ProcessedArticle.article_id == Article.id)
        .where(ProcessedArticle.article_id != source_article_id)
        .where(ProcessedArticle.embedding.is_not(None))
        .order_by(distance)
        .limit(STAGE1_LIMIT)
    )


async def main(article_id: str) -> tuple[list[dict], list[dict]]:
    """
    Recommend articles similar to article_id: pgvector top 50 (stage 1), then cross-encoder top 5 (stage 2).

    Returns (top5, similar). `similar` is stage 1 with score = cosine distance; top5 has score = cross-encoder
    score. Both are empty when the article hasn't been processed or embedded yet, or no other article has.
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ProcessedArticle).where(ProcessedArticle.article_id == article_id)
        )
        source = result.scalars().first()
        if source is None or source.embedding is None:
            logger.info("Article %s has no embedding yet, so no recommendations", article_id)
            return [], []

        await session.execute(text(f"SET LOCAL hnsw.ef_search = {HNSW_EF_SEARCH}"))
        result = await session.execute(build_stage1_query(source.article_id, source.embedding))
        candidates = [dict(row) for row in result.mappings()]

    logger.debug("Stage 1 returned %d candidates for article %s", len(candidates), article_id)
    if not candidates:
        return [], []

    top5 = await rerank_top_k(source.cleaned_text or "", candidates, top_n=5)
    similar = [{**candidate, "score": candidate["distance"]} for candidate in candidates]
    return top5, similar
