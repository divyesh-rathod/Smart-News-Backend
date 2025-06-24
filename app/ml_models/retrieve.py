# app/services/similarity_service.py

import asyncio
from typing import List, Tuple

from sqlalchemy import Float, asc
from sqlalchemy import select
from sqlalchemy.orm import load_only
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import AsyncSessionLocal
from app.db.models.processed_article import ProcessedArticle, Article
from app.ml_models.rerank import rerank_top_k

async def get_top_50_cosine_similar_articles(
    session: AsyncSession,
    article_id: str
) -> List[Tuple[ProcessedArticle, float]]:
    """
    Return the 50 nearest neighbors by cosine distance (using pgvector's <-> operator).
    """
    # 1) Fetch the source article
    result = await session.execute(
        select(ProcessedArticle).where(ProcessedArticle.article_id == article_id)
    )
    source_article = result.scalars().first()
    if not source_article:
        raise ValueError(f"No article found with article_id: {article_id}")
    if source_article.embedding is None:
        raise ValueError(f"Article {article_id} does not have an embedding.")

    # 2) Build the distance expression
    distance_expr = (
        ProcessedArticle.embedding
        .op("<->")(source_article.embedding)  # Euclidean on normalized → cosine
        .cast(Float)
        .label("distance")
    )

    # 3) Query for nearest 50 (excluding the source)
    stmt = (
        select(
            ProcessedArticle,
            Article.title,
            Article.link,
            distance_expr
        )
        .join(Article, ProcessedArticle.article_id == Article.id)  
        .where(ProcessedArticle.article_id != article_id)
        .order_by(asc(distance_expr))
        .limit(50)
    )
    results = await session.execute(stmt)
    return results.all()


async def main(article_id: str = None):
    example_article_id = "fecf8133-412a-40ae-9462-f4e86308e843"
    async with AsyncSessionLocal() as session:
        # 1) Get top-50 by vector distance
        similar = await get_top_50_cosine_similar_articles(session, article_id)
        candidates = [art for art, _ in similar]

        # 2) Rerank top-50 with cross-encoder
        #    Fetch the query text
    stmt = (
        select(ProcessedArticle)
        .options(load_only("article_id", "cleaned_text", "category_1", "category_2"))
        .where(ProcessedArticle.article_id == example_article_id)
    )
    result = await session.execute(stmt)

    query_text = result.scalars().first() or ""

    top5 = await rerank_top_k(query_text, candidates, top_n=5)

    return top5, similar

        # # 3) Output
        # for art, score in top5:
        #     print(f"{art.article_id} → cross-encoder score {score:.4f}")
        # for art, distance in similar:
        #     print(f"Article ID: {art.article_id}, Cosine Distance: {distance:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
