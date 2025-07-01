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
) -> List[Tuple[ProcessedArticle, str, str,float]]:
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
    results = results.all()
    if results:
        print(f"DEBUG: First result format: {len(results[0])} values")
        print(f"DEBUG: First result: {results[0]}")
    return results


async def main(article_id: str = None):
    async with AsyncSessionLocal() as session:
        # 1) Get top-50 by vector distance
        similar = await get_top_50_cosine_similar_articles(session, article_id)
        print(f"DEBUG: Found {len(similar)} similar articles")
        print(f"DEBUG: First similar article: {similar[0]}")
        print("THis is Divyesh debugging")
        candidates = [art_obj for art_obj, title, link, distance in similar]

        # 2) Rerank top-50 with cross-encoder
        #    Fetch the query text
    stmt = (
        select(ProcessedArticle)
        .options(load_only("article_id", "cleaned_text", "category_1", "category_2"))
        .where(ProcessedArticle.article_id == article_id)
    )
    result = await session.execute(stmt)

    query_article = result.scalars().first()
    if not query_article:
            raise ValueError(f"Query article {article_id} not found")
    query_text = query_article.cleaned_text or ""
    detached_similar = []
    for art_obj, title, link, distance in similar:
            # Create plain dictionaries instead of ORM objects
            detached_similar.append({
                'article_id': str(art_obj.article_id),
                'cleaned_text': art_obj.cleaned_text,
                'category_1': art_obj.category_1,
                'category_2': art_obj.category_2,
                'title': title,
                'link': link,
                'distance': distance
            })

    top5 = await rerank_top_k(query_text, detached_similar, top_n=5)

    return top5, similar

        # # 3) Output
        # for art, score in top5:
        #     print(f"{art.article_id} → cross-encoder score {score:.4f}")
        # for art, distance in similar:
        #     print(f"Article ID: {art.article_id}, Cosine Distance: {distance:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
