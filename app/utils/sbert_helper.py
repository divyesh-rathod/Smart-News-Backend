import asyncio
import functools
from sentence_transformers import SentenceTransformer
from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.db.models.processed_article import ProcessedArticle


@functools.cache
def get_model() -> SentenceTransformer:
    """Load SBERT on first use, so importing this module doesn't download weights."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def generate_embedding(text: str) -> list[float] | None:
    """Unit-length SBERT embedding of a cleaned text string, or None for empty text."""
    if not text:
        return None
    return get_model().encode(text, normalize_embeddings=True).tolist()

async def embed_articles(batch_size: int = 100) -> int:
    """Embed up to batch_size articles that have text but no embedding yet. Returns how many were embedded."""
    async with AsyncSessionLocal() as session:
        # Articles without text can never be embedded; selecting them would return them in every batch.
        result = await session.execute(
            select(ProcessedArticle)
            .filter(ProcessedArticle.embedding.is_(None))
            .filter(ProcessedArticle.category_2.is_not(None), ProcessedArticle.category_2 != "")
            .limit(batch_size)
        )
        articles = result.scalars().all()

        for article in articles:
            # run the blocking generate_embedding in a thread pool
            article.embedding = await asyncio.get_running_loop().run_in_executor(
                None, generate_embedding, article.category_2
            )

        await session.commit()
        return len(articles)


