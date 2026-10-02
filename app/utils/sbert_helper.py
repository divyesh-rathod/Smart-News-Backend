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

async def embed_articles(batch_size: int = 100) -> str:
    """
    Loop through articles with null embeddings and generate them asynchronously.
    """
    async with AsyncSessionLocal() as session:  
        # fetch articles needing embeddings
        result = await session.execute(
            select(ProcessedArticle)
            .filter(ProcessedArticle.embedding.is_(None))
            .limit(batch_size)
        )
        articles = result.scalars().all()

        for article in articles:
            if article.category_2:
                # run the blocking generate_embedding in a thread pool
                embedding = await asyncio.get_running_loop().run_in_executor(
                    None, generate_embedding, article.category_2
                )
                article.embedding = embedding

        await session.commit()
        return f"{len(articles)} articles embedded and saved."


