import asyncio
from sentence_transformers import SentenceTransformer
from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.db.models.processed_article import ProcessedArticle
import numpy as np

# Load model once (singleton-style)
model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embedding(text: str) -> list[float]:
    """
    Generate SBERT embedding from a cleaned text string.
    """
    if not text:
        return None
    embedding = model.encode(text)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding.tolist()
    return (embedding / norm).tolist()

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


