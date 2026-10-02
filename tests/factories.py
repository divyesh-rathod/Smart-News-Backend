import uuid
from datetime import datetime, timezone

from app.db.models import Article, ProcessedArticle, User
from app.db.session import AsyncSessionLocal

EMBEDDING_DIM = 384


def vector(*values: float) -> list[float]:
    """A 384-d embedding whose leading components are `values` and the rest zero."""
    return list(values) + [0.0] * (EMBEDDING_DIM - len(values))


async def add_article(
    text: str = "some article text",
    embedding: list[float] | None = None,
    processed: bool = True,
) -> uuid.UUID:
    """Insert an article and, if `processed`, its processed_articles row. Returns the article id."""
    async with AsyncSessionLocal() as session:
        article = Article(
            title=f"Title: {text}",
            link=f"https://example.com/{uuid.uuid4()}",
            pub_date=datetime(2026, 10, 1, tzinfo=timezone.utc),
            description=text,
            categories=["News"],
            processed=processed,
        )
        session.add(article)
        await session.flush()
        if processed:
            session.add(ProcessedArticle(
                article_id=article.id,
                cleaned_text=text,
                category_1="News",
                category_2=f"{text} news",
                embedding=embedding,
            ))
        await session.commit()
        return article.id


async def add_user() -> User:
    async with AsyncSessionLocal() as session:
        suffix = uuid.uuid4().hex[:8]
        user = User(
            name="Test User",
            email=f"user-{suffix}@example.com",
            phone_number=f"+1555{suffix}",
            password="not-a-real-hash",
        )
        session.add(user)
        await session.commit()
        return user
