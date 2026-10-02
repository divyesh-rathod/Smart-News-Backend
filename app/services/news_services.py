import logging
from app.db.models import Article, Like, ProcessedArticle, User, UserFeedPosition, UserRead
from typing import List
from datetime import datetime, timezone
from sqlalchemy import update
from sqlalchemy.orm import selectinload
from sqlalchemy import select, and_
from app.db.session import AsyncSessionLocal
from app.schemas.news_schema import UnseenProcessedArticle, UnseenArticlesResponse, UnseenArticlesQuery, ArticleScore
from app.ml_models.retrieve import main
from app.services import recommendation_cache

from uuid import UUID

logger = logging.getLogger(__name__)


async def mark_article_as_read(article_id: UUID, current_user: User) -> str:
    async with AsyncSessionLocal() as session:
        # First, verify the article exists
        result = await session.execute(
            select(Article).where(Article.id == article_id)
        )
        article = result.scalar_one_or_none()
        if not article:
            raise ValueError("Article not found")
        
        # Use PostgreSQL's ON CONFLICT DO NOTHING for atomic upsert
        from sqlalchemy.dialects.postgresql import insert
        
        stmt = insert(UserRead).values(
            user_id=current_user.id,
            article_id=article_id
        )
        # If the record already exists, do nothing (no error thrown)
        stmt = stmt.on_conflict_do_nothing(index_elements=['user_id', 'article_id'])
        
        await session.execute(stmt)
        await session.commit()
        
        return "Article marked as read successfully"
      
async def get_unseen_processed_articles_for_user(
    current_user: User,
    params: UnseenArticlesQuery
) -> UnseenArticlesResponse:
    async with AsyncSessionLocal() as session:

        PA, A, UR = ProcessedArticle, Article, UserRead

        # 1) Load or create the feed position row
        feed_pos = await session.get(UserFeedPosition, current_user.id)
        if not feed_pos:
            feed_pos = UserFeedPosition(user_id=current_user.id, cursor=None)
            session.add(feed_pos)
            await session.flush()     # now it exists with cursor=None

        current_cursor = feed_pos.cursor

        # 2) Fetch the next batch of unread articles
        base_q = (
            select(PA)
            .options(selectinload(ProcessedArticle.article))  
            .join(A, PA.article_id == A.id)
            .outerjoin(UR, and_(UR.article_id == A.id,
                                UR.user_id    == current_user.id))
            .where(UR.article_id.is_(None))
        )

        if current_cursor is None:
            base_q = base_q.order_by(A.pub_date.desc())
        else:
            base_q = base_q.where(A.pub_date < current_cursor)\
                           .order_by(A.pub_date.desc())

        stmt = base_q.limit(params.limit).offset(params.offset or 0)
        result = await session.execute(stmt)
        articles = result.scalars().all()
        response_items = serialize_processed_articles(articles)

        # 3) Compute the new cursor as the oldest pub_date in this page
        if articles:
            next_cursor = articles[-1].article.pub_date
        else:
            next_cursor = None

        # 4) Update the cursor on the existing ORM object and commit
        feed_pos.cursor = next_cursor
        await session.commit()

        # 5) Return the response
        return UnseenArticlesResponse(
            results=response_items,
            next_cursor=next_cursor
        )

       
       
    
async def set_last_read_date(current_user: User, explicit_date: datetime | None = None) -> str:
  
    async with AsyncSessionLocal() as session:  
        # Check if row exists
        row = await session.get(UserFeedPosition, current_user.id)
        if row:
            new_date = explicit_date or datetime.now(timezone.utc)
            stmt = (
                update(UserFeedPosition)
                .where(UserFeedPosition.user_id == current_user.id)
                .values(last_read_date=new_date, updated_at=datetime.now(timezone.utc))
            )
            await session.execute(stmt)
        else:
            new_date = explicit_date or datetime.now(timezone.utc)
            new_row = UserFeedPosition(user_id=current_user.id, last_read_date=new_date)
            session.add(new_row)



        await session.commit()
        return "Last read date updated successfully"
    


async def toggle_article_like(
    article_id: UUID,
    current_user: User
) -> tuple[str, bool, list[dict], list[dict]]:
    """Flip the user's like and commit it. Recommendations for a new like are best effort and never undo it."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Article.id).where(Article.id == article_id))
        if result.scalar_one_or_none() is None:
            raise ValueError("Article not found")

        result = await session.execute(
            select(Like).where(
                Like.article_id == article_id,
                Like.user_id    == current_user.id
            )
        )
        existing_like = result.scalar_one_or_none()

        if existing_like:
            existing_like.is_liked = existing_like.is_liked ^ 1
            liked = existing_like.is_liked == 1
        else:
            session.add(Like(
                user_id    = current_user.id,
                article_id = article_id,
                is_liked   = 1
            ))
            liked = True
        await session.commit()

    if not liked:
        return "Like removed", False, [], []
    top5, similar = await recommend_similar_articles(article_id)
    return "Article liked", True, top5, similar


async def recommend_similar_articles(article_id: UUID) -> tuple[list[dict], list[dict]]:
    """retrieve.main for a just-liked article (cached per article), or empty lists (logged) if ranking fails."""
    key = str(article_id)
    cached = recommendation_cache.get(key)
    if cached is not None:
        return cached
    try:
        recommendations = await main(key)
    except Exception:
        logger.exception("Could not compute recommendations for liked article %s", article_id)
        return [], []
    # Empty means not embedded yet or no neighbours yet; caching that would hide later results for the TTL.
    if recommendations[0]:
        recommendation_cache.put(key, recommendations)
    return recommendations


def serialize_article_scores(raw: list[dict]) -> list[ArticleScore]:
    """Convert the top5 / similar dicts from retrieve.main into ArticleScore models."""
    return [ArticleScore.model_validate(row) for row in raw]

def serialize_processed_articles(
    pa_list: List[ProcessedArticle]
) -> List[UnseenProcessedArticle]:
    """
    Turn a list of ProcessedArticle (with .article relationship loaded)
    into a liof UnseenProcessedArticle Pydantic model
    """
    out: List[UnseenProcessedArticle] = []
    for pa in pa_list:
        art = pa.article
        out.append(UnseenProcessedArticle(
            article_id   = pa.article_id,
            cleaned_text = pa.cleaned_text,
            category_1   = pa.category_1,
            category_2   = pa.category_2,
            processed_at = pa.processed_at,
            pub_date     = art.pub_date,
            title        = art.title,
            link         = art.link,
            description  = art.description,
            categories   = art.categories,
        ))
    return out


    

    

        
     
