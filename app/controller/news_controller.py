import logging
from fastapi import HTTPException,status
from app.services.news_services import mark_article_as_read, get_unseen_processed_articles_for_user,set_last_read_date, set_article_like
from app.schemas.user_schema import UserResponse
from app.schemas.news_schema import UnseenArticlesQuery, UnseenArticlesResponse,UpdateLastReadRequest

logger = logging.getLogger(__name__)

async def mark_article_as_read_controller(article_id: str, current_user: UserResponse) -> str:
    try:
        result = await mark_article_as_read(article_id, current_user)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    
async def get_unseen_processed_articles_controller(
    current_user: UserResponse,
    params: UnseenArticlesQuery
) -> UnseenArticlesResponse:
    try:
        result = await get_unseen_processed_articles_for_user(current_user, params)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    # except Exception as e:
    #     raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred")
   
async def set_last_read_date_controller(current_user: UserResponse,payload:UpdateLastReadRequest) -> str:
    try:
        result = await set_last_read_date(current_user, payload.last_read_date)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        logger.exception("Failed to set last read date")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred")

async def set_article_like_controller(
    article_id: str, liked: bool, current_user: UserResponse
) -> tuple[str, bool, list[dict], list[dict]]:
    try:
        return await set_article_like(article_id, current_user, liked)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        logger.exception("Failed to set like for article %s", article_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred")
    

