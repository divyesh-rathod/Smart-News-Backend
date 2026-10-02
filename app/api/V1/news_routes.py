from fastapi import APIRouter, Depends
from app.controller.news_controller import mark_article_as_read_controller,get_unseen_processed_articles_controller, set_last_read_date_controller, toggle_article_like_controller
from app.schemas.user_schema import UserResponse   
from app.schemas.news_schema import ToggleLikeResponse, UnseenArticlesResponse, UnseenArticlesQuery, UpdateLastReadRequest
from app.services.news_services import serialize_article_scores
from app.utils.auth import get_current_user 

router = APIRouter(
    tags=["News"],
    dependencies=[Depends(get_current_user)],  # protect every endpoint here
)

@router.post("/mark-as-read/{article_id}", response_model=str)
async def mark_article_as_read(article_id: str, current_user: UserResponse = Depends(get_current_user)):
 
    return await mark_article_as_read_controller(article_id, current_user)  


@router.get("/unseen-articles", response_model=UnseenArticlesResponse)
async def get_unseen_processed_articles(
    current_user: UserResponse = Depends(get_current_user),
    params: UnseenArticlesQuery = Depends()
) -> UnseenArticlesResponse:
    return await get_unseen_processed_articles_controller(current_user, params)

@router.post("/set-date", response_model=str)
async def set_last_read_date(
    current_user: UserResponse = Depends(get_current_user),
    payload: UpdateLastReadRequest = Depends()
) -> str:
    return await set_last_read_date_controller(current_user, payload)

@router.post("/toggle-like/{article_id}", response_model=ToggleLikeResponse)
async def toggle_article_like(
    article_id: str,
    current_user: UserResponse = Depends(get_current_user)
) -> ToggleLikeResponse:
    message, liked, raw_top5, raw_similar = await toggle_article_like_controller(article_id, current_user)
    top5_serialized = serialize_article_scores(raw_top5)
    similar_serialized = serialize_article_scores(raw_similar)
    return ToggleLikeResponse(
        message=message,
        liked=liked,
        top5=top5_serialized,
        similar=similar_serialized
    )



