from fastapi import APIRouter, Depends
from app.controller.news_controller import mark_article_as_read_controller,get_unseen_processed_articles_controller, set_last_read_date_controller, set_article_like_controller
from app.schemas.user_schema import UserResponse   
from app.schemas.news_schema import LikeResponse, SetLikeRequest, UnseenArticlesResponse, UnseenArticlesQuery, UpdateLastReadRequest
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

@router.put("/like/{article_id}", response_model=LikeResponse)
async def set_article_like(
    article_id: str,
    payload: SetLikeRequest,
    current_user: UserResponse = Depends(get_current_user)
) -> LikeResponse:
    message, liked, raw_top5, raw_similar = await set_article_like_controller(article_id, payload.liked, current_user)
    return LikeResponse(
        message=message,
        liked=liked,
        top5=serialize_article_scores(raw_top5),
        similar=serialize_article_scores(raw_similar),
    )
