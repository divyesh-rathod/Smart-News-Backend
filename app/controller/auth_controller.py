# app/controllers/user_controller.py
from fastapi import HTTPException, status
from app.schemas.user_schema import UserCreate, UserWithToken
from app.services.user_services import create_user, login_user

async def create_user_controller(user_details: UserCreate) -> UserWithToken:
    try:
        user, token = await create_user(user_details)
    except ValueError as e:
        # map domain error to HTTP 400
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return {"user": user, "access_token": token, "token_type": "bearer"}


async def login_user_controller(email: str, password: str) -> UserWithToken:
    try:
        user, token = await login_user(email, password)
    except ValueError as e:
        # map domain error to HTTP 400
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return {"user": user, "access_token": token, "token_type": "bearer"}