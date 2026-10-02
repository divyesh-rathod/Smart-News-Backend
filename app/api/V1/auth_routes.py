# app/api/v1/auth_router.py
from fastapi import APIRouter
from app.schemas.user_schema import UserCreate, UserWithToken, UserLogin
from app.controller.auth_controller import create_user_controller, login_user_controller

router = APIRouter(
    tags=["Auth"],
    responses={404: {"description": "Not found"}},
)

@router.post("/signup", response_model=UserWithToken)
async def signup(user_details: UserCreate):
    return await create_user_controller(user_details)

@router.post("/login", response_model=UserWithToken)
async def login(user_credentials: UserLogin):
    return await login_user_controller(user_credentials.email, user_credentials.password)
