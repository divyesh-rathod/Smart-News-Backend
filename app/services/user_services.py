import asyncio
from sqlalchemy import select
from passlib.context import CryptContext
from app.db.session import AsyncSessionLocal
from app.db.models.user import User
from app.schemas.user_schema import UserCreate,UpdateUserSchema
from app.utils.auth import create_access_token

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def create_user(user_details: UserCreate) -> User:
   
    loop = asyncio.get_running_loop()
    hashed_pw = await loop.run_in_executor(None, pwd_context.hash, user_details.password)

    if await email_exists(user_details.email):
        raise ValueError("Email already exists")    
    
    async with AsyncSessionLocal() as session:  
        new_user = User(
            name=user_details.name,
            email=user_details.email,
            phone_number=user_details.phone_number,
            password=hashed_pw,
        )
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)

    # At this point new_user.id is available
    access_token = await create_access_token(
        data={"user_id": str(new_user.id), "email": new_user.email}
    )

    return new_user, access_token

async def get_user_by_email(email: str) -> User | None:

    async with AsyncSessionLocal() as session:  
        result = await session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()
    
async def email_exists(email:str)-> bool:
    async with AsyncSessionLocal() as session:  
        result = await session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none() is not None
    
async def login_user(email: str, password: str) -> tuple[User, str]:
    user = await get_user_by_email(email)
    
    if not user or not pwd_context.verify(password, user.password):
        raise ValueError("Invalid email or password")
    
    if user.is_blocked :
        raise ValueError("User is blocked")
    if user.is_deleted:
        raise ValueError("Your account has been deleted")
    
    access_token = await create_access_token(
        data={"user_id": str(user.id), "email": user.email}
    )
    
    return user, access_token

async def update_user(
    user_details: UpdateUserSchema,
    current_user: User
) -> User:
    async with AsyncSessionLocal() as session:  
   
        result = await session.execute(
            select(User).where(User.id == current_user.id)
        )
        user = result.scalar_one_or_none()
        if not user:
            raise ValueError("User not found")

    
        if user_details.name:
            user.name = user_details.name
        if user_details.phone_number:
            user.phone_number = user_details.phone_number
        if user_details.profile_picture:
            user.profile_picture = user_details.profile_picture

       
        await session.commit()
        await session.refresh(user)

        return user


async def get_user_by_id(user_id: str) -> User | None:
    async with AsyncSessionLocal() as session:  
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()  
    
