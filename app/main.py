# app/main.py - Complete CORS setup

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import api_router
from .config import secret_key_problem, settings
from .ml_models import rerank


@asynccontextmanager
async def lifespan(app: FastAPI):
    if problem := secret_key_problem(settings.SECRET_KEY):
        raise RuntimeError(
            f"Refusing to start: SECRET_KEY {problem}, so anyone could forge login tokens. "
            'Set it in .env; generate one with: python -c "import secrets; print(secrets.token_urlsafe(32))"'
        )
    # uvicorn only configures its own loggers; without this, app logs below WARNING are dropped.
    logging.basicConfig(level=settings.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Hugging Face's update checks would log every HTTP request at INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Load the cross-encoder before serving, so the first like doesn't pay for it.
    await asyncio.to_thread(rerank.get_model)
    yield


app = FastAPI(
    title="Smart News API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware must be added BEFORE including routers
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative React dev server
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include routers AFTER CORS middleware
app.include_router(api_router, prefix="/api")