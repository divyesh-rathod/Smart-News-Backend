import os

# app.config requires DATABASE_URL at import time. Unit tests never connect, so any URL works.
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
