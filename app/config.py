from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class Settings(BaseSettings):
    # Define the settings you need, with defaults and environment variable names.
    # For example, a PostgreSQL connection URL:
    DATABASE_URL: str = Field(..., env="DATABASE_URL")
    
    # Other configuration values, such as secret keys or debug mode:
    SECRET_KEY: str = Field("your-default-secret", env="SECRET_KEY")
    ALGORITHM: str = Field("HS256", env="ALGORITHM")
    ACCESS_TOKEN_EXPIRE_DAYS: int = Field(15, env="ACCESS_TOKEN_EXPIRE_DAYS")

    DEBUG: bool = Field(False, env="DEBUG")
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")

    # Shared secret for admin-only endpoints (sent as X-Admin-Token). Unset or empty disables them.
    ADMIN_API_KEY: str | None = Field(None, env="ADMIN_API_KEY")

    # Cross-encoder runtime. MODEL_DEVICE is a torch device ("cpu", "cuda", "mps").
    # TORCH_NUM_THREADS unset keeps torch's default; lower it when several workers share a machine.
    MODEL_DEVICE: str = Field("cpu", env="MODEL_DEVICE")
    TORCH_NUM_THREADS: int | None = Field(None, env="TORCH_NUM_THREADS")

    # Per-process cache of each liked article's recommendations. The TTL bounds staleness after a CLI
    # ingest, which the API can't observe; a pipeline run triggered through the API clears it right away.
    RECOMMENDATION_CACHE_SIZE: int = Field(256, env="RECOMMENDATION_CACHE_SIZE")
    RECOMMENDATION_CACHE_TTL_SECONDS: int = Field(900, env="RECOMMENDATION_CACHE_TTL_SECONDS")
    
    # Additional settings can be added here:
    # For instance, port, host settings, API version, etc.
    APP_HOST: str = Field("127.0.0.1", env="APP_HOST")
    APP_PORT: int = Field(8080, env="APP_PORT")

    class Config:
        env_file = str(env_path)
        env_file_encoding = "utf-8"

# Instantiate the settings object which will be used throughout your app:
settings = Settings()

# Keys that must never sign real tokens: the default above and the .env.example placeholder.
PLACEHOLDER_SECRET_KEYS = {"your-default-secret", "your_secret_key_here"}
MIN_SECRET_KEY_LENGTH = 32


def secret_key_problem(secret_key: str) -> str | None:
    """Why `secret_key` is unsafe for signing JWTs, or None if it's fine."""
    if secret_key in PLACEHOLDER_SECRET_KEYS:
        return "is a placeholder"
    if len(secret_key) < MIN_SECRET_KEY_LENGTH:
        return f"is shorter than {MIN_SECRET_KEY_LENGTH} characters"
    return None
