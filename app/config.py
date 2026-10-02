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
    
    # Additional settings can be added here:
    # For instance, port, host settings, API version, etc.
    APP_HOST: str = Field("127.0.0.1", env="APP_HOST")
    APP_PORT: int = Field(8080, env="APP_PORT")

    class Config:
        env_file = str(env_path)
        env_file_encoding = "utf-8"

# Instantiate the settings object which will be used throughout your app:
settings = Settings()
