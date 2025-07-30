from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "Indian Stock Recommendation System"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./data/stocks.db"
    
    # Redis settings
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 3600  # 1 hour
    
    # API Keys (set these in environment variables)
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    NEWS_API_KEY: Optional[str] = None
    
    # Data collection settings
    DATA_UPDATE_INTERVAL: int = 300  # 5 minutes
    MAX_STOCKS_PER_REQUEST: int = 10
    
    # ML Model settings
    MODEL_RETRAIN_INTERVAL: int = 86400  # 24 hours
    PREDICTION_CONFIDENCE_THRESHOLD: float = 0.7
    
    class Config:
        env_file = ".env"


settings = Settings()
