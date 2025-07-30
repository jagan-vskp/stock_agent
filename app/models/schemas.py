from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class RecommendationType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class AnalysisType(str, Enum):
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    COMPREHENSIVE = "comprehensive"


class StockRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., RELIANCE.NS)")
    analysis_type: AnalysisType = Field(default=AnalysisType.COMPREHENSIVE)


class FundamentalData(BaseModel):
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    book_value: Optional[float] = None
    debt_to_equity: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    profit_margin: Optional[float] = None
    revenue_growth: Optional[float] = None
    dividend_yield: Optional[float] = None
    market_cap: Optional[float] = None


class TechnicalData(BaseModel):
    rsi: Optional[float] = None
    macd: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None


class SentimentData(BaseModel):
    news_sentiment: Optional[float] = None
    social_sentiment: Optional[float] = None
    overall_sentiment: Optional[float] = None
    news_count: Optional[int] = None


class StockAnalysis(BaseModel):
    symbol: str
    current_price: float
    fundamental: Optional[FundamentalData] = None
    technical: Optional[TechnicalData] = None
    sentiment: Optional[SentimentData] = None
    analysis_timestamp: datetime


class Recommendation(BaseModel):
    symbol: str
    recommendation: RecommendationType
    target_price: float
    current_price: float
    confidence_score: float
    reasoning: str
    fundamental_score: Optional[float] = None
    technical_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    created_at: datetime


class StockInfo(BaseModel):
    symbol: str
    name: str
    exchange: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    current_price: Optional[float] = None


class ApiResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    error: Optional[str] = None
