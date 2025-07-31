from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.schemas import StockRequest, StockAnalysis, ApiResponse, AnalysisType
from app.services.data_collector import DataCollector
from app.services.fundamental_analyzer import FundamentalAnalyzer
from app.services.technical_analyzer import TechnicalAnalyzer
from app.services.sentiment_analyzer import SentimentAnalyzer
from app.services.ml_predictor import MLTargetPricePredictor
from app.services.comprehensive_analyzer import ComprehensiveAnalyzer
from datetime import datetime

router = APIRouter()
data_collector = DataCollector()
fundamental_analyzer = FundamentalAnalyzer()
technical_analyzer = TechnicalAnalyzer()
sentiment_analyzer = SentimentAnalyzer()
ml_predictor = MLTargetPricePredictor()
comprehensive_analyzer = ComprehensiveAnalyzer()

@router.post("/analyze", response_model=ApiResponse)
async def analyze_stock(request: StockRequest, db: Session = Depends(get_db)):
    """Perform comprehensive stock analysis using enhanced analyzers"""
    try:
        symbol = request.symbol
        analysis_type = request.analysis_type
        
        # Use the comprehensive analyzer for all analysis types
        result = comprehensive_analyzer.analyze_stock(symbol, analysis_type)
        
        if not result.get("success", False):
            error_msg = result.get("error", "Analysis failed")
            raise HTTPException(status_code=500, detail=error_msg)
        
        return ApiResponse(
            success=True,
            data=result,
            message=f"Enhanced {analysis_type} analysis completed for {symbol}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fundamental/{symbol}", response_model=ApiResponse)
async def get_fundamental_analysis(symbol: str, db: Session = Depends(get_db)):
    """Get fundamental analysis for a specific stock"""
    try:
        result = fundamental_analyzer.analyze(symbol)
        return ApiResponse(
            success=True,
            data=result,
            message=f"Fundamental analysis completed for {symbol}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/technical/{symbol}", response_model=ApiResponse)
async def get_technical_analysis(symbol: str, period: str = "1y", db: Session = Depends(get_db)):
    """Get technical analysis for a specific stock"""
    try:
        result = technical_analyzer.analyze(symbol, period)
        return ApiResponse(
            success=True,
            data=result,
            message=f"Technical analysis completed for {symbol}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sentiment/{symbol}", response_model=ApiResponse)
async def get_sentiment_analysis(symbol: str, days_back: int = 30, db: Session = Depends(get_db)):
    """Get sentiment analysis for a specific stock"""
    try:
        result = sentiment_analyzer.analyze(symbol, days_back)
        return ApiResponse(
            success=True,
            data=result,
            message=f"Sentiment analysis completed for {symbol}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ml-prediction/{symbol}", response_model=ApiResponse)
async def get_ml_prediction(symbol: str, horizon_days: int = 30, db: Session = Depends(get_db)):
    """Get ML-based target price prediction for a specific stock"""
    try:
        result = ml_predictor.predict_target_price(symbol, horizon_days)
        return ApiResponse(
            success=True,
            data=result,
            message=f"ML prediction completed for {symbol}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
