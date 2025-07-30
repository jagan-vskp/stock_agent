from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.schemas import StockRequest, StockAnalysis, ApiResponse, AnalysisType
from app.services.data_collector import DataCollector
from app.services.fundamental_analyzer import FundamentalAnalyzer
from datetime import datetime

router = APIRouter()
data_collector = DataCollector()
fundamental_analyzer = FundamentalAnalyzer()

@router.post("/analyze", response_model=ApiResponse)
async def analyze_stock(request: StockRequest, db: Session = Depends(get_db)):
    """Perform comprehensive stock analysis"""
    try:
        symbol = request.symbol
        analysis_type = request.analysis_type
        
        result = {}
        
        # Get basic stock info
        stock_info = data_collector.get_stock_info(symbol)
        if not stock_info:
            raise HTTPException(status_code=404, detail="Stock not found")
        
        result["stock_info"] = stock_info
        result["current_price"] = stock_info.get("current_price", 0)
        
        # Perform different types of analysis based on request
        if analysis_type in [AnalysisType.FUNDAMENTAL, AnalysisType.COMPREHENSIVE]:
            fundamental_result = fundamental_analyzer.analyze(symbol)
            result["fundamental"] = fundamental_result
        
        if analysis_type in [AnalysisType.TECHNICAL, AnalysisType.COMPREHENSIVE]:
            # Placeholder for technical analysis
            result["technical"] = {
                "message": "Technical analysis coming soon",
                "rsi": 55.0,
                "macd": 1.2
            }
        
        if analysis_type in [AnalysisType.SENTIMENT, AnalysisType.COMPREHENSIVE]:
            # Get news for sentiment analysis
            news_data = data_collector.get_news_data(symbol)
            result["sentiment"] = {
                "news_count": len(news_data),
                "news_data": news_data,
                "sentiment_score": 0.7  # Placeholder
            }
        
        result["analysis_timestamp"] = datetime.now().isoformat()
        
        return ApiResponse(
            success=True,
            data=result,
            message=f"Analysis completed for {symbol}"
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
