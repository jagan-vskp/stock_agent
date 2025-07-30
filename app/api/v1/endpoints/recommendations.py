from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from app.models.schemas import Recommendation, ApiResponse
from app.services.fundamental_analyzer import FundamentalAnalyzer
from app.services.data_collector import DataCollector
from datetime import datetime

router = APIRouter()
fundamental_analyzer = FundamentalAnalyzer()
data_collector = DataCollector()

@router.get("/top-picks", response_model=ApiResponse)
async def get_top_recommendations(limit: int = 10, db: Session = Depends(get_db)):
    """Get top stock recommendations"""
    try:
        # Sample Indian stocks for demonstration
        sample_stocks = [
            "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS",
            "KOTAKBANK.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS", "ASIANPAINT.NS"
        ]
        
        recommendations = []
        
        for symbol in sample_stocks[:limit]:
            try:
                # Get fundamental analysis
                analysis = fundamental_analyzer.analyze(symbol)
                stock_info = data_collector.get_stock_info(symbol)
                
                if analysis.get("overall_score", 0) > 60:  # Only include good scores
                    recommendation = {
                        "symbol": symbol,
                        "name": stock_info.get("name", "N/A"),
                        "recommendation": analysis.get("recommendation", "HOLD"),
                        "current_price": stock_info.get("current_price", 0),
                        "target_price": stock_info.get("current_price", 0) * 1.1,  # 10% upside
                        "confidence_score": analysis.get("overall_score", 0) / 100,
                        "reasoning": f"Based on fundamental analysis with score: {analysis.get('overall_score', 0):.1f}",
                        "fundamental_score": analysis.get("overall_score", 0) / 100,
                        "created_at": datetime.now().isoformat()
                    }
                    recommendations.append(recommendation)
            except Exception as e:
                continue  # Skip stocks with errors
        
        # Sort by confidence score
        recommendations.sort(key=lambda x: x["confidence_score"], reverse=True)
        
        return ApiResponse(
            success=True,
            data=recommendations,
            message=f"Retrieved {len(recommendations)} recommendations"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{symbol}", response_model=ApiResponse)
async def get_stock_recommendation(symbol: str, db: Session = Depends(get_db)):
    """Get recommendation for a specific stock"""
    try:
        # Get fundamental analysis
        analysis = fundamental_analyzer.analyze(symbol)
        stock_info = data_collector.get_stock_info(symbol)
        
        if not stock_info:
            raise HTTPException(status_code=404, detail="Stock not found")
        
        current_price = stock_info.get("current_price", 0)
        overall_score = analysis.get("overall_score", 0)
        
        # Calculate target price based on analysis
        if overall_score >= 75:
            target_multiplier = 1.15  # 15% upside for BUY
        elif overall_score >= 60:
            target_multiplier = 1.05  # 5% upside for HOLD
        else:
            target_multiplier = 0.95  # 5% downside for SELL
        
        recommendation = {
            "symbol": symbol,
            "name": stock_info.get("name", "N/A"),
            "recommendation": analysis.get("recommendation", "HOLD"),
            "current_price": current_price,
            "target_price": current_price * target_multiplier,
            "confidence_score": overall_score / 100,
            "reasoning": "; ".join(analysis.get("insights", [])),
            "fundamental_score": overall_score / 100,
            "technical_score": None,
            "sentiment_score": None,
            "created_at": datetime.now().isoformat(),
            "detailed_analysis": analysis
        }
        
        return ApiResponse(
            success=True,
            data=recommendation,
            message=f"Recommendation generated for {symbol}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
