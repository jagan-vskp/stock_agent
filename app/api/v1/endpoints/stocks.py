from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from app.models.schemas import StockInfo, ApiResponse
from app.services.data_collector import DataCollector

router = APIRouter()
data_collector = DataCollector()

@router.get("/info/{symbol}", response_model=ApiResponse)
async def get_stock_info(symbol: str, db: Session = Depends(get_db)):
    """Get basic stock information"""
    try:
        stock_data = data_collector.get_stock_info(symbol)
        if not stock_data:
            raise HTTPException(status_code=404, detail="Stock not found")
        
        return ApiResponse(
            success=True,
            data=stock_data,
            message=f"Stock information retrieved for {symbol}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market-data", response_model=ApiResponse)
async def get_market_data():
    """Get overall market data (Nifty, Sensex)"""
    try:
        market_data = data_collector.get_market_data()
        return ApiResponse(
            success=True,
            data=market_data,
            message="Market data retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/historical/{symbol}", response_model=ApiResponse)
async def get_historical_data(symbol: str, period: str = "1y"):
    """Get historical price data"""
    try:
        hist_data = data_collector.get_historical_data(symbol, period)
        if hist_data.empty:
            raise HTTPException(status_code=404, detail="No historical data found")
        
        # Convert DataFrame to dict for JSON response
        hist_dict = hist_data.to_dict('records')
        
        return ApiResponse(
            success=True,
            data={
                "symbol": symbol,
                "period": period,
                "data": hist_dict
            },
            message=f"Historical data retrieved for {symbol}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
