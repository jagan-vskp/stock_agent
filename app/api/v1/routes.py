from fastapi import APIRouter
from app.api.v1.endpoints import stocks, analysis, recommendations

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(stocks.router, prefix="/stocks", tags=["stocks"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["analysis"]) 
api_router.include_router(recommendations.router, prefix="/recommendations", tags=["recommendations"])
