import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta
from app.core.cache import cache
from app.core.config import settings

logger = logging.getLogger(__name__)


class DataCollector:
    def __init__(self):
        self.session = requests.Session()
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get basic stock information"""
        cache_key = f"stock_info:{symbol}"
        cached_data = cache.get(cache_key)
        
        if cached_data:
            return cached_data
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            stock_data = {
                "symbol": symbol,
                "name": info.get("longName", "N/A"),
                "exchange": info.get("exchange", "NSE"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap"),
                "current_price": info.get("currentPrice"),
                "currency": info.get("currency", "INR"),
                "country": info.get("country", "India")
            }
            
            cache.set(cache_key, stock_data, ttl=3600)  # Cache for 1 hour
            return stock_data
            
        except Exception as e:
            logger.error(f"Error getting stock info for {symbol}: {e}")
            return {}
    
    def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get historical price data"""
        cache_key = f"historical:{symbol}:{period}"
        cached_data = cache.get(cache_key)
        
        if cached_data:
            return pd.DataFrame(cached_data)
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if not hist.empty:
                hist_dict = hist.to_dict('records')
                cache.set(cache_key, hist_dict, ttl=1800)  # Cache for 30 minutes
                return hist
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
        
        return pd.DataFrame()
    
    def get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """Get fundamental analysis data"""
        cache_key = f"fundamental:{symbol}"
        cached_data = cache.get(cache_key)
        
        if cached_data:
            return cached_data
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            fundamental_data = {
                "pe_ratio": info.get("trailingPE"),
                "eps": info.get("trailingEps"),
                "book_value": info.get("bookValue"),
                "debt_to_equity": info.get("debtToEquity"),
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
                "profit_margin": info.get("profitMargins"),
                "revenue_growth": info.get("revenueGrowth"),
                "dividend_yield": info.get("dividendYield"),
                "market_cap": info.get("marketCap"),
                "beta": info.get("beta"),
                "price_to_book": info.get("priceToBook"),
                "enterprise_value": info.get("enterpriseValue"),
                "forward_pe": info.get("forwardPE")
            }
            
            cache.set(cache_key, fundamental_data, ttl=3600)  # Cache for 1 hour
            return fundamental_data
            
        except Exception as e:
            logger.error(f"Error getting fundamental data for {symbol}: {e}")
            return {}
    
    def get_news_data(self, symbol: str, limit: int = 10) -> list:
        """Get news data for sentiment analysis"""
        cache_key = f"news:{symbol}:{limit}"
        cached_data = cache.get(cache_key)
        
        if cached_data:
            return cached_data
        
        try:
            # Clean symbol for search (remove .NS suffix)
            search_symbol = symbol.replace(".NS", "").replace(".BO", "")
            
            # Get company name for better search
            ticker = yf.Ticker(symbol)
            company_name = ticker.info.get("longName", search_symbol)
            
            # Search for news using requests and BeautifulSoup
            search_query = f"{company_name} stock news"
            news_data = self._scrape_news(search_query, limit)
            
            cache.set(cache_key, news_data, ttl=1800)  # Cache for 30 minutes
            return news_data
            
        except Exception as e:
            logger.error(f"Error getting news data for {symbol}: {e}")
            return []
    
    def _scrape_news(self, query: str, limit: int) -> list:
        """Scrape news from Google News"""
        try:
            # This is a simplified news scraper
            # In production, use proper news APIs like NewsAPI
            news_items = []
            
            # Placeholder news data for demo
            for i in range(min(limit, 5)):
                news_items.append({
                    "title": f"Sample news about {query} {i+1}",
                    "description": f"This is a sample news description for {query}",
                    "url": f"https://example.com/news/{i+1}",
                    "published_at": datetime.now().isoformat(),
                    "source": "Sample News"
                })
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error scraping news: {e}")
            return []
    
    def get_market_data(self) -> Dict[str, Any]:
        """Get overall market data (Nifty, Sensex)"""
        cache_key = "market_data"
        cached_data = cache.get(cache_key)
        
        if cached_data:
            return cached_data
        
        try:
            # Get Nifty 50 data
            nifty = yf.Ticker("^NSEI")
            nifty_info = nifty.info
            
            # Get Sensex data
            sensex = yf.Ticker("^BSESN")
            sensex_info = sensex.info
            
            market_data = {
                "nifty": {
                    "price": nifty_info.get("regularMarketPrice"),
                    "change": nifty_info.get("regularMarketChange"),
                    "change_percent": nifty_info.get("regularMarketChangePercent")
                },
                "sensex": {
                    "price": sensex_info.get("regularMarketPrice"),
                    "change": sensex_info.get("regularMarketChange"),
                    "change_percent": sensex_info.get("regularMarketChangePercent")
                },
                "timestamp": datetime.now().isoformat()
            }
            
            cache.set(cache_key, market_data, ttl=300)  # Cache for 5 minutes
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {}
