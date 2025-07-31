import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
import re
from datetime import datetime, timedelta
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from app.services.data_collector import DataCollector
from app.core.cache import cache

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    def __init__(self):
        self.data_collector = DataCollector()
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Financial keywords for enhanced sentiment analysis
        self.positive_financial_keywords = [
            'profit', 'growth', 'increase', 'rise', 'gain', 'boost', 'strong', 'positive',
            'bullish', 'outperform', 'upgrade', 'buy', 'recommend', 'expansion', 'revenue',
            'earnings', 'dividend', 'acquisition', 'merger', 'partnership', 'breakthrough',
            'recovery', 'rebound', 'rally', 'momentum', 'optimistic', 'beat', 'exceed'
        ]
        
        self.negative_financial_keywords = [
            'loss', 'decline', 'fall', 'drop', 'weak', 'negative', 'bearish', 'underperform',
            'downgrade', 'sell', 'concern', 'risk', 'crisis', 'recession', 'debt', 'lawsuit',
            'investigation', 'scandal', 'fraud', 'bankruptcy', 'layoff', 'closure', 'miss',
            'disappoint', 'warning', 'caution', 'volatile', 'uncertainty', 'crash'
        ]
        
        # Market sentiment indicators
        self.market_indicators = {
            'vix': {'threshold': 20, 'interpretation': 'volatility'},
            'nifty_change': {'threshold': 1, 'interpretation': 'market_trend'},
            'sector_rotation': {'threshold': 0.5, 'interpretation': 'sector_strength'}
        }
    
    def analyze(self, symbol: str, days_back: int = 30) -> Dict[str, Any]:
        """Perform comprehensive sentiment analysis"""
        try:
            # Get news data
            news_data = self.data_collector.get_news_data(symbol, limit=50)
            
            if not news_data:
                return {"error": "No news data available for sentiment analysis"}
            
            # Filter recent news
            recent_news = self._filter_recent_news(news_data, days_back)
            
            # Analyze news sentiment
            news_sentiment = self._analyze_news_sentiment(recent_news)
            
            # Get social media sentiment (simulated)
            social_sentiment = self._get_social_sentiment(symbol)
            
            # Get market sentiment
            market_sentiment = self._get_market_sentiment()
            
            # Calculate overall sentiment score
            overall_sentiment = self._calculate_overall_sentiment(
                news_sentiment, social_sentiment, market_sentiment
            )
            
            # Generate sentiment insights
            insights = self._generate_sentiment_insights(
                news_sentiment, social_sentiment, market_sentiment, recent_news
            )
            
            # Get sentiment-based recommendation
            recommendation = self._get_sentiment_recommendation(overall_sentiment)
            
            return {
                "symbol": symbol,
                "news_sentiment": news_sentiment,
                "social_sentiment": social_sentiment,
                "market_sentiment": market_sentiment,
                "overall_sentiment": overall_sentiment,
                "news_count": len(recent_news),
                "insights": insights,
                "recommendation": recommendation,
                "analysis_type": "sentiment",
                "analysis_period_days": days_back
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis for {symbol}: {e}")
            return {"error": str(e)}
    
    def _filter_recent_news(self, news_data: List[Dict], days_back: int) -> List[Dict]:
        """Filter news from recent days"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_news = []
        
        for article in news_data:
            try:
                # Try to parse published_at date
                pub_date = datetime.fromisoformat(article.get('published_at', '').replace('Z', '+00:00'))
                if pub_date >= cutoff_date:
                    recent_news.append(article)
            except:
                # If date parsing fails, include the article anyway
                recent_news.append(article)
        
        return recent_news
    
    def _analyze_news_sentiment(self, news_data: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment from news articles"""
        if not news_data:
            return {"score": 0.5, "confidence": 0, "details": []}
        
        sentiments = []
        article_details = []
        
        for article in news_data:
            title = article.get('title', '')
            description = article.get('description', '')
            text = f"{title} {description}"
            
            if not text.strip():
                continue
            
            # Clean text
            clean_text = self._clean_text(text)
            
            # Multiple sentiment analysis approaches
            blob_sentiment = TextBlob(clean_text).sentiment.polarity
            vader_sentiment = self.vader_analyzer.polarity_scores(clean_text)['compound']
            
            # Enhanced financial sentiment
            financial_sentiment = self._calculate_financial_sentiment(clean_text)
            
            # Weighted average of different approaches
            combined_sentiment = (
                blob_sentiment * 0.3 + 
                vader_sentiment * 0.4 + 
                financial_sentiment * 0.3
            )
            
            sentiments.append(combined_sentiment)
            
            article_details.append({
                "title": title,
                "sentiment": combined_sentiment,
                "confidence": abs(combined_sentiment),
                "source": article.get('source', 'Unknown')
            })
        
        if not sentiments:
            return {"score": 0.5, "confidence": 0, "details": []}
        
        # Calculate overall news sentiment
        avg_sentiment = np.mean(sentiments)
        sentiment_std = np.std(sentiments)
        confidence = min(len(sentiments) / 10, 1.0) * (1 - min(sentiment_std, 1.0))
        
        # Normalize to 0-1 scale
        normalized_score = (avg_sentiment + 1) / 2
        
        return {
            "score": normalized_score,
            "confidence": confidence,
            "article_count": len(sentiments),
            "positive_articles": len([s for s in sentiments if s > 0.1]),
            "negative_articles": len([s for s in sentiments if s < -0.1]),
            "neutral_articles": len([s for s in sentiments if -0.1 <= s <= 0.1]),
            "details": article_details[:10]  # Top 10 articles
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove HTML tags
        text = re.sub('<[^<]+?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.lower()
    
    def _calculate_financial_sentiment(self, text: str) -> float:
        """Calculate sentiment based on financial keywords"""
        words = text.lower().split()
        
        positive_count = sum(1 for word in words if any(keyword in word for keyword in self.positive_financial_keywords))
        negative_count = sum(1 for word in words if any(keyword in word for keyword in self.negative_financial_keywords))
        
        total_keywords = positive_count + negative_count
        
        if total_keywords == 0:
            return 0.0
        
        # Calculate sentiment score
        sentiment_score = (positive_count - negative_count) / total_keywords
        
        # Apply sigmoid function to normalize
        return np.tanh(sentiment_score)
    
    def _get_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get social media sentiment (simulated for demo)"""
        # In production, integrate with Twitter API, Reddit API, etc.
        cache_key = f"social_sentiment:{symbol}"
        cached_data = cache.get(cache_key)
        
        if cached_data:
            return cached_data
        
        # Simulated social sentiment with some realistic variation
        base_sentiment = 0.5 + np.random.normal(0, 0.15)
        base_sentiment = max(0, min(1, base_sentiment))  # Clamp to 0-1
        
        social_data = {
            "score": base_sentiment,
            "confidence": 0.6,
            "twitter_mentions": np.random.randint(50, 500),
            "reddit_discussions": np.random.randint(5, 50),
            "sentiment_trend": "neutral",
            "engagement_score": 0.7
        }
        
        # Determine trend
        if base_sentiment > 0.6:
            social_data["sentiment_trend"] = "positive"
        elif base_sentiment < 0.4:
            social_data["sentiment_trend"] = "negative"
        
        cache.set(cache_key, social_data, ttl=1800)  # Cache for 30 minutes
        return social_data
    
    def _get_market_sentiment(self) -> Dict[str, Any]:
        """Get overall market sentiment"""
        try:
            # Get market data
            market_data = self.data_collector.get_market_data()
            
            sentiment_score = 0.5  # Neutral base
            factors = []
            
            # Analyze Nifty trend
            nifty_data = market_data.get('nifty', {})
            nifty_change_pct = nifty_data.get('change_percent', 0)
            
            if nifty_change_pct > 1:
                sentiment_score += 0.2
                factors.append("Strong positive market trend")
            elif nifty_change_pct > 0:
                sentiment_score += 0.1
                factors.append("Positive market trend")
            elif nifty_change_pct < -1:
                sentiment_score -= 0.2
                factors.append("Negative market trend")
            elif nifty_change_pct < 0:
                sentiment_score -= 0.1
                factors.append("Mild negative market trend")
            
            # Simulate VIX-like volatility indicator
            volatility = abs(nifty_change_pct) + np.random.uniform(0, 0.5)
            if volatility > 2:
                sentiment_score -= 0.1
                factors.append("High market volatility")
            elif volatility < 0.5:
                sentiment_score += 0.05
                factors.append("Low market volatility")
            
            # Clamp to valid range
            sentiment_score = max(0, min(1, sentiment_score))
            
            return {
                "score": sentiment_score,
                "volatility": volatility,
                "market_trend": "positive" if nifty_change_pct > 0 else "negative",
                "factors": factors,
                "confidence": 0.8
            }
            
        except Exception as e:
            logger.error(f"Error getting market sentiment: {e}")
            return {
                "score": 0.5,
                "volatility": 1.0,
                "market_trend": "neutral",
                "factors": ["Market data unavailable"],
                "confidence": 0.3
            }
    
    def _calculate_overall_sentiment(self, news_sentiment: Dict, social_sentiment: Dict, market_sentiment: Dict) -> Dict[str, Any]:
        """Calculate weighted overall sentiment score"""
        # Weights for different sentiment sources
        weights = {
            'news': 0.5,      # News has highest impact
            'social': 0.3,    # Social media sentiment
            'market': 0.2     # Overall market sentiment
        }
        
        # Extract scores
        news_score = news_sentiment.get('score', 0.5)
        social_score = social_sentiment.get('score', 0.5)
        market_score = market_sentiment.get('score', 0.5)
        
        # Calculate weighted average
        overall_score = (
            news_score * weights['news'] +
            social_score * weights['social'] +
            market_score * weights['market']
        )
        
        # Calculate confidence based on data availability and consistency
        confidence_factors = [
            news_sentiment.get('confidence', 0),
            social_sentiment.get('confidence', 0),
            market_sentiment.get('confidence', 0)
        ]
        
        overall_confidence = np.mean(confidence_factors)
        
        return {
            "score": overall_score,
            "confidence": overall_confidence,
            "interpretation": self._interpret_sentiment(overall_score),
            "strength": self._get_sentiment_strength(overall_score)
        }
    
    def _interpret_sentiment(self, score: float) -> str:
        """Convert sentiment score to human-readable interpretation"""
        if score >= 0.7:
            return "Very Positive"
        elif score >= 0.6:
            return "Positive"
        elif score >= 0.4:
            return "Neutral"
        elif score >= 0.3:
            return "Negative"
        else:
            return "Very Negative"
    
    def _get_sentiment_strength(self, score: float) -> str:
        """Get sentiment strength"""
        deviation = abs(score - 0.5)
        if deviation >= 0.3:
            return "Strong"
        elif deviation >= 0.15:
            return "Moderate"
        else:
            return "Weak"
    
    def _generate_sentiment_insights(self, news_sentiment: Dict, social_sentiment: Dict, market_sentiment: Dict, recent_news: List) -> List[str]:
        """Generate sentiment-based insights"""
        insights = []
        
        # News sentiment insights
        news_score = news_sentiment.get('score', 0.5)
        if news_score > 0.7:
            insights.append("Strong positive news sentiment indicates favorable market perception")
        elif news_score < 0.3:
            insights.append("Negative news sentiment suggests concerns about the stock")
        
        # Article balance insights
        positive_articles = news_sentiment.get('positive_articles', 0)
        negative_articles = news_sentiment.get('negative_articles', 0)
        total_articles = news_sentiment.get('article_count', 0)
        
        if total_articles > 0:
            positive_ratio = positive_articles / total_articles
            if positive_ratio > 0.7:
                insights.append(f"{positive_ratio:.0%} of recent articles are positive")
            elif positive_ratio < 0.3:
                insights.append(f"Only {positive_ratio:.0%} of recent articles are positive")
        
        # Social sentiment insights
        social_score = social_sentiment.get('score', 0.5)
        if social_score > 0.6:
            insights.append("Social media sentiment is bullish on this stock")
        elif social_score < 0.4:
            insights.append("Social media sentiment shows bearish views")
        
        # Market sentiment insights
        market_trend = market_sentiment.get('market_trend', 'neutral')
        if market_trend == 'positive':
            insights.append("Overall market sentiment is supportive")
        elif market_trend == 'negative':
            insights.append("Negative market sentiment may impact stock performance")
        
        # Combined insights
        overall_score = (news_score + social_score + market_sentiment.get('score', 0.5)) / 3
        if overall_score > 0.6 and all(s > 0.5 for s in [news_score, social_score]):
            insights.append("All sentiment sources show positive alignment")
        elif overall_score < 0.4:
            insights.append("Multiple sentiment sources indicate caution")
        
        return insights
    
    def _get_sentiment_recommendation(self, overall_sentiment: Dict) -> str:
        """Get recommendation based on sentiment analysis"""
        score = overall_sentiment.get('score', 0.5)
        confidence = overall_sentiment.get('confidence', 0)
        
        # Require minimum confidence for strong recommendations
        if confidence < 0.3:
            return "HOLD"  # Low confidence = hold
        
        if score >= 0.65:
            return "BUY"
        elif score <= 0.35:
            return "SELL"
        else:
            return "HOLD"
