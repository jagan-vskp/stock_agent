import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from app.services.data_collector import DataCollector

logger = logging.getLogger(__name__)


class FundamentalAnalyzer:
    def __init__(self):
        self.data_collector = DataCollector()
        
        # Industry average ratios (simplified for demo)
        self.industry_benchmarks = {
            "default": {
                "pe_ratio": 20.0,
                "debt_to_equity": 0.5,
                "roe": 15.0,
                "profit_margin": 10.0
            }
        }
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """Perform comprehensive fundamental analysis"""
        try:
            # Get fundamental data
            fundamental_data = self.data_collector.get_fundamental_data(symbol)
            stock_info = self.data_collector.get_stock_info(symbol)
            
            if not fundamental_data:
                return {"error": "Unable to fetch fundamental data"}
            
            # Calculate scores for different metrics
            scores = self._calculate_scores(fundamental_data)
            
            # Calculate overall fundamental score
            overall_score = self._calculate_overall_score(scores)
            
            # Generate insights
            insights = self._generate_insights(fundamental_data, scores)
            
            analysis_result = {
                "symbol": symbol,
                "fundamental_data": fundamental_data,
                "scores": scores,
                "overall_score": overall_score,
                "insights": insights,
                "recommendation": self._get_recommendation(overall_score),
                "analysis_type": "fundamental"
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis for {symbol}: {e}")
            return {"error": str(e)}
    
    def _calculate_scores(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate scores for various fundamental metrics"""
        scores = {}
        
        # P/E Ratio Score (lower is better, but not too low)
        pe_ratio = data.get("pe_ratio")
        if pe_ratio:
            if 10 <= pe_ratio <= 25:
                scores["pe_score"] = 80
            elif pe_ratio < 10:
                scores["pe_score"] = 60  # Might be undervalued or problematic
            elif pe_ratio <= 35:
                scores["pe_score"] = 60
            else:
                scores["pe_score"] = 30
        else:
            scores["pe_score"] = 50
        
        # ROE Score (higher is better)
        roe = data.get("roe")
        if roe:
            roe_percent = roe * 100 if roe < 1 else roe
            if roe_percent >= 20:
                scores["roe_score"] = 90
            elif roe_percent >= 15:
                scores["roe_score"] = 80
            elif roe_percent >= 10:
                scores["roe_score"] = 60
            else:
                scores["roe_score"] = 40
        else:
            scores["roe_score"] = 50
        
        # Debt to Equity Score (lower is better)
        debt_to_equity = data.get("debt_to_equity")
        if debt_to_equity is not None:
            if debt_to_equity <= 0.3:
                scores["debt_score"] = 90
            elif debt_to_equity <= 0.6:
                scores["debt_score"] = 70
            elif debt_to_equity <= 1.0:
                scores["debt_score"] = 50
            else:
                scores["debt_score"] = 30
        else:
            scores["debt_score"] = 50
        
        # Profit Margin Score (higher is better)
        profit_margin = data.get("profit_margin")
        if profit_margin:
            margin_percent = profit_margin * 100 if profit_margin < 1 else profit_margin
            if margin_percent >= 20:
                scores["margin_score"] = 90
            elif margin_percent >= 15:
                scores["margin_score"] = 80
            elif margin_percent >= 10:
                scores["margin_score"] = 60
            else:
                scores["margin_score"] = 40
        else:
            scores["margin_score"] = 50
        
        # EPS Score (growth trend would be ideal, but using absolute for now)
        eps = data.get("eps")
        if eps and eps > 0:
            if eps >= 50:
                scores["eps_score"] = 80
            elif eps >= 20:
                scores["eps_score"] = 60
            else:
                scores["eps_score"] = 40
        else:
            scores["eps_score"] = 30
        
        return scores
    
    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall fundamental score"""
        weights = {
            "pe_score": 0.25,
            "roe_score": 0.25,
            "debt_score": 0.20,
            "margin_score": 0.20,
            "eps_score": 0.10
        }
        
        total_score = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            if metric in scores:
                total_score += scores[metric] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 50
    
    def _generate_insights(self, data: Dict[str, Any], scores: Dict[str, float]) -> list:
        """Generate human-readable insights"""
        insights = []
        
        # P/E Ratio insights
        pe_ratio = data.get("pe_ratio")
        if pe_ratio:
            if pe_ratio < 10:
                insights.append("Low P/E ratio may indicate undervaluation or potential issues")
            elif pe_ratio > 30:
                insights.append("High P/E ratio suggests high growth expectations or overvaluation")
            else:
                insights.append("P/E ratio is within reasonable range")
        
        # ROE insights
        roe = data.get("roe")
        if roe:
            roe_percent = roe * 100 if roe < 1 else roe
            if roe_percent >= 15:
                insights.append("Strong return on equity indicates efficient management")
            else:
                insights.append("ROE could be improved for better shareholder returns")
        
        # Debt insights
        debt_to_equity = data.get("debt_to_equity")
        if debt_to_equity is not None:
            if debt_to_equity <= 0.3:
                insights.append("Low debt levels indicate strong financial health")
            elif debt_to_equity > 1.0:
                insights.append("High debt levels may pose financial risk")
        
        # Profit margin insights
        profit_margin = data.get("profit_margin")
        if profit_margin:
            margin_percent = profit_margin * 100 if profit_margin < 1 else profit_margin
            if margin_percent >= 15:
                insights.append("Strong profit margins indicate good operational efficiency")
            else:
                insights.append("Profit margins could be improved")
        
        return insights
    
    def _get_recommendation(self, overall_score: float) -> str:
        """Get buy/sell/hold recommendation based on score"""
        if overall_score >= 75:
            return "BUY"
        elif overall_score >= 60:
            return "HOLD"
        else:
            return "SELL"
