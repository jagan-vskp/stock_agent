import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from app.services.fundamental_analyzer import FundamentalAnalyzer
from app.services.technical_analyzer import TechnicalAnalyzer
from app.services.sentiment_analyzer import SentimentAnalyzer
from app.services.ml_predictor import MLTargetPricePredictor
from app.services.data_collector import DataCollector

logger = logging.getLogger(__name__)


class ComprehensiveAnalyzer:
    def __init__(self):
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.ml_predictor = MLTargetPricePredictor()
        self.data_collector = DataCollector()
        
        # Analysis weights for final recommendation
        self.analysis_weights = {
            'fundamental': 0.35,    # Fundamental analysis weight
            'technical': 0.25,      # Technical analysis weight
            'sentiment': 0.20,      # Sentiment analysis weight
            'ml_prediction': 0.20   # ML prediction weight
        }
    
    def analyze_stock(self, symbol: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Perform comprehensive stock analysis"""
        try:
            logger.info(f"Starting comprehensive analysis for {symbol}")
            
            # Get basic stock information
            stock_info = self.data_collector.get_stock_info(symbol)
            if not stock_info:
                return {"error": f"Unable to fetch stock information for {symbol}"}
            
            current_price = stock_info.get('current_price', 0)
            if current_price == 0:
                return {"error": f"Unable to get current price for {symbol}"}
            
            analysis_results = {
                "symbol": symbol,
                "stock_info": stock_info,
                "current_price": current_price,
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_type": analysis_type
            }
            
            # Perform different types of analysis based on request
            if analysis_type in ["fundamental", "comprehensive"]:
                logger.info(f"Performing fundamental analysis for {symbol}")
                fundamental_result = self.fundamental_analyzer.analyze(symbol)
                analysis_results["fundamental"] = fundamental_result
            
            if analysis_type in ["technical", "comprehensive"]:
                logger.info(f"Performing technical analysis for {symbol}")
                technical_result = self.technical_analyzer.analyze(symbol)
                analysis_results["technical"] = technical_result
            
            if analysis_type in ["sentiment", "comprehensive"]:
                logger.info(f"Performing sentiment analysis for {symbol}")
                sentiment_result = self.sentiment_analyzer.analyze(symbol)
                analysis_results["sentiment"] = sentiment_result
            
            if analysis_type in ["comprehensive"]:
                logger.info(f"Performing ML prediction for {symbol}")
                ml_result = self.ml_predictor.predict_target_price(symbol)
                analysis_results["ml_prediction"] = ml_result
            
            # Generate comprehensive insights and recommendation
            if analysis_type == "comprehensive":
                comprehensive_analysis = self._create_comprehensive_analysis(analysis_results)
                analysis_results.update(comprehensive_analysis)
            
            analysis_results["success"] = True
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis for {symbol}: {e}")
            return {"error": str(e), "success": False}
    
    def _create_comprehensive_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive analysis combining all analysis types"""
        try:
            # Extract scores from different analysis types
            scores = self._extract_analysis_scores(analysis_results)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(scores)
            
            # Generate final recommendation
            final_recommendation = self._generate_final_recommendation(scores, analysis_results)
            
            # Create comprehensive insights
            comprehensive_insights = self._generate_comprehensive_insights(analysis_results, scores)
            
            # Calculate target price range
            target_price_analysis = self._calculate_target_price_range(analysis_results)
            
            # Risk assessment
            risk_assessment = self._assess_overall_risk(analysis_results, scores)
            
            # Investment thesis
            investment_thesis = self._create_investment_thesis(
                analysis_results, final_recommendation, comprehensive_insights
            )
            
            return {
                "comprehensive_score": overall_score,
                "individual_scores": scores,
                "final_recommendation": final_recommendation,
                "comprehensive_insights": comprehensive_insights,
                "target_price_analysis": target_price_analysis,
                "risk_assessment": risk_assessment,
                "investment_thesis": investment_thesis,
                "analysis_summary": self._create_analysis_summary(analysis_results, scores)
            }
            
        except Exception as e:
            logger.error(f"Error creating comprehensive analysis: {e}")
            return {"error": "Unable to create comprehensive analysis"}
    
    def _extract_analysis_scores(self, analysis_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract normalized scores from different analysis types"""
        scores = {}
        
        # Fundamental score
        fundamental = analysis_results.get("fundamental", {})
        if fundamental and not fundamental.get("error"):
            scores["fundamental"] = fundamental.get("overall_score", 50) / 100
        
        # Technical score
        technical = analysis_results.get("technical", {})
        if technical and not technical.get("error"):
            scores["technical"] = technical.get("overall_score", 50) / 100
        
        # Sentiment score
        sentiment = analysis_results.get("sentiment", {})
        if sentiment and not sentiment.get("error"):
            overall_sentiment = sentiment.get("overall_sentiment", {})
            scores["sentiment"] = overall_sentiment.get("score", 0.5)
        
        # ML prediction score (based on upside potential)
        ml_prediction = analysis_results.get("ml_prediction", {})
        if ml_prediction and not ml_prediction.get("error"):
            upside_potential = ml_prediction.get("upside_potential_pct", 0)
            confidence = ml_prediction.get("confidence_metrics", {}).get("overall", 0.5)
            
            # Convert upside potential to 0-1 score
            # Positive upside = higher score, negative = lower score
            upside_score = 0.5 + (upside_potential / 100) * 0.5  # Normalize around 0.5
            upside_score = max(0, min(1, upside_score))  # Clamp to 0-1
            
            # Weight by confidence
            scores["ml_prediction"] = upside_score * confidence + 0.5 * (1 - confidence)
        
        return scores
    
    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall score"""
        if not scores:
            return 0.5
        
        total_score = 0
        total_weight = 0
        
        for analysis_type, score in scores.items():
            if analysis_type in self.analysis_weights:
                weight = self.analysis_weights[analysis_type]
                total_score += score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    def _generate_final_recommendation(self, scores: Dict[str, float], analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final recommendation based on all analysis types"""
        overall_score = self._calculate_overall_score(scores)
        
        # Count individual recommendations
        recommendations = {}
        for analysis_type in ["fundamental", "technical", "sentiment", "ml_prediction"]:
            analysis_data = analysis_results.get(analysis_type, {})
            if analysis_data and not analysis_data.get("error"):
                rec = analysis_data.get("recommendation", "HOLD")
                recommendations[analysis_type] = rec
        
        # Count votes
        buy_votes = sum(1 for rec in recommendations.values() if rec == "BUY")
        sell_votes = sum(1 for rec in recommendations.values() if rec == "SELL")
        hold_votes = sum(1 for rec in recommendations.values() if rec == "HOLD")
        
        # Determine final recommendation
        if overall_score >= 0.65 and buy_votes >= sell_votes:
            final_rec = "BUY"
        elif overall_score <= 0.35 and sell_votes >= buy_votes:
            final_rec = "SELL"
        else:
            final_rec = "HOLD"
        
        # Calculate confidence based on consensus
        total_votes = len(recommendations)
        if total_votes > 0:
            if final_rec == "BUY":
                confidence = (buy_votes / total_votes) * overall_score
            elif final_rec == "SELL":
                confidence = (sell_votes / total_votes) * (1 - overall_score)
            else:
                confidence = (hold_votes / total_votes) * 0.7  # Medium confidence for hold
        else:
            confidence = 0.5
        
        return {
            "recommendation": final_rec,
            "confidence": confidence,
            "overall_score": overall_score,
            "individual_recommendations": recommendations,
            "vote_breakdown": {
                "BUY": buy_votes,
                "SELL": sell_votes,
                "HOLD": hold_votes
            }
        }
    
    def _generate_comprehensive_insights(self, analysis_results: Dict[str, Any], scores: Dict[str, float]) -> List[str]:
        """Generate comprehensive insights combining all analysis types"""
        insights = []
        
        # Overall score insight
        overall_score = self._calculate_overall_score(scores)
        if overall_score > 0.7:
            insights.append("Strong positive signals across multiple analysis dimensions")
        elif overall_score < 0.3:
            insights.append("Multiple analysis methods indicate concerns")
        else:
            insights.append("Mixed signals requiring careful consideration")
        
        # Score breakdown insights
        if scores:
            highest_score_type = max(scores.items(), key=lambda x: x[1])
            lowest_score_type = min(scores.items(), key=lambda x: x[1])
            
            if highest_score_type[1] > 0.7:
                insights.append(f"{highest_score_type[0].title()} analysis shows particularly strong signals")
            
            if lowest_score_type[1] < 0.3:
                insights.append(f"{lowest_score_type[0].title()} analysis raises concerns")
        
        # Specific insights from each analysis type
        for analysis_type in ["fundamental", "technical", "sentiment", "ml_prediction"]:
            analysis_data = analysis_results.get(analysis_type, {})
            if analysis_data and not analysis_data.get("error"):
                type_insights = analysis_data.get("insights", [])
                if type_insights:
                    # Add top 2 insights from each analysis type
                    for insight in type_insights[:2]:
                        insights.append(f"{analysis_type.title()}: {insight}")
        
        # Consensus insights
        recommendations = {}
        for analysis_type in ["fundamental", "technical", "sentiment", "ml_prediction"]:
            analysis_data = analysis_results.get(analysis_type, {})
            if analysis_data and not analysis_data.get("error"):
                recommendations[analysis_type] = analysis_data.get("recommendation", "HOLD")
        
        if recommendations:
            unique_recs = set(recommendations.values())
            if len(unique_recs) == 1:
                insights.append(f"All analysis methods agree on {list(unique_recs)[0]} recommendation")
            elif len(unique_recs) == 3:
                insights.append("Analysis methods show divergent views - requires careful evaluation")
        
        return insights[:10]  # Limit to top 10 insights
    
    def _calculate_target_price_range(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate target price range from different analysis methods"""
        current_price = analysis_results.get("current_price", 0)
        target_prices = []
        
        # ML prediction targets
        ml_prediction = analysis_results.get("ml_prediction", {})
        if ml_prediction and not ml_prediction.get("error"):
            ml_targets = ml_prediction.get("price_targets", {})
            if ml_targets:
                target_prices.extend([
                    ml_targets.get("conservative", current_price),
                    ml_targets.get("base", current_price),
                    ml_targets.get("optimistic", current_price)
                ])
        
        # Fundamental-based target (simplified)
        fundamental = analysis_results.get("fundamental", {})
        if fundamental and not fundamental.get("error"):
            fund_score = fundamental.get("overall_score", 50)
            if fund_score > 75:
                fund_target = current_price * 1.15  # 15% upside for strong fundamentals
            elif fund_score > 60:
                fund_target = current_price * 1.05  # 5% upside
            else:
                fund_target = current_price * 0.95  # 5% downside
            target_prices.append(fund_target)
        
        # Technical-based target (simplified)
        technical = analysis_results.get("technical", {})
        if technical and not technical.get("error"):
            indicators = technical.get("indicators", {})
            resistance = indicators.get("resistance", current_price * 1.1)
            support = indicators.get("support", current_price * 0.9)
            
            tech_score = technical.get("overall_score", 50)
            if tech_score > 70:
                tech_target = min(resistance, current_price * 1.1)
            elif tech_score < 40:
                tech_target = max(support, current_price * 0.9)
            else:
                tech_target = current_price
            
            target_prices.append(tech_target)
        
        if target_prices:
            return {
                "min_target": min(target_prices),
                "max_target": max(target_prices),
                "average_target": np.mean(target_prices),
                "median_target": np.median(target_prices),
                "upside_potential_pct": (np.mean(target_prices) - current_price) / current_price * 100,
                "target_range_pct": (max(target_prices) - min(target_prices)) / current_price * 100
            }
        else:
            return {
                "min_target": current_price,
                "max_target": current_price,
                "average_target": current_price,
                "median_target": current_price,
                "upside_potential_pct": 0,
                "target_range_pct": 0
            }
    
    def _assess_overall_risk(self, analysis_results: Dict[str, Any], scores: Dict[str, float]) -> Dict[str, Any]:
        """Assess overall investment risk"""
        risk_factors = []
        risk_score = 0.5  # Neutral risk
        
        # Fundamental risks
        fundamental = analysis_results.get("fundamental", {})
        if fundamental and not fundamental.get("error"):
            fund_data = fundamental.get("fundamental_data", {})
            
            debt_to_equity = fund_data.get("debt_to_equity", 0.5)
            if debt_to_equity > 1.0:
                risk_factors.append("High debt levels")
                risk_score += 0.1
            
            pe_ratio = fund_data.get("pe_ratio", 20)
            if pe_ratio > 40:
                risk_factors.append("High valuation risk")
                risk_score += 0.1
        
        # Technical risks
        technical = analysis_results.get("technical", {})
        if technical and not technical.get("error"):
            indicators = technical.get("indicators", {})
            
            volatility = indicators.get("atr", 10)
            current_price = analysis_results.get("current_price", 100)
            volatility_pct = (volatility / current_price) * 100
            
            if volatility_pct > 5:
                risk_factors.append("High price volatility")
                risk_score += 0.1
        
        # Sentiment risks
        sentiment = analysis_results.get("sentiment", {})
        if sentiment and not sentiment.get("error"):
            overall_sentiment = sentiment.get("overall_sentiment", {})
            sentiment_score = overall_sentiment.get("score", 0.5)
            
            if sentiment_score < 0.3:
                risk_factors.append("Negative market sentiment")
                risk_score += 0.1
        
        # ML prediction risks
        ml_prediction = analysis_results.get("ml_prediction", {})
        if ml_prediction and not ml_prediction.get("error"):
            ml_risk = ml_prediction.get("risk_level", "Medium")
            confidence = ml_prediction.get("confidence_metrics", {}).get("overall", 0.5)
            
            if ml_risk == "High" or confidence < 0.4:
                risk_factors.append("High prediction uncertainty")
                risk_score += 0.1
        
        # Score dispersion risk
        if scores:
            score_std = np.std(list(scores.values()))
            if score_std > 0.2:
                risk_factors.append("Conflicting analysis signals")
                risk_score += 0.1
        
        # Determine risk level
        if risk_score <= 0.4:
            risk_level = "Low"
        elif risk_score <= 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            "risk_level": risk_level,
            "risk_score": min(1.0, risk_score),
            "risk_factors": risk_factors,
            "recommendation": self._get_risk_recommendation(risk_level, risk_factors)
        }
    
    def _get_risk_recommendation(self, risk_level: str, risk_factors: List[str]) -> str:
        """Get risk-based recommendation"""
        if risk_level == "Low":
            return "Suitable for most investors"
        elif risk_level == "Medium":
            return "Suitable for moderate risk tolerance investors"
        else:
            return "Only suitable for high risk tolerance investors"
    
    def _create_investment_thesis(self, analysis_results: Dict[str, Any], 
                                final_recommendation: Dict[str, Any], 
                                insights: List[str]) -> Dict[str, Any]:
        """Create investment thesis"""
        symbol = analysis_results.get("symbol", "")
        stock_info = analysis_results.get("stock_info", {})
        recommendation = final_recommendation.get("recommendation", "HOLD")
        confidence = final_recommendation.get("confidence", 0.5)
        
        # Create thesis summary
        if recommendation == "BUY":
            thesis_summary = f"Strong investment opportunity in {symbol} with multiple positive signals"
        elif recommendation == "SELL":
            thesis_summary = f"Concerns identified in {symbol} suggest reducing exposure"
        else:
            thesis_summary = f"Mixed signals for {symbol} suggest maintaining current position"
        
        # Key strengths and weaknesses
        strengths = []
        weaknesses = []
        
        # Extract from insights
        for insight in insights:
            if any(word in insight.lower() for word in ["strong", "positive", "good", "high", "growth", "bullish"]):
                strengths.append(insight)
            elif any(word in insight.lower() for word in ["weak", "negative", "concern", "risk", "decline", "bearish"]):
                weaknesses.append(insight)
        
        return {
            "summary": thesis_summary,
            "recommendation": recommendation,
            "confidence_level": "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low",
            "key_strengths": strengths[:5],  # Top 5 strengths
            "key_concerns": weaknesses[:5],  # Top 5 concerns
            "sector": stock_info.get("sector", "N/A"),
            "market_cap_category": self._get_market_cap_category(stock_info.get("market_cap", 0))
        }
    
    def _get_market_cap_category(self, market_cap: float) -> str:
        """Categorize stock by market cap"""
        if market_cap > 200000000000:  # 2 lakh crore
            return "Large Cap"
        elif market_cap > 50000000000:  # 50,000 crore
            return "Mid Cap"
        else:
            return "Small Cap"
    
    def _create_analysis_summary(self, analysis_results: Dict[str, Any], scores: Dict[str, float]) -> Dict[str, str]:
        """Create summary of each analysis type"""
        summary = {}
        
        for analysis_type in ["fundamental", "technical", "sentiment", "ml_prediction"]:
            analysis_data = analysis_results.get(analysis_type, {})
            
            if analysis_data and not analysis_data.get("error"):
                score = scores.get(analysis_type, 0.5) * 100
                recommendation = analysis_data.get("recommendation", "HOLD")
                
                if analysis_type == "fundamental":
                    summary[analysis_type] = f"Score: {score:.0f}/100, Recommendation: {recommendation}"
                elif analysis_type == "technical":
                    summary[analysis_type] = f"Score: {score:.0f}/100, Signals: {recommendation}"
                elif analysis_type == "sentiment":
                    sentiment_interp = analysis_data.get("overall_sentiment", {}).get("interpretation", "Neutral")
                    summary[analysis_type] = f"Sentiment: {sentiment_interp}, Recommendation: {recommendation}"
                elif analysis_type == "ml_prediction":
                    upside = analysis_data.get("upside_potential_pct", 0)
                    summary[analysis_type] = f"Predicted upside: {upside:+.1f}%, Recommendation: {recommendation}"
            else:
                summary[analysis_type] = "Analysis not available"
        
        return summary
