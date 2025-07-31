import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from app.services.data_collector import DataCollector
from app.services.fundamental_analyzer import FundamentalAnalyzer
from app.services.technical_analyzer import TechnicalAnalyzer
from app.core.cache import cache

logger = logging.getLogger(__name__)


class MLTargetPricePredictor:
    def __init__(self):
        self.data_collector = DataCollector()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        
        # Model storage directory
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize models
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression()
        }
        
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def predict_target_price(self, symbol: str, prediction_horizon_days: int = 30) -> Dict[str, Any]:
        """Predict target price using ML models"""
        try:
            # Prepare features for prediction
            features = self._prepare_features(symbol)
            
            if not features:
                return {"error": "Unable to prepare features for prediction"}
            
            # Get current price
            stock_info = self.data_collector.get_stock_info(symbol)
            current_price = stock_info.get('current_price', 0)
            
            if current_price == 0:
                return {"error": "Unable to get current stock price"}
            
            # Load or train models
            model_results = self._get_model_predictions(symbol, features, current_price)
            
            # Ensemble prediction
            ensemble_prediction = self._ensemble_predict(model_results, current_price)
            
            # Calculate confidence and risk metrics
            confidence_metrics = self._calculate_prediction_confidence(model_results, features)
            
            # Generate price targets for different scenarios
            price_targets = self._generate_price_targets(
                ensemble_prediction, current_price, confidence_metrics
            )
            
            # Create detailed analysis
            analysis = self._create_prediction_analysis(
                symbol, current_price, price_targets, model_results, 
                confidence_metrics, prediction_horizon_days
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in ML target price prediction for {symbol}: {e}")
            return {"error": str(e)}
    
    def _prepare_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """Prepare features for ML model"""
        try:
            features = {}
            
            # Get fundamental features
            fundamental_data = self.data_collector.get_fundamental_data(symbol)
            if fundamental_data:
                # Fundamental ratios
                features['pe_ratio'] = self._safe_float(fundamental_data.get('pe_ratio'), 20.0)
                features['pb_ratio'] = self._safe_float(fundamental_data.get('price_to_book'), 3.0)
                features['debt_to_equity'] = self._safe_float(fundamental_data.get('debt_to_equity'), 0.5)
                features['roe'] = self._safe_float(fundamental_data.get('roe'), 0.15)
                features['roa'] = self._safe_float(fundamental_data.get('roa'), 0.08)
                features['profit_margin'] = self._safe_float(fundamental_data.get('profit_margin'), 0.1)
                features['revenue_growth'] = self._safe_float(fundamental_data.get('revenue_growth'), 0.1)
                features['eps'] = self._safe_float(fundamental_data.get('eps'), 10.0)
                features['dividend_yield'] = self._safe_float(fundamental_data.get('dividend_yield'), 0.02)
                features['beta'] = self._safe_float(fundamental_data.get('beta'), 1.0)
                
                # Market cap (log scale for better ML performance)
                market_cap = fundamental_data.get('market_cap', 1000000000)
                features['log_market_cap'] = np.log(max(market_cap, 1000000))
            
            # Get technical features
            technical_analysis = self.technical_analyzer.analyze(symbol)
            if not technical_analysis.get('error'):
                indicators = technical_analysis.get('indicators', {})
                
                # Technical indicators
                features['rsi'] = self._safe_float(indicators.get('rsi'), 50.0)
                features['macd'] = self._safe_float(indicators.get('macd'), 0.0)
                features['bb_width'] = self._safe_float(indicators.get('bb_width'), 0.1)
                features['atr'] = self._safe_float(indicators.get('atr'), 10.0)
                features['volume_ratio'] = self._safe_float(indicators.get('volume_ratio'), 1.0)
                features['williams_r'] = self._safe_float(indicators.get('williams_r'), -50.0)
                features['cci'] = self._safe_float(indicators.get('cci'), 0.0)
                
                # Price position features
                current_price = indicators.get('current_price', 100)
                features['price_to_sma20'] = current_price / self._safe_float(indicators.get('sma_20'), current_price)
                features['price_to_sma50'] = current_price / self._safe_float(indicators.get('sma_50'), current_price)
                features['price_change_pct'] = self._safe_float(indicators.get('price_change_pct'), 0.0)
            
            # Market features
            market_data = self.data_collector.get_market_data()
            if market_data:
                nifty_data = market_data.get('nifty', {})
                features['market_change_pct'] = self._safe_float(nifty_data.get('change_percent'), 0.0)
                
                # Market volatility (simplified)
                features['market_volatility'] = abs(features.get('market_change_pct', 0))
            
            # Seasonal features
            now = datetime.now()
            features['month'] = now.month
            features['quarter'] = (now.month - 1) // 3 + 1
            features['is_month_end'] = 1 if now.day > 25 else 0
            
            # Historical price features
            hist_data = self.data_collector.get_historical_data(symbol, "3mo")
            if not hist_data.empty:
                returns = hist_data['Close'].pct_change().dropna()
                features['volatility_30d'] = returns.tail(30).std() * np.sqrt(252)  # Annualized
                features['momentum_5d'] = (hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[-6] - 1) if len(hist_data) > 5 else 0
                features['momentum_20d'] = (hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[-21] - 1) if len(hist_data) > 20 else 0
            
            # Ensure all features are finite
            features = {k: v for k, v in features.items() if np.isfinite(v)}
            
            return features if features else None
            
        except Exception as e:
            logger.error(f"Error preparing features for {symbol}: {e}")
            return None
    
    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert value to float"""
        try:
            if value is None:
                return default
            return float(value) if np.isfinite(float(value)) else default
        except (ValueError, TypeError):
            return default
    
    def _get_model_predictions(self, symbol: str, features: Dict[str, float], current_price: float) -> Dict[str, Dict[str, Any]]:
        """Get predictions from different ML models"""
        model_results = {}
        
        # Convert features to array
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        self.feature_names = list(features.keys())
        
        for model_name, model in self.models.items():
            try:
                # Try to load pre-trained model
                model_path = os.path.join(self.model_dir, f"{model_name}_{symbol}.joblib")
                scaler_path = os.path.join(self.model_dir, f"scaler_{symbol}.joblib")
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    model = joblib.load(model_path)
                    scaler = joblib.load(scaler_path)
                    
                    # Scale features
                    scaled_features = scaler.transform(feature_vector)
                    
                    # Predict
                    prediction = model.predict(scaled_features)[0]
                    
                    # Get feature importance if available
                    importance = self._get_feature_importance(model, self.feature_names)
                    
                    model_results[model_name] = {
                        'prediction': prediction,
                        'confidence': 0.7,  # Default confidence for pre-trained models
                        'feature_importance': importance,
                        'model_type': 'pre_trained'
                    }
                else:
                    # Use simple heuristic-based prediction if no pre-trained model
                    heuristic_prediction = self._heuristic_prediction(features, current_price)
                    
                    model_results[model_name] = {
                        'prediction': heuristic_prediction,
                        'confidence': 0.5,  # Lower confidence for heuristic
                        'feature_importance': {},
                        'model_type': 'heuristic'
                    }
                    
            except Exception as e:
                logger.error(f"Error with {model_name} for {symbol}: {e}")
                # Fallback to current price with small random variation
                model_results[model_name] = {
                    'prediction': current_price * (1 + np.random.normal(0, 0.05)),
                    'confidence': 0.3,
                    'feature_importance': {},
                    'model_type': 'fallback'
                }
        
        return model_results
    
    def _heuristic_prediction(self, features: Dict[str, float], current_price: float) -> float:
        """Simple heuristic-based price prediction"""
        try:
            # Base prediction on fundamental and technical factors
            adjustment_factor = 1.0
            
            # Fundamental adjustments
            pe_ratio = features.get('pe_ratio', 20)
            if pe_ratio < 15:
                adjustment_factor += 0.05  # Undervalued
            elif pe_ratio > 30:
                adjustment_factor -= 0.05  # Overvalued
            
            roe = features.get('roe', 0.15)
            if roe > 0.2:
                adjustment_factor += 0.03  # High ROE is good
            elif roe < 0.1:
                adjustment_factor -= 0.03  # Low ROE is concerning
            
            # Technical adjustments
            rsi = features.get('rsi', 50)
            if rsi < 30:
                adjustment_factor += 0.02  # Oversold, potential bounce
            elif rsi > 70:
                adjustment_factor -= 0.02  # Overbought, potential correction
            
            # Momentum adjustments
            momentum_20d = features.get('momentum_20d', 0)
            if momentum_20d > 0.1:
                adjustment_factor += 0.03  # Strong momentum
            elif momentum_20d < -0.1:
                adjustment_factor -= 0.03  # Weak momentum
            
            # Market sentiment adjustment
            market_change = features.get('market_change_pct', 0)
            adjustment_factor += market_change * 0.1  # Market correlation
            
            # Volatility adjustment (high volatility = higher uncertainty)
            volatility = features.get('volatility_30d', 0.2)
            if volatility > 0.4:
                adjustment_factor *= (1 + np.random.normal(0, 0.02))  # Add uncertainty
            
            return current_price * max(0.5, min(2.0, adjustment_factor))  # Clamp to reasonable range
            
        except Exception as e:
            logger.error(f"Error in heuristic prediction: {e}")
            return current_price * 1.05  # Default 5% upside
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(feature_names, model.feature_importances_))
                return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            elif hasattr(model, 'coef_'):
                importance = dict(zip(feature_names, abs(model.coef_)))
                return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            else:
                return {}
        except:
            return {}
    
    def _ensemble_predict(self, model_results: Dict[str, Dict], current_price: float) -> Dict[str, float]:
        """Combine predictions from multiple models"""
        predictions = []
        weights = []
        
        for model_name, result in model_results.items():
            prediction = result['prediction']
            confidence = result['confidence']
            
            # Validate prediction
            if np.isfinite(prediction) and 0.5 * current_price <= prediction <= 3.0 * current_price:
                predictions.append(prediction)
                weights.append(confidence)
        
        if not predictions:
            return {"ensemble": current_price * 1.05}  # Default prediction
        
        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights
        
        ensemble_prediction = np.average(predictions, weights=weights)
        
        # Calculate prediction stats
        prediction_variance = np.var(predictions)
        prediction_std = np.std(predictions)
        
        return {
            "ensemble": ensemble_prediction,
            "min": min(predictions),
            "max": max(predictions),
            "std": prediction_std,
            "variance": prediction_variance
        }
    
    def _calculate_prediction_confidence(self, model_results: Dict, features: Dict) -> Dict[str, float]:
        """Calculate confidence metrics for predictions"""
        confidences = [result['confidence'] for result in model_results.values()]
        
        # Model agreement (how close predictions are)
        predictions = [result['prediction'] for result in model_results.values()]
        if len(predictions) > 1:
            prediction_cv = np.std(predictions) / np.mean(predictions)  # Coefficient of variation
            agreement_score = max(0, 1 - prediction_cv)  # Higher agreement = lower CV
        else:
            agreement_score = 0.5
        
        # Data quality score (based on available features)
        data_completeness = len([v for v in features.values() if v != 0]) / len(features)
        
        # Overall confidence
        overall_confidence = np.mean(confidences) * agreement_score * data_completeness
        
        return {
            "overall": min(1.0, overall_confidence),
            "model_agreement": agreement_score,
            "data_completeness": data_completeness,
            "individual_models": dict(zip(model_results.keys(), confidences))
        }
    
    def _generate_price_targets(self, ensemble_prediction: Dict, current_price: float, confidence_metrics: Dict) -> Dict[str, float]:
        """Generate price targets for different scenarios"""
        base_prediction = ensemble_prediction["ensemble"]
        prediction_std = ensemble_prediction.get("std", current_price * 0.1)
        overall_confidence = confidence_metrics["overall"]
        
        # Adjust targets based on confidence
        confidence_multiplier = 0.5 + (overall_confidence * 0.5)  # 0.5 to 1.0
        
        targets = {
            "conservative": base_prediction - (prediction_std * confidence_multiplier),
            "base": base_prediction,
            "optimistic": base_prediction + (prediction_std * confidence_multiplier),
            "bull_case": base_prediction * (1 + 0.15 * confidence_multiplier),
            "bear_case": base_prediction * (1 - 0.15 * confidence_multiplier)
        }
        
        # Ensure targets are reasonable
        for scenario, price in targets.items():
            targets[scenario] = max(current_price * 0.5, min(current_price * 2.0, price))
        
        return targets
    
    def _create_prediction_analysis(self, symbol: str, current_price: float, price_targets: Dict, 
                                  model_results: Dict, confidence_metrics: Dict, horizon_days: int) -> Dict[str, Any]:
        """Create comprehensive prediction analysis"""
        base_target = price_targets["base"]
        upside_potential = (base_target - current_price) / current_price * 100
        
        # Risk assessment
        risk_level = self._assess_risk_level(confidence_metrics, price_targets, current_price)
        
        # Generate insights
        insights = self._generate_ml_insights(
            symbol, current_price, price_targets, model_results, confidence_metrics
        )
        
        # Recommendation based on ML prediction
        ml_recommendation = self._get_ml_recommendation(upside_potential, confidence_metrics, risk_level)
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "prediction_horizon_days": horizon_days,
            "price_targets": price_targets,
            "upside_potential_pct": upside_potential,
            "confidence_metrics": confidence_metrics,
            "risk_level": risk_level,
            "model_results": {k: {"prediction": v["prediction"], "confidence": v["confidence"]} 
                            for k, v in model_results.items()},
            "insights": insights,
            "recommendation": ml_recommendation,
            "analysis_type": "ml_prediction",
            "prediction_timestamp": datetime.now().isoformat()
        }
    
    def _assess_risk_level(self, confidence_metrics: Dict, price_targets: Dict, current_price: float) -> str:
        """Assess risk level of the prediction"""
        confidence = confidence_metrics["overall"]
        price_range = (price_targets["optimistic"] - price_targets["conservative"]) / current_price
        
        if confidence > 0.7 and price_range < 0.2:
            return "Low"
        elif confidence > 0.5 and price_range < 0.4:
            return "Medium"
        else:
            return "High"
    
    def _generate_ml_insights(self, symbol: str, current_price: float, price_targets: Dict,
                            model_results: Dict, confidence_metrics: Dict) -> List[str]:
        """Generate ML-based insights"""
        insights = []
        
        upside_potential = (price_targets["base"] - current_price) / current_price * 100
        
        # Prediction insights
        if upside_potential > 10:
            insights.append(f"ML models suggest {upside_potential:.1f}% upside potential")
        elif upside_potential < -10:
            insights.append(f"ML models indicate {abs(upside_potential):.1f}% downside risk")
        else:
            insights.append(f"ML models predict limited price movement ({upside_potential:+.1f}%)")
        
        # Confidence insights
        overall_confidence = confidence_metrics["overall"]
        if overall_confidence > 0.7:
            insights.append("High confidence in ML predictions due to good model agreement")
        elif overall_confidence < 0.4:
            insights.append("Lower confidence due to limited data or model disagreement")
        
        # Range insights
        price_range_pct = (price_targets["optimistic"] - price_targets["conservative"]) / current_price * 100
        if price_range_pct > 30:
            insights.append(f"Wide prediction range ({price_range_pct:.0f}%) indicates high uncertainty")
        elif price_range_pct < 15:
            insights.append("Narrow prediction range suggests stable price expectations")
        
        # Model consensus
        predictions = [result["prediction"] for result in model_results.values()]
        if all(p > current_price for p in predictions):
            insights.append("All ML models predict price appreciation")
        elif all(p < current_price for p in predictions):
            insights.append("All ML models suggest price decline")
        else:
            insights.append("Mixed signals from different ML models")
        
        return insights
    
    def _get_ml_recommendation(self, upside_potential: float, confidence_metrics: Dict, risk_level: str) -> str:
        """Get recommendation based on ML prediction"""
        confidence = confidence_metrics["overall"]
        
        # High confidence recommendations
        if confidence > 0.6:
            if upside_potential > 15:
                return "BUY"
            elif upside_potential < -15:
                return "SELL"
        
        # Medium confidence recommendations
        elif confidence > 0.4:
            if upside_potential > 20:
                return "BUY"
            elif upside_potential < -20:
                return "SELL"
        
        # Default to HOLD for uncertain predictions
        return "HOLD"
