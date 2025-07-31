import pandas as pd
import numpy as np
# import pandas_ta as ta
from typing import Dict, Any, Optional, List
import logging
from app.services.data_collector import DataCollector

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    def __init__(self):
        self.data_collector = DataCollector()
    
    def analyze(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """Perform comprehensive technical analysis"""
        try:
            # Get historical data
            hist_data = self.data_collector.get_historical_data(symbol, period)
            
            if hist_data.empty:
                return {"error": "No historical data available"}
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(hist_data)
            
            # Calculate technical scores
            scores = self._calculate_technical_scores(indicators, hist_data)
            
            # Generate trading signals
            signals = self._generate_signals(indicators, hist_data)
            
            # Calculate overall technical score
            overall_score = self._calculate_overall_technical_score(scores)
            
            # Generate insights
            insights = self._generate_technical_insights(indicators, signals, hist_data)
            
            return {
                "symbol": symbol,
                "indicators": indicators,
                "scores": scores,
                "signals": signals,
                "overall_score": overall_score,
                "insights": insights,
                "recommendation": self._get_technical_recommendation(overall_score, signals),
                "analysis_type": "technical"
            }
            
        except Exception as e:
            logger.error(f"Error in technical analysis for {symbol}: {e}")
            return {"error": str(e)}
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators"""
        indicators = {}
        
        # Ensure we have enough data
        if len(df) < 50:
            return {"error": "Insufficient data for technical analysis"}
        
        # Price data
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # Moving Averages
        indicators['sma_20'] = close.rolling(20).mean().iloc[-1]
        indicators['sma_50'] = close.rolling(50).mean().iloc[-1]
        indicators['ema_12'] = close.ewm(span=12).mean().iloc[-1]
        indicators['ema_26'] = close.ewm(span=26).mean().iloc[-1]
        
        # RSI (Relative Strength Index)
        indicators['rsi'] = self._calculate_rsi(close, 14)
        
        # MACD (Moving Average Convergence Divergence)
        macd_data = self._calculate_macd(close)
        indicators.update(macd_data)
        
        # Bollinger Bands
        bb_data = self._calculate_bollinger_bands(close, 20, 2.0)
        indicators.update(bb_data)
        
        # Stochastic Oscillator
        stoch_data = self._calculate_stochastic(high, low, close, 14, 3, 3)
        indicators.update(stoch_data)
        
        # Williams %R
        indicators['williams_r'] = self._calculate_williams_r(high, low, close, 14)
        
        # Average True Range (ATR) - Volatility
        indicators['atr'] = self._calculate_atr(high, low, close, 14)
        
        # On-Balance Volume (OBV)
        indicators['obv'] = self._calculate_obv(close, volume)
        
        # Commodity Channel Index (CCI)
        indicators['cci'] = self._calculate_cci(high, low, close, 20)
        
        # Support and Resistance Levels
        recent_data = df.tail(50)
        indicators['resistance'] = recent_data['High'].max()
        indicators['support'] = recent_data['Low'].min()
        
        # Current price position
        current_price = close.iloc[-1]
        indicators['current_price'] = current_price
        indicators['price_change_pct'] = ((current_price - close.iloc[-2]) / close.iloc[-2]) * 100
        
        # Volume analysis
        indicators['avg_volume_20'] = volume.rolling(20).mean().iloc[-1]
        indicators['volume_ratio'] = volume.iloc[-1] / indicators['avg_volume_20']
        
        return indicators
    
    def _calculate_technical_scores(self, indicators: Dict[str, Any], df: pd.DataFrame) -> Dict[str, float]:
        """Calculate scores for technical indicators"""
        scores = {}
        
        current_price = indicators.get('current_price', 0)
        
        # RSI Score
        rsi = indicators.get('rsi', 50)
        if 30 <= rsi <= 70:
            scores['rsi_score'] = 80  # Neutral zone
        elif 20 <= rsi < 30:
            scores['rsi_score'] = 90  # Oversold - potential buy
        elif 70 < rsi <= 80:
            scores['rsi_score'] = 70  # Overbought warning
        elif rsi < 20:
            scores['rsi_score'] = 95  # Extremely oversold
        else:  # rsi > 80
            scores['rsi_score'] = 30  # Extremely overbought
        
        # MACD Score
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        if macd > macd_signal:
            scores['macd_score'] = 80  # Bullish
        else:
            scores['macd_score'] = 40  # Bearish
        
        # Moving Average Score
        sma_20 = indicators.get('sma_20', current_price)
        sma_50 = indicators.get('sma_50', current_price)
        if current_price > sma_20 > sma_50:
            scores['ma_score'] = 90  # Strong uptrend
        elif current_price > sma_20:
            scores['ma_score'] = 70  # Mild uptrend
        elif current_price < sma_20 < sma_50:
            scores['ma_score'] = 20  # Strong downtrend
        else:
            scores['ma_score'] = 50  # Neutral
        
        # Bollinger Bands Score
        bb_upper = indicators.get('bb_upper', current_price)
        bb_lower = indicators.get('bb_lower', current_price)
        if bb_lower <= current_price <= bb_upper:
            if current_price < (bb_lower + bb_upper) / 2:
                scores['bb_score'] = 80  # Lower half - potential buy
            else:
                scores['bb_score'] = 60  # Upper half - be cautious
        elif current_price < bb_lower:
            scores['bb_score'] = 90  # Oversold
        else:  # current_price > bb_upper
            scores['bb_score'] = 30  # Overbought
        
        # Stochastic Score
        stoch_k = indicators.get('stoch_k', 50)
        stoch_d = indicators.get('stoch_d', 50)
        if stoch_k < 20 and stoch_d < 20:
            scores['stoch_score'] = 90  # Oversold
        elif stoch_k > 80 and stoch_d > 80:
            scores['stoch_score'] = 30  # Overbought
        elif stoch_k > stoch_d:
            scores['stoch_score'] = 70  # Bullish crossover
        else:
            scores['stoch_score'] = 50  # Neutral
        
        # Volume Score
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > 1.5:
            scores['volume_score'] = 80  # High volume - strong signal
        elif volume_ratio > 1.2:
            scores['volume_score'] = 70  # Above average volume
        elif volume_ratio < 0.8:
            scores['volume_score'] = 40  # Low volume - weak signal
        else:
            scores['volume_score'] = 60  # Normal volume
        
        return scores
    
    def _generate_signals(self, indicators: Dict[str, Any], df: pd.DataFrame) -> Dict[str, str]:
        """Generate trading signals"""
        signals = {}
        
        current_price = indicators.get('current_price', 0)
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        sma_20 = indicators.get('sma_20', current_price)
        sma_50 = indicators.get('sma_50', current_price)
        
        # RSI Signals
        if rsi < 30:
            signals['rsi_signal'] = 'BUY'
        elif rsi > 70:
            signals['rsi_signal'] = 'SELL'
        else:
            signals['rsi_signal'] = 'HOLD'
        
        # MACD Signals
        if macd > macd_signal:
            signals['macd_signal'] = 'BUY'
        else:
            signals['macd_signal'] = 'SELL'
        
        # Moving Average Signals
        if current_price > sma_20 > sma_50:
            signals['ma_signal'] = 'BUY'
        elif current_price < sma_20 < sma_50:
            signals['ma_signal'] = 'SELL'
        else:
            signals['ma_signal'] = 'HOLD'
        
        # Support/Resistance Signals
        support = indicators.get('support', 0)
        resistance = indicators.get('resistance', 0)
        if current_price <= support * 1.02:  # Near support
            signals['sr_signal'] = 'BUY'
        elif current_price >= resistance * 0.98:  # Near resistance
            signals['sr_signal'] = 'SELL'
        else:
            signals['sr_signal'] = 'HOLD'
        
        return signals
    
    def _calculate_overall_technical_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall technical score"""
        weights = {
            'rsi_score': 0.20,
            'macd_score': 0.20,
            'ma_score': 0.25,
            'bb_score': 0.15,
            'stoch_score': 0.10,
            'volume_score': 0.10
        }
        
        total_score = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            if metric in scores:
                total_score += scores[metric] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 50
    
    def _generate_technical_insights(self, indicators: Dict[str, Any], signals: Dict[str, str], df: pd.DataFrame) -> List[str]:
        """Generate technical analysis insights"""
        insights = []
        
        current_price = indicators.get('current_price', 0)
        rsi = indicators.get('rsi', 50)
        
        # RSI insights
        if rsi < 30:
            insights.append(f"RSI at {rsi:.1f} indicates oversold conditions - potential buying opportunity")
        elif rsi > 70:
            insights.append(f"RSI at {rsi:.1f} suggests overbought conditions - consider taking profits")
        
        # Moving average insights
        sma_20 = indicators.get('sma_20', current_price)
        sma_50 = indicators.get('sma_50', current_price)
        if current_price > sma_20 > sma_50:
            insights.append("Price above both 20 and 50-day MA indicates strong uptrend")
        elif current_price < sma_20 < sma_50:
            insights.append("Price below both moving averages suggests downtrend")
        
        # MACD insights
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        if macd > macd_signal:
            insights.append("MACD above signal line indicates bullish momentum")
        else:
            insights.append("MACD below signal line suggests bearish momentum")
        
        # Volume insights
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > 1.5:
            insights.append("High volume confirms the price movement strength")
        elif volume_ratio < 0.8:
            insights.append("Low volume indicates weak conviction in current trend")
        
        # Support/Resistance insights
        support = indicators.get('support', 0)
        resistance = indicators.get('resistance', 0)
        if current_price <= support * 1.02:
            insights.append(f"Price near support level of ₹{support:.2f} - potential reversal zone")
        elif current_price >= resistance * 0.98:
            insights.append(f"Price near resistance level of ₹{resistance:.2f} - breakout or reversal expected")
        
        return insights
    
    def _get_technical_recommendation(self, overall_score: float, signals: Dict[str, str]) -> str:
        """Get technical recommendation based on score and signals"""
        # Count buy/sell signals
        buy_signals = sum(1 for signal in signals.values() if signal == 'BUY')
        sell_signals = sum(1 for signal in signals.values() if signal == 'SELL')
        
        # Combine score and signal consensus
        if overall_score >= 75 and buy_signals >= sell_signals:
            return "BUY"
        elif overall_score <= 40 and sell_signals >= buy_signals:
            return "SELL"
        else:
            return "HOLD"
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            return {
                'macd': float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
                'macd_signal': float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0,
                'macd_histogram': float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0
            }
        except:
            return {'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0}
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> dict:
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            current_upper = float(upper_band.iloc[-1]) if not pd.isna(upper_band.iloc[-1]) else float(prices.iloc[-1])
            current_middle = float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else float(prices.iloc[-1])
            current_lower = float(lower_band.iloc[-1]) if not pd.isna(lower_band.iloc[-1]) else float(prices.iloc[-1])
            
            return {
                'bb_upper': current_upper,
                'bb_middle': current_middle,
                'bb_lower': current_lower,
                'bb_width': (current_upper - current_lower) / current_middle if current_middle != 0 else 0.0
            }
        except:
            current_price = float(prices.iloc[-1])
            return {
                'bb_upper': current_price * 1.02,
                'bb_middle': current_price,
                'bb_lower': current_price * 0.98,
                'bb_width': 0.04
            }
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_period: int = 14, k_smooth: int = 3, d_smooth: int = 3) -> dict:
        """Calculate Stochastic Oscillator"""
        try:
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            k_smooth_values = k_percent.rolling(window=k_smooth).mean()
            d_smooth_values = k_smooth_values.rolling(window=d_smooth).mean()
            
            return {
                'stoch_k': float(k_smooth_values.iloc[-1]) if not pd.isna(k_smooth_values.iloc[-1]) else 50.0,
                'stoch_d': float(d_smooth_values.iloc[-1]) if not pd.isna(d_smooth_values.iloc[-1]) else 50.0
            }
        except:
            return {'stoch_k': 50.0, 'stoch_d': 50.0}
    
    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate Williams %R"""
        try:
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
            return float(williams_r.iloc[-1]) if not pd.isna(williams_r.iloc[-1]) else -50.0
        except:
            return -50.0
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate Average True Range (ATR)"""
        try:
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else float(close.iloc[-1] * 0.02)
        except:
            return float(close.iloc[-1] * 0.02) if len(close) > 0 else 1.0
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> float:
        """Calculate On-Balance Volume (OBV)"""
        try:
            obv = pd.Series(index=close.index, dtype=float)
            obv.iloc[0] = volume.iloc[0]
            
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            return float(obv.iloc[-1]) if not pd.isna(obv.iloc[-1]) else 0.0
        except:
            return 0.0
    
    def _calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> float:
        """Calculate Commodity Channel Index (CCI)"""
        try:
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(lambda x: pd.Series(x).mad())
            
            cci = (typical_price - sma_tp) / (0.015 * mad)
            return float(cci.iloc[-1]) if not pd.isna(cci.iloc[-1]) else 0.0
        except:
            return 0.0
