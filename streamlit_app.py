import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Indian Stock Recommendation System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
}

.recommendation-buy {
    background-color: #d4edda;
    color: #155724;
    padding: 10px;
    border-radius: 5px;
    font-weight: bold;
    text-align: center;
}

.recommendation-sell {
    background-color: #f8d7da;
    color: #721c24;
    padding: 10px;
    border-radius: 5px;
    font-weight: bold;
    text-align: center;
}

.recommendation-hold {
    background-color: #fff3cd;
    color: #856404;
    padding: 10px;
    border-radius: 5px;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# API Base URL
API_BASE_URL = "http://localhost:8000/api/v1"

class StockAnalysisApp:
    def __init__(self):
        self.session = requests.Session()
    
    def check_api_health(self):
        """Check if the FastAPI backend is running"""
        try:
            response = self.session.get("http://localhost:8000/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_stock_analysis(self, symbol, analysis_type):
        """Get comprehensive stock analysis"""
        try:
            url = f"{API_BASE_URL}/analysis/analyze"
            payload = {
                "symbol": symbol,
                "analysis_type": analysis_type
            }
            response = self.session.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_stock_info(self, symbol):
        """Get basic stock information"""
        try:
            url = f"{API_BASE_URL}/stocks/info/{symbol}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_recommendations(self, limit=10):
        """Get top stock recommendations"""
        try:
            url = f"{API_BASE_URL}/recommendations/top-picks?limit={limit}"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_historical_data(self, symbol, period="1y"):
        """Get historical stock data"""
        try:
            url = f"{API_BASE_URL}/stocks/historical/{symbol}?period={period}"
            response = self.session.get(url, timeout=20)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

def main():
    app = StockAnalysisApp()
    
    # Header
    st.markdown('<h1 class="main-header">📈 Indian Stock Recommendation System</h1>', unsafe_allow_html=True)
    
    # Check API health
    if not app.check_api_health():
        st.error("🚨 **FastAPI Backend is not running!**")
        st.info("Please start the FastAPI server by running: `python -m app.main`")
        st.stop()
    
    st.success("✅ **Backend API is running**")
    
    # Sidebar
    st.sidebar.header("🔧 Analysis Configuration")
    
    # Stock symbol input
    stock_symbol = st.sidebar.text_input(
        "Stock Symbol", 
        value="RELIANCE.NS",
        help="Enter Indian stock symbol (e.g., RELIANCE.NS, TCS.NS, INFY.NS)"
    ).upper()
    
    # Analysis type selection
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["comprehensive", "fundamental", "technical", "sentiment"],
        help="Choose the type of analysis to perform"
    )
    
    # Period for historical data
    period = st.sidebar.selectbox(
        "Historical Data Period",
        ["1y", "6mo", "3mo", "1mo", "5d"],
        help="Select time period for historical data"
    )
    
    # Main content
    if st.sidebar.button("🔍 Analyze Stock", type="primary"):
        if stock_symbol:
            analyze_stock(app, stock_symbol, analysis_type, period)
        else:
            st.error("Please enter a stock symbol")
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📊 Stock Analysis", "🏆 Top Recommendations", "📈 Market Overview"])
    
    with tab1:
        st.header("Stock Analysis")
        if stock_symbol:
            display_quick_info(app, stock_symbol)
    
    with tab2:
        st.header("Top Stock Recommendations")
        display_top_recommendations(app)
    
    with tab3:
        st.header("Market Overview")
        display_market_overview(app)

def analyze_stock(app, symbol, analysis_type, period):
    """Perform comprehensive stock analysis"""
    
    with st.spinner(f"Analyzing {symbol}..."):
        # Get analysis
        analysis_result = app.get_stock_analysis(symbol, analysis_type)
        
        if not analysis_result.get("success"):
            st.error(f"Error: {analysis_result.get('error', 'Unknown error')}")
            return
        
        data = analysis_result["data"]
        
        # Display stock information header
        display_stock_header(data)
        
        # Show comprehensive analysis first if available
        if analysis_type == "comprehensive" and data.get("comprehensive_score"):
            display_comprehensive_analysis(data)
            st.divider()
        
        # Create tabs for different analysis types
        available_analyses = []
        if data.get("fundamental") and not data["fundamental"].get("error"):
            available_analyses.append("Fundamental")
        if data.get("technical") and not data["technical"].get("error"):
            available_analyses.append("Technical")
        if data.get("sentiment") and not data["sentiment"].get("error"):
            available_analyses.append("Sentiment")
        if data.get("ml_prediction") and not data["ml_prediction"].get("error"):
            available_analyses.append("ML Prediction")
        
        if len(available_analyses) > 1:
            tabs = st.tabs(available_analyses + ["📊 Historical Chart"])
            
            tab_index = 0
            
            # Fundamental analysis tab
            if "Fundamental" in available_analyses:
                with tabs[tab_index]:
                    display_fundamental_analysis(data["fundamental"])
                tab_index += 1
            
            # Technical analysis tab
            if "Technical" in available_analyses:
                with tabs[tab_index]:
                    display_technical_analysis(data["technical"])
                tab_index += 1
            
            # Sentiment analysis tab
            if "Sentiment" in available_analyses:
                with tabs[tab_index]:
                    display_sentiment_analysis(data["sentiment"])
                tab_index += 1
            
            # ML Prediction tab
            if "ML Prediction" in available_analyses:
                with tabs[tab_index]:
                    display_ml_prediction(data["ml_prediction"])
                tab_index += 1
            
            # Historical chart tab
            with tabs[tab_index]:
                display_historical_chart(app, symbol, period)
        
        else:
            # Single analysis type
            if data.get("fundamental") and not data["fundamental"].get("error"):
                display_fundamental_analysis(data["fundamental"])
            elif data.get("technical") and not data["technical"].get("error"):
                display_technical_analysis(data["technical"])
            elif data.get("sentiment") and not data["sentiment"].get("error"):
                display_sentiment_analysis(data["sentiment"])
            elif data.get("ml_prediction") and not data["ml_prediction"].get("error"):
                display_ml_prediction(data["ml_prediction"])
            
            # Always show historical chart
            st.divider()
            display_historical_chart(app, symbol, period)

def display_stock_header(data):
    """Display stock basic information"""
    stock_info = data.get("stock_info", {})
    
    st.subheader(f"🏢 {stock_info.get('name', 'N/A')}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Price",
            value=f"₹{stock_info.get('current_price', 0):,.2f}",
            delta=None
        )
    
    with col2:
        market_cap = stock_info.get('market_cap', 0)
        if market_cap:
            market_cap_cr = market_cap / 10000000  # Convert to crores
            st.metric(
                label="Market Cap",
                value=f"₹{market_cap_cr:,.0f} Cr"
            )
    
    with col3:
        st.metric(
            label="Sector",
            value=stock_info.get('sector', 'N/A')
        )
    
    with col4:
        st.metric(
            label="Exchange",
            value=stock_info.get('exchange', 'N/A')
        )

def display_fundamental_analysis(fundamental_data):
    """Display enhanced fundamental analysis results"""
    st.subheader("📊 Fundamental Analysis")
    
    overall_score = fundamental_data.get("overall_score", 0)
    recommendation = fundamental_data.get("recommendation", "HOLD")
    
    # Recommendation badge
    if recommendation == "BUY":
        st.markdown(f'<div class="recommendation-buy">🟢 {recommendation}</div>', unsafe_allow_html=True)
    elif recommendation == "SELL":
        st.markdown(f'<div class="recommendation-sell">🔴 {recommendation}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="recommendation-hold">🟡 {recommendation}</div>', unsafe_allow_html=True)
    
    # Overall score with progress bar
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Overall Score", f"{overall_score:.1f}/100")
    with col2:
        st.progress(overall_score / 100)
    
    # Key metrics in expandable sections
    fund_data = fundamental_data.get("fundamental_data", {})
    scores = fundamental_data.get("scores", {})
    
    with st.expander("📈 Key Financial Ratios"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if fund_data.get("pe_ratio"):
                st.metric("P/E Ratio", f"{fund_data['pe_ratio']:.2f}")
            if fund_data.get("eps"):
                st.metric("EPS", f"₹{fund_data['eps']:.2f}")
        
        with col2:
            if fund_data.get("roe"):
                roe_pct = fund_data['roe'] * 100 if fund_data['roe'] < 1 else fund_data['roe']
                st.metric("ROE", f"{roe_pct:.1f}%")
            if fund_data.get("debt_to_equity"):
                st.metric("Debt/Equity", f"{fund_data['debt_to_equity']:.2f}")
        
        with col3:
            if fund_data.get("profit_margin"):
                margin_pct = fund_data['profit_margin'] * 100 if fund_data['profit_margin'] < 1 else fund_data['profit_margin']
                st.metric("Profit Margin", f"{margin_pct:.1f}%")
            if fund_data.get("dividend_yield"):
                div_pct = fund_data['dividend_yield'] * 100 if fund_data['dividend_yield'] < 1 else fund_data['dividend_yield']
                st.metric("Dividend Yield", f"{div_pct:.2f}%")
    
    with st.expander("🎯 Scoring Breakdown"):
        if scores:
            score_df = pd.DataFrame([
                {"Metric": metric.replace("_score", "").replace("_", " ").title(), 
                 "Score": f"{score:.0f}/100"}
                for metric, score in scores.items()
            ])
            st.table(score_df)
    
    # Enhanced insights
    insights = fundamental_data.get("insights", [])
    if insights:
        st.write("**💡 Key Insights:**")
        for i, insight in enumerate(insights, 1):
            st.write(f"{i}. {insight}")

def display_technical_analysis(technical_data):
    """Display enhanced technical analysis results"""
    st.subheader("📈 Technical Analysis")
    
    if technical_data.get("error"):
        st.error(f"Technical Analysis Error: {technical_data['error']}")
        return
    
    overall_score = technical_data.get("overall_score", 50)
    recommendation = technical_data.get("recommendation", "HOLD")
    indicators = technical_data.get("indicators", {})
    signals = technical_data.get("signals", {})
    
    # Recommendation and score
    col1, col2 = st.columns(2)
    with col1:
        if recommendation == "BUY":
            st.success(f"🟢 {recommendation}")
        elif recommendation == "SELL":
            st.error(f"🔴 {recommendation}")
        else:
            st.warning(f"🟡 {recommendation}")
    
    with col2:
        st.metric("Technical Score", f"{overall_score:.1f}/100")
    
    # Technical indicators
    with st.expander("📊 Technical Indicators"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if indicators.get("rsi"):
                rsi_color = "🟢" if 30 <= indicators["rsi"] <= 70 else "🔴" if indicators["rsi"] > 70 else "🟡"
                st.metric("RSI", f"{indicators['rsi']:.1f} {rsi_color}")
            
            if indicators.get("macd"):
                st.metric("MACD", f"{indicators['macd']:.3f}")
        
        with col2:
            if indicators.get("sma_20"):
                st.metric("SMA 20", f"₹{indicators['sma_20']:.2f}")
            if indicators.get("sma_50"):
                st.metric("SMA 50", f"₹{indicators['sma_50']:.2f}")
        
        with col3:
            if indicators.get("volume_ratio"):
                vol_color = "🟢" if indicators["volume_ratio"] > 1.2 else "🔴" if indicators["volume_ratio"] < 0.8 else "🟡"
                st.metric("Volume Ratio", f"{indicators['volume_ratio']:.2f} {vol_color}")
            
            if indicators.get("atr"):
                st.metric("ATR", f"₹{indicators['atr']:.2f}")
    
    # Trading signals
    with st.expander("⚡ Trading Signals"):
        if signals:
            signal_df = pd.DataFrame([
                {"Indicator": indicator.replace("_signal", "").replace("_", " ").title(),
                 "Signal": "🟢 " + signal if signal == "BUY" else "🔴 " + signal if signal == "SELL" else "🟡 " + signal}
                for indicator, signal in signals.items()
            ])
            st.table(signal_df)
    
    # Support and Resistance
    if indicators.get("support") and indicators.get("resistance"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Support Level", f"₹{indicators['support']:.2f}")
        with col2:
            st.metric("Resistance Level", f"₹{indicators['resistance']:.2f}")
    
    # Technical insights
    insights = technical_data.get("insights", [])
    if insights:
        st.write("**💡 Technical Insights:**")
        for i, insight in enumerate(insights, 1):
            st.write(f"{i}. {insight}")

def display_sentiment_analysis(sentiment_data):
    """Display enhanced sentiment analysis results"""
    st.subheader("📰 Sentiment Analysis")
    
    if sentiment_data.get("error"):
        st.error(f"Sentiment Analysis Error: {sentiment_data['error']}")
        return
    
    overall_sentiment = sentiment_data.get("overall_sentiment", {})
    sentiment_score = overall_sentiment.get("score", 0.5)
    sentiment_interp = overall_sentiment.get("interpretation", "Neutral")
    news_count = sentiment_data.get("news_count", 0)
    
    # Overall sentiment display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Sentiment", sentiment_interp)
    
    with col2:
        st.metric("Sentiment Score", f"{sentiment_score:.2f}")
    
    with col3:
        st.metric("News Articles", news_count)
    
    # Sentiment gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = sentiment_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Sentiment Score"},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "red"},
                {'range': [0.3, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.9
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed sentiment breakdown
    with st.expander("📊 Sentiment Breakdown"):
        news_sentiment = sentiment_data.get("news_sentiment", {})
        social_sentiment = sentiment_data.get("social_sentiment", {})
        market_sentiment = sentiment_data.get("market_sentiment", {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**📰 News Sentiment**")
            if news_sentiment:
                st.metric("Score", f"{news_sentiment.get('score', 0):.2f}")
                st.metric("Positive Articles", news_sentiment.get('positive_articles', 0))
                st.metric("Negative Articles", news_sentiment.get('negative_articles', 0))
        
        with col2:
            st.write("**💬 Social Sentiment**")
            if social_sentiment:
                st.metric("Score", f"{social_sentiment.get('score', 0):.2f}")
                trend = social_sentiment.get('sentiment_trend', 'neutral')
                st.write(f"Trend: {trend.title()}")
        
        with col3:
            st.write("**📈 Market Sentiment**")
            if market_sentiment:
                st.metric("Score", f"{market_sentiment.get('score', 0):.2f}")
                trend = market_sentiment.get('market_trend', 'neutral')
                st.write(f"Market: {trend.title()}")
    
    # Sentiment insights
    insights = sentiment_data.get("insights", [])
    if insights:
        st.write("**💡 Sentiment Insights:**")
        for i, insight in enumerate(insights, 1):
            st.write(f"{i}. {insight}")

def display_ml_prediction(ml_data):
    """Display ML prediction analysis"""
    st.subheader("🤖 ML Target Price Prediction")
    
    if ml_data.get("error"):
        st.error(f"ML Prediction Error: {ml_data['error']}")
        return
    
    current_price = ml_data.get("current_price", 0)
    price_targets = ml_data.get("price_targets", {})
    upside_potential = ml_data.get("upside_potential_pct", 0)
    confidence_metrics = ml_data.get("confidence_metrics", {})
    risk_level = ml_data.get("risk_level", "Medium")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"₹{current_price:.2f}")
    
    with col2:
        base_target = price_targets.get("base", current_price)
        st.metric("Target Price", f"₹{base_target:.2f}")
    
    with col3:
        upside_color = "🟢" if upside_potential > 0 else "🔴"
        st.metric("Upside Potential", f"{upside_potential:+.1f}% {upside_color}")
    
    with col4:
        overall_confidence = confidence_metrics.get("overall", 0.5) * 100
        st.metric("Confidence", f"{overall_confidence:.0f}%")
    
    # Price targets chart
    if price_targets:
        target_df = pd.DataFrame([
            {"Scenario": scenario.replace("_", " ").title(), "Price": f"₹{price:.2f}", "Value": price}
            for scenario, price in price_targets.items()
        ])
        
        fig = px.bar(target_df, x="Scenario", y="Value", 
                    title="ML Price Targets by Scenario",
                    labels={"Value": "Price (₹)"})
        fig.add_hline(y=current_price, line_dash="dash", line_color="red", 
                     annotation_text="Current Price")
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk assessment
    col1, col2 = st.columns(2)
    with col1:
        risk_color = "🟢" if risk_level == "Low" else "🟡" if risk_level == "Medium" else "🔴"
        st.metric("Risk Level", f"{risk_level} {risk_color}")
    
    with col2:
        horizon = ml_data.get("prediction_horizon_days", 30)
        st.metric("Prediction Horizon", f"{horizon} days")
    
    # ML insights
    insights = ml_data.get("insights", [])
    if insights:
        st.write("**💡 ML Insights:**")
        for i, insight in enumerate(insights, 1):
            st.write(f"{i}. {insight}")

def display_comprehensive_analysis(data):
    """Display comprehensive analysis results"""
    if not data.get("comprehensive_score"):
        return
    
    st.subheader("🎯 Comprehensive Analysis Summary")
    
    comprehensive_score = data.get("comprehensive_score", 0.5) * 100
    final_recommendation = data.get("final_recommendation", {})
    recommendation = final_recommendation.get("recommendation", "HOLD")
    confidence = final_recommendation.get("confidence", 0.5) * 100
    
    # Overall results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Score", f"{comprehensive_score:.0f}/100")
        st.progress(comprehensive_score / 100)
    
    with col2:
        if recommendation == "BUY":
            st.success(f"🟢 **{recommendation}**")
        elif recommendation == "SELL":
            st.error(f"🔴 **{recommendation}**")
        else:
            st.warning(f"🟡 **{recommendation}**")
    
    with col3:
        st.metric("Confidence", f"{confidence:.0f}%")
    
    # Individual scores
    individual_scores = data.get("individual_scores", {})
    if individual_scores:
        st.write("**📊 Analysis Breakdown:**")
        score_df = pd.DataFrame([
            {"Analysis": analysis.replace("_", " ").title(), 
             "Score": f"{score*100:.0f}/100"}
            for analysis, score in individual_scores.items()
        ])
        st.table(score_df)
    
    # Vote breakdown
    vote_breakdown = final_recommendation.get("vote_breakdown", {})
    if vote_breakdown:
        with st.expander("🗳️ Recommendation Consensus"):
            vote_df = pd.DataFrame([
                {"Recommendation": rec, "Votes": votes}
                for rec, votes in vote_breakdown.items()
            ])
            fig = px.pie(vote_df, values="Votes", names="Recommendation", 
                        title="Analysis Methods Consensus")
            st.plotly_chart(fig, use_container_width=True)
    
    # Target price analysis
    target_price_analysis = data.get("target_price_analysis", {})
    if target_price_analysis:
        with st.expander("🎯 Target Price Analysis"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_target = target_price_analysis.get("average_target", 0)
                st.metric("Average Target", f"₹{avg_target:.2f}")
            
            with col2:
                upside_pct = target_price_analysis.get("upside_potential_pct", 0)
                st.metric("Average Upside", f"{upside_pct:+.1f}%")
            
            with col3:
                target_range = target_price_analysis.get("target_range_pct", 0)
                st.metric("Price Range", f"{target_range:.1f}%")
    
    # Risk assessment
    risk_assessment = data.get("risk_assessment", {})
    if risk_assessment:
        with st.expander("⚠️ Risk Assessment"):
            risk_level = risk_assessment.get("risk_level", "Medium")
            risk_factors = risk_assessment.get("risk_factors", [])
            
            col1, col2 = st.columns(2)
            with col1:
                risk_color = "🟢" if risk_level == "Low" else "🟡" if risk_level == "Medium" else "🔴"
                st.metric("Risk Level", f"{risk_level} {risk_color}")
            
            with col2:
                recommendation_text = risk_assessment.get("recommendation", "")
                st.write(f"**Suitability:** {recommendation_text}")
            
            if risk_factors:
                st.write("**Risk Factors:**")
                for factor in risk_factors:
                    st.write(f"• {factor}")
    
    # Investment thesis
    investment_thesis = data.get("investment_thesis", {})
    if investment_thesis:
        with st.expander("📋 Investment Thesis"):
            st.write(f"**Summary:** {investment_thesis.get('summary', 'N/A')}")
            
            strengths = investment_thesis.get("key_strengths", [])
            concerns = investment_thesis.get("key_concerns", [])
            
            if strengths or concerns:
                col1, col2 = st.columns(2)
                
                with col1:
                    if strengths:
                        st.write("**💪 Key Strengths:**")
                        for strength in strengths:
                            st.write(f"• {strength}")
                
                with col2:
                    if concerns:
                        st.write("**⚠️ Key Concerns:**")
                        for concern in concerns:
                            st.write(f"• {concern}")
    
    # Comprehensive insights
    comprehensive_insights = data.get("comprehensive_insights", [])
    if comprehensive_insights:
        st.write("**💡 Key Insights:**")
        for i, insight in enumerate(comprehensive_insights, 1):
            st.write(f"{i}. {insight}")

def display_historical_chart(app, symbol, period):
    """Display historical price chart"""
    st.subheader("📊 Price History")
    
    with st.spinner("Loading historical data..."):
        hist_result = app.get_historical_data(symbol, period)
        
        if hist_result.get("success") and hist_result.get("data", {}).get("data"):
            hist_data = hist_result["data"]["data"]
            df = pd.DataFrame(hist_data)
            
            if not df.empty and 'Close' in df.columns:
                # Convert Date column to datetime
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                else:
                    df.reset_index(inplace=True)
                    df['Date'] = pd.to_datetime(df.index)
                
                # Create candlestick chart
                fig = go.Figure(data=go.Candlestick(
                    x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close']
                ))
                
                fig.update_layout(
                    title=f"{symbol} Price Chart ({period})",
                    yaxis_title="Price (₹)",
                    xaxis_title="Date",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Price statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current", f"₹{df['Close'].iloc[-1]:.2f}")
                
                with col2:
                    change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
                    change_pct = (change / df['Close'].iloc[-2]) * 100
                    st.metric("Change", f"₹{change:.2f}", f"{change_pct:+.2f}%")
                
                with col3:
                    st.metric("High", f"₹{df['High'].max():.2f}")
                
                with col4:
                    st.metric("Low", f"₹{df['Low'].min():.2f}")
            else:
                st.warning("No historical data available")
        else:
            st.error("Unable to load historical data")

def display_quick_info(app, symbol):
    """Display quick stock information"""
    if symbol:
        with st.spinner(f"Loading {symbol} info..."):
            info_result = app.get_stock_info(symbol)
            
            if info_result.get("success"):
                stock_data = info_result["data"]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Price", f"₹{stock_data.get('current_price', 0):,.2f}")
                
                with col2:
                    market_cap = stock_data.get('market_cap', 0)
                    if market_cap:
                        market_cap_cr = market_cap / 10000000
                        st.metric("Market Cap", f"₹{market_cap_cr:,.0f} Cr")
                
                with col3:
                    st.metric("Sector", stock_data.get('sector', 'N/A'))

def display_top_recommendations(app):
    """Display top stock recommendations"""
    with st.spinner("Loading top recommendations..."):
        reco_result = app.get_recommendations(10)
        
        if reco_result.get("success") and reco_result.get("data"):
            recommendations = reco_result["data"]
            
            for reco in recommendations:
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
                    
                    with col1:
                        st.write(f"**{reco.get('name', reco.get('symbol', 'N/A'))}**")
                        st.write(f"({reco.get('symbol', 'N/A')})")
                    
                    with col2:
                        recommendation = reco.get('recommendation', 'HOLD')
                        if recommendation == 'BUY':
                            st.success(f"🟢 {recommendation}")
                        elif recommendation == 'SELL':
                            st.error(f"🔴 {recommendation}")
                        else:
                            st.warning(f"🟡 {recommendation}")
                    
                    with col3:
                        st.metric("Current", f"₹{reco.get('current_price', 0):,.2f}")
                    
                    with col4:
                        st.metric("Target", f"₹{reco.get('target_price', 0):,.2f}")
                    
                    with col5:
                        confidence = reco.get('confidence_score', 0) * 100
                        st.metric("Confidence", f"{confidence:.0f}%")
                    
                    st.divider()
        else:
            st.error("Unable to load recommendations")

def display_market_overview(app):
    """Display market overview"""
    st.info("Market overview will be implemented with real-time market data")
    
    # Placeholder market data
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("NIFTY 50")
        st.metric("Index", "19,500", "+150 (+0.77%)")
    
    with col2:
        st.subheader("SENSEX")
        st.metric("Index", "65,800", "+450 (+0.69%)")

if __name__ == "__main__":
    main()
