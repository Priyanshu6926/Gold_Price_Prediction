"""
Indian Gold Price Prediction & Quantitative Forecasting Dashboard.
Built for the Indian Financial Ecosystem with Streamlit and Plotly.
Tracks Nippon India ETF Gold BeES (NSE), 10g 24K Gold (₹), USD/INR, NIFTY 50 & SENSEX.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Path setups
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from data_loader import download_indian_market_data, RAW_DATA_PATH
from features import prepare_full_features, PROCESSED_DATA_PATH
from forecast import forecast_future_prices
from backtest import run_strategy_backtest
from train import train_and_export_all_models, MODELS_DIR

# Set Page Config
st.set_page_config(
    page_title="Indian Gold Price AI Forecaster | Live Market Intelligence",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Indian Gold Theme
st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    .stMetric {
        background-color: #141c2e;
        border: 1px solid #23304c;
        padding: 12px 18px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.4);
    }
    .stMetric label { color: #94a3b8 !important; font-weight: 500; font-size: 0.9rem; }
    .stMetric [data-testid="stMetricValue"] { color: #f8fafc !important; font-size: 1.6rem !important; font-weight: 700; }
    .gold-header {
        background: linear-gradient(90deg, #ffd700, #f59e0b, #d97706);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.3rem;
    }
    .badge-card {
        background: #141c2e;
        border: 1px solid #23304c;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_cached_market_data():
    if os.path.exists(RAW_DATA_PATH):
        df = pd.read_csv(RAW_DATA_PATH, index_col=0, parse_dates=True)
    else:
        df = download_indian_market_data()
    return df


@st.cache_data(ttl=3600)
def load_cached_features():
    if os.path.exists(PROCESSED_DATA_PATH):
        df = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
    else:
        raw = load_cached_market_data()
        df = prepare_full_features(raw)
    return df


def load_test_artifacts():
    metrics_path = os.path.join(MODELS_DIR, 'test_metrics.json')
    preds_path = os.path.join(MODELS_DIR, 'test_predictions.json')
    cv_path = os.path.join(MODELS_DIR, 'cv_metrics.csv')
    feat_imp_path = os.path.join(MODELS_DIR, 'feature_importance.csv')
    
    metrics, preds, cv_df, feat_imp = None, None, None, None
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    if os.path.exists(preds_path):
        with open(preds_path, 'r') as f:
            preds = json.load(f)
    if os.path.exists(cv_path):
        cv_df = pd.read_csv(cv_path, index_col=0)
    if os.path.exists(feat_imp_path):
        feat_imp = pd.read_csv(feat_imp_path, index_col=0)
        
    return metrics, preds, cv_df, feat_imp


# --- App Header ---
st.markdown('<h1 class="gold-header">🇮🇳 Indian Gold Price AI Forecaster & Market Intelligence</h1>', unsafe_allow_html=True)
st.markdown("Live Multi-Asset Intelligence for Indian Markets • NSE Gold BeES & Spot 24K Gold • Multi-Horizon AI Projections")

# Load data
raw_data = load_cached_market_data()
feat_data = load_cached_features()
metrics, test_preds, cv_df, feat_imp = load_test_artifacts()

latest_bees = float(raw_data['GOLDBEES'].iloc[-1])
prev_bees = float(raw_data['GOLDBEES'].iloc[-2])
day_change = latest_bees - prev_bees
day_pct = (day_change / prev_bees) * 100.0

latest_10g = float(raw_data['GOLD_INR_10G'].iloc[-1]) if 'GOLD_INR_10G' in raw_data else latest_bees * 1000
prev_10g = float(raw_data['GOLD_INR_10G'].iloc[-2]) if 'GOLD_INR_10G' in raw_data else prev_bees * 1000
change_10g = latest_10g - prev_10g
latest_date = raw_data.index[-1].strftime('%B %d, %Y')

# --- Top Banner Metric Ticker ---
t1, t2, t3, t4, t5, t6 = st.columns(6)
with t1:
    st.metric("Gold BeES (NSE)", f"₹{latest_bees:.2f}", f"{day_change:+.2f} ({day_pct:+.2f}%)")
with t2:
    st.metric("24K Gold (₹/10g)", f"₹{latest_10g:,.0f}", f"{change_10g:+,.0f}")
with t3:
    if 'USDINR' in raw_data.columns:
        st.metric("USD / INR", f"₹{raw_data['USDINR'].iloc[-1]:.2f}")
with t4:
    if 'NIFTY50' in raw_data.columns:
        st.metric("NIFTY 50", f"{raw_data['NIFTY50'].iloc[-1]:,.2f}")
with t5:
    if 'SENSEX' in raw_data.columns:
        st.metric("BSE SENSEX", f"{raw_data['SENSEX'].iloc[-1]:,.2f}")
with t6:
    if 'INDIA_VIX' in raw_data.columns:
        st.metric("India VIX", f"{raw_data['INDIA_VIX'].iloc[-1]:.2f}")

# --- Sidebar ---
st.sidebar.markdown("### 🇮🇳 Indian Market Controls")
st.sidebar.caption(f"Latest Market Date: **{latest_date}**")

target_view = st.sidebar.selectbox(
    "Target Instrument Display:",
    ["Nippon India ETF Gold BeES (₹/unit)", "Domestic Spot Gold (₹/10 Grams - 24K)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Pipeline Control")

if st.sidebar.button("🔄 Refresh Live Indian Market Data", use_container_width=True):
    with st.spinner("Downloading live data from NSE, BSE & Yahoo Finance..."):
        download_indian_market_data(force_refresh=True)
        prepare_full_features(load_cached_market_data(), save=True)
        st.cache_data.clear()
        st.success("Indian market dataset updated!")
        st.rerun()

if st.sidebar.button("🧠 Retrain All ML & DL Models", use_container_width=True):
    with st.spinner("Training Random Forest, XGBoost, LightGBM, LSTM on Indian Market Data..."):
        train_and_export_all_models()
        st.cache_data.clear()
        st.success("All Indian market models retrained and saved!")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Technology & Data Stack")
st.sidebar.caption("• Target: GOLDBEES.NS (NSE) & 24K Gold Rate\n• Macro: USD/INR, NIFTY 50, SENSEX, India VIX, Crude\n• Models: LightGBM, XGBoost, PyTorch LSTM, Ensemble\n• Cross-Validation: Walk-Forward Expanding TimeSeriesSplit")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Indian Market & Technicals",
    "🤖 Model Arena & Benchmark",
    "🔮 Future Price Forecast (₹)",
    "📊 Strategy Backtest (INR)",
    "📚 Indian Market Dynamics"
])

# ==========================================
# TAB 1: INDIAN MARKET & TECHNICAL ANALYSIS
# ==========================================
with tab1:
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("50-Day Moving Avg", f"₹{feat_data['SMA_50'].iloc[-1]:.2f}")
    with col_m2:
        st.metric("200-Day Moving Avg", f"₹{feat_data['SMA_200'].iloc[-1]:.2f}")
    with col_m3:
        rsi_val = feat_data['RSI_14'].iloc[-1]
        rsi_state = "Overbought (>70)" if rsi_val > 70 else ("Oversold (<30)" if rsi_val < 30 else "Neutral")
        st.metric("14-Day RSI", f"{rsi_val:.1f}", rsi_state)
    with col_m4:
        st.metric("Gold / Silver Ratio (INR)", f"{feat_data['Gold_Silver_Ratio_INR'].iloc[-1]:.2f}" if 'Gold_Silver_Ratio_INR' in feat_data else "N/A")

    st.markdown("#### 🕯️ Interactive Indian Gold Price & Technical Indicators")
    time_range = st.radio("Timeframe Range:", ["1 Year", "3 Years", "5 Years", "All Time (2010-Present)"], horizontal=True, index=0)
    
    if time_range == "1 Year":
        plot_df = feat_data.tail(252)
    elif time_range == "3 Years":
        plot_df = feat_data.tail(252 * 3)
    elif time_range == "5 Years":
        plot_df = feat_data.tail(252 * 5)
    else:
        plot_df = feat_data

    fig_price = go.Figure()
    
    if 'GOLDBEES_Open' in plot_df.columns:
        fig_price.add_trace(go.Candlestick(
            x=plot_df.index,
            open=plot_df['GOLDBEES_Open'],
            high=plot_df['GOLDBEES_High'],
            low=plot_df['GOLDBEES_Low'],
            close=plot_df['GOLDBEES_Close'],
            name="Gold BeES OHLC",
            increasing_line_color='#00e676',
            decreasing_line_color='#ff5252'
        ))
    else:
        fig_price.add_trace(go.Scatter(x=plot_df.index, y=plot_df['GOLDBEES'], mode='lines', name='Gold BeES (₹)', line=dict(color='#ffd700', width=2)))

    fig_price.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_50'], name='SMA 50', line=dict(color='#29b6f6', width=1.5)))
    fig_price.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_200'], name='SMA 200', line=dict(color='#ab47bc', width=1.5)))
    fig_price.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_Upper'], name='Upper Bollinger Band', line=dict(color='rgba(255,255,255,0.3)', dash='dot')))
    fig_price.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_Lower'], name='Lower Bollinger Band', line=dict(color='rgba(255,255,255,0.3)', dash='dot'), fill='tonexty', fillcolor='rgba(255,215,0,0.04)'))

    fig_price.update_layout(
        template='plotly_dark',
        height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # Momentum Subplots (RSI & MACD)
    col_sub1, col_sub2 = st.columns(2)
    with col_sub1:
        st.markdown("##### ⚡ RSI (Relative Strength Index - 14)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI_14'], line=dict(color='#ffd700', width=1.5), name='RSI 14'))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ff5252", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#00e676", annotation_text="Oversold (30)")
        fig_rsi.update_layout(template='plotly_dark', height=240, margin=dict(l=20, r=20, t=20, b=20), yaxis=dict(range=[10, 90]))
        st.plotly_chart(fig_rsi, use_container_width=True)

    with col_sub2:
        st.markdown("##### 🌊 MACD Indicator")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MACD'], line=dict(color='#29b6f6', width=1.5), name='MACD'))
        fig_macd.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MACD_Signal'], line=dict(color='#ff9100', width=1.5), name='Signal'))
        fig_macd.add_trace(go.Bar(x=plot_df.index, y=plot_df['MACD_Diff'], name='Histogram', marker_color='#80cbc4'))
        fig_macd.update_layout(template='plotly_dark', height=240, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_macd, use_container_width=True)

    # Indian Market Correlation Matrix
    st.markdown("#### 🔥 Indian Multi-Asset Correlation Matrix")
    corr_assets = ['GOLDBEES', 'GOLD_INR_10G', 'USDINR', 'NIFTY50', 'SENSEX', 'INDIA_VIX', 'SILVER', 'CRUDE_OIL']
    available_corr = [c for c in corr_assets if c in raw_data.columns]
    corr_matrix = raw_data[available_corr].corr().round(3)
    
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Viridis",
        template="plotly_dark",
        title="Domestic Asset Correlation Heatmap (Gold BeES, NIFTY 50, SENSEX, USD/INR, India VIX)"
    )
    fig_corr.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)


# ==========================================
# TAB 2: MODEL BENCHMARK & PERFORMANCE
# ==========================================
with tab2:
    st.markdown("### 🏆 Indian Market Model Comparison & Holdout Evaluation")
    st.caption("Evaluated on chronologically isolated Indian market holdout test data.")

    if metrics is not None:
        metrics_df = pd.DataFrame(metrics).T
        st.dataframe(
            metrics_df.style.format({
                'RMSE (₹)': '₹{:.3f}',
                'MAE (₹)': '₹{:.3f}',
                'MAPE (%)': '{:.2f}%',
                'R2_Score': '{:.4f}',
                'Directional_Accuracy (%)': '{:.2f}%'
            }).highlight_min(subset=['RMSE (₹)', 'MAE (₹)', 'MAPE (%)'], color='#1b4d3e')
              .highlight_max(subset=['R2_Score', 'Directional_Accuracy (%)'], color='#1b4d3e'),
            use_container_width=True
        )

    if test_preds is not None:
        st.markdown("#### 🎯 Actual vs Predicted Prices on Indian Holdout Test Set")
        test_dates = pd.to_datetime(test_preds['Date'])
        actual_vals = test_preds['Actual']
        
        selected_models = st.multiselect(
            "Select Models to Compare:",
            options=[k for k in test_preds.keys() if k not in ['Actual', 'Date']],
            default=['Stacking_Ensemble', 'XGBoost', 'LightGBM']
        )
        
        fig_test = go.Figure()
        fig_test.add_trace(go.Scatter(x=test_dates, y=actual_vals, mode='lines', name='Actual Gold BeES (₹)', line=dict(color='#ffffff', width=2.5)))
        
        colors = {'Stacking_Ensemble': '#ffd700', 'XGBoost': '#00e676', 'LightGBM': '#29b6f6', 'Random_Forest': '#ab47bc', 'Ridge': '#ff9100', 'PyTorch_LSTM': '#ff4081'}
        
        for m_name in selected_models:
            preds_m = test_preds[m_name]
            fig_test.add_trace(go.Scatter(
                x=test_dates,
                y=preds_m,
                mode='lines',
                name=m_name,
                line=dict(color=colors.get(m_name, '#90caf9'), width=1.5, dash='dash' if m_name != 'Stacking_Ensemble' else 'solid')
            ))
            
        fig_test.update_layout(
            template='plotly_dark',
            height=460,
            xaxis_title="Date",
            yaxis_title="Nippon Gold BeES Price (₹)",
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_test, use_container_width=True)

    if feat_imp is not None:
        st.markdown("#### 🧬 Top 20 Most Influential Feature Predictors (XGBoost)")
        top_feats = feat_imp.head(20).reset_index()
        top_feats.columns = ['Feature', 'Importance']
        
        fig_imp = px.bar(
            top_feats,
            x='Importance',
            y='Feature',
            orientation='h',
            template='plotly_dark',
            color='Importance',
            color_continuous_scale='YlOrBr'
        )
        fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=480, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_imp, use_container_width=True)


# ==========================================
# TAB 3: FUTURE PRICE FORECAST (INR)
# ==========================================
with tab3:
    st.markdown("### 🔮 Multi-Horizon Indian Gold Price Projection")
    st.caption("Forecast future price trajectory for Gold BeES & Spot 24K Gold with 95% confidence intervals.")
    
    col_fc1, col_fc2 = st.columns([1, 3])
    with col_fc1:
        st.markdown("##### Forecast Settings")
        days_ahead = st.slider("Forecast Horizon (Trading Days):", min_value=1, max_value=30, value=14, step=1)
        
        forecast_df = forecast_future_prices(days_ahead=days_ahead)
        
        next_day_bees = forecast_df.iloc[0]['Predicted_GoldBeES (₹)']
        next_day_10g = forecast_df.iloc[0]['Predicted_10g_24K (₹)']
        next_day_ret = forecast_df.iloc[0]['Expected_Return (%)']
        
        final_bees = forecast_df.iloc[-1]['Predicted_GoldBeES (₹)']
        final_10g = forecast_df.iloc[-1]['Predicted_10g_24K (₹)']
        final_ret = forecast_df.iloc[-1]['Expected_Return (%)']
        
        st.metric("1-Day Gold BeES", f"₹{next_day_bees:.2f}", f"{next_day_ret:+.2f}%")
        st.metric("1-Day 24K Gold (10g)", f"₹{next_day_10g:,.0f}")
        st.metric(f"{days_ahead}-Day Target BeES", f"₹{final_bees:.2f}", f"{final_ret:+.2f}%")
        st.metric(f"{days_ahead}-Day 24K Gold (10g)", f"₹{final_10g:,.0f}")
        
    with col_fc2:
        hist_context = raw_data['GOLDBEES'].tail(60)
        future_dates = pd.to_datetime(forecast_df['Date'])
        
        fig_fore = go.Figure()
        fig_fore.add_trace(go.Scatter(
            x=hist_context.index,
            y=hist_context.values,
            mode='lines',
            name='Historical Gold BeES (Last 60 Days)',
            line=dict(color='#90caf9', width=2)
        ))
        fig_fore.add_trace(go.Scatter(
            x=future_dates,
            y=forecast_df['Predicted_GoldBeES (₹)'],
            mode='lines+markers',
            name='Projected Gold BeES (₹)',
            line=dict(color='#ffd700', width=2.5)
        ))
        fig_fore.add_trace(go.Scatter(
            x=future_dates,
            y=forecast_df['Upper_Bound_BeES (₹)'],
            mode='lines',
            line=dict(color='rgba(255,215,0,0.2)', dash='dot'),
            showlegend=False
        ))
        fig_fore.add_trace(go.Scatter(
            x=future_dates,
            y=forecast_df['Lower_Bound_BeES (₹)'],
            mode='lines',
            line=dict(color='rgba(255,215,0,0.2)', dash='dot'),
            fill='tonexty',
            fillcolor='rgba(255,215,0,0.12)',
            name='95% Confidence Interval'
        ))
        
        fig_fore.update_layout(
            template='plotly_dark',
            height=460,
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_fore, use_container_width=True)

    st.markdown("#### 📋 Detailed Day-by-Day Indian Price Projections")
    st.dataframe(
        forecast_df.style.format({
            'Predicted_GoldBeES (₹)': '₹{:.2f}',
            'Lower_Bound_BeES (₹)': '₹{:.2f}',
            'Upper_Bound_BeES (₹)': '₹{:.2f}',
            'Predicted_10g_24K (₹)': '₹{:,.0f}',
            'Lower_10g_24K (₹)': '₹{:,.0f}',
            'Upper_10g_24K (₹)': '₹{:,.0f}',
            'Expected_Return (%)': '{:+.2f}%'
        }),
        use_container_width=True
    )


# ==========================================
# TAB 4: INDIAN STRATEGY BACKTESTER
# ==========================================
with tab4:
    st.markdown("### 📊 Algorithmic Strategy Backtesting in INR (₹)")
    st.caption("Simulates quantitative trading on Nippon India ETF Gold BeES vs Buy & Hold.")
    
    col_bt1, col_bt2 = st.columns([1, 3])
    with col_bt1:
        st.markdown("##### Backtest Capital")
        capital_inr = st.number_input("Starting Capital (₹):", min_value=10000, max_value=10000000, value=100000, step=10000)
        allow_short = st.checkbox("Allow Short Position (Long/Short)", value=False)
        
        bt_results = run_strategy_backtest(allow_short=allow_short, initial_capital_inr=capital_inr)
        bt_summary = bt_results['Summary']
        
        st.metric("Strategy Final Portfolio", f"₹{bt_summary['Strategy_Final_Value (₹)']:,.2f}")
        st.metric("Benchmark Final Portfolio", f"₹{bt_summary['Benchmark_Final_Value (₹)']:,.2f}")
        st.metric("Strategy Total Return", f"{bt_summary['Strategy_Total_Return (%)']:+.2f}%")
        st.metric("Strategy Sharpe Ratio", f"{bt_summary['Strategy_Sharpe']:.2f}")
        st.metric("Strategy Max Drawdown", f"{bt_summary['Strategy_Max_Drawdown (%)']:.2f}%")
        st.metric("Trade Win Rate", f"{bt_summary['Win_Rate (%)']:.2f}%")
        
    with col_bt2:
        bt_dates = pd.to_datetime(bt_results['Equity_Curve']['Dates'])
        strat_equity = bt_results['Equity_Curve']['Strategy']
        bench_equity = bt_results['Equity_Curve']['Benchmark']
        
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(x=bt_dates, y=strat_equity, mode='lines', name='AI Trading Strategy (₹)', line=dict(color='#00e676', width=2.5)))
        fig_equity.add_trace(go.Scatter(x=bt_dates, y=bench_equity, mode='lines', name='Buy & Hold Gold BeES (₹)', line=dict(color='#90caf9', width=1.8, dash='dash')))
        
        fig_equity.update_layout(
            template='plotly_dark',
            height=460,
            xaxis_title="Date",
            yaxis_title="Portfolio Value (₹ INR)",
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_equity, use_container_width=True)


# ==========================================
# TAB 5: INDIAN MARKET DYNAMICS
# ==========================================
with tab5:
    st.markdown("""
    ### 🇮🇳 Indian Gold Market Mechanics & Macro Dynamics

    #### 1. The Domestic Price Equation
    Unlike the US where Gold is priced purely in USD per Troy Ounce, in India the price of 24K Gold per 10 grams is determined by a 3-pillar formula:
    $$\\text{Gold (₹ / 10g)} = \\left( \\frac{\\text{Spot Gold (USD/oz)} \\times \\text{USD/INR}}{31.1035} \\right) \\times 10 \\times (1 + \\text{Customs Duty} + \\text{AIDC})$$

    #### 2. Key Domestic Drivers Incorporated in our AI Models:
    - **USD/INR Exchange Rate**: India imports over 90% of its domestic gold requirement. Depreciation in the Rupee directly magnifies domestic gold prices even when international dollar gold is flat.
    - **NIFTY 50 & BSE SENSEX**: Domestic equity market sentiment affects asset allocation cycles between Indian equities and safe-haven gold.
    - **India VIX**: The National Stock Exchange volatility index captures domestic market fear and risk-off rotation into Gold BeES.
    - **Crude Oil (Brent/WTI in INR)**: Crude oil is India's largest import component, influencing trade deficit, inflation, and currency pressure.
    - **Nippon India ETF Gold BeES (`GOLDBEES.NS`)**: India's oldest and most liquid Gold ETF on the National Stock Exchange, trading in fractional units corresponding directly to physical 24K gold backing.
    """)
