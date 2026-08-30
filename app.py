"""
Gold Price Prediction & Quantitative Forecasting Dashboard.
Built with Streamlit and Plotly for high-precision financial analytics.
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

from data_loader import download_all_market_data, RAW_DATA_PATH
from features import prepare_full_features, PROCESSED_DATA_PATH
from forecast import forecast_future_prices
from backtest import run_strategy_backtest
from train import train_and_export_all_models, MODELS_DIR

# Set Page Config
st.set_page_config(
    page_title="Gold Price AI Forecaster | Live Market Intelligence",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Luxury Gold Theme
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stMetric {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 12px 18px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stMetric label { color: #8b949e !important; font-weight: 500; }
    .stMetric [data-testid="stMetricValue"] { color: #f0f6fc !important; font-size: 1.6rem !important; }
    .gold-header {
        background: linear-gradient(90deg, #ffd700, #ffb300, #ff8f00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem;
    }
    .badge-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .badge-title { color: #ffd700; font-weight: 600; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_cached_market_data():
    if os.path.exists(RAW_DATA_PATH):
        df = pd.read_csv(RAW_DATA_PATH, index_col=0, parse_dates=True)
    else:
        df = download_all_market_data()
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


# --- App Header & Navigation ---
st.markdown('<h1 class="gold-header">🪙 AI-Powered Gold Price Forecasting & Analytics</h1>', unsafe_allow_html=True)
st.markdown("Live Multi-Asset Market Intelligence • Non-Lookahead ML & Deep Learning • Multi-Horizon Projections")

# Load data
raw_data = load_cached_market_data()
feat_data = load_cached_features()
metrics, test_preds, cv_df, feat_imp = load_test_artifacts()

latest_price = float(raw_data['GLD'].iloc[-1])
prev_price = float(raw_data['GLD'].iloc[-2])
day_change = latest_price - prev_price
day_pct = (day_change / prev_price) * 100.0
latest_date = raw_data.index[-1].strftime('%B %d, %Y')

# --- Sidebar Controls ---
st.sidebar.markdown("### 📊 Market Snapshot")
st.sidebar.metric(
    label=f"GLD Price ({latest_date})",
    value=f"${latest_price:.2f}",
    delta=f"{day_change:+.2f} ({day_pct:+.2f}%)"
)

if 'GC_F' in raw_data.columns:
    st.sidebar.metric(label="Gold Futures (GC=F)", value=f"${raw_data['GC_F'].iloc[-1]:.2f}")
if 'DXY' in raw_data.columns:
    st.sidebar.metric(label="US Dollar Index (DXY)", value=f"{raw_data['DXY'].iloc[-1]:.2f}")
if 'TNX' in raw_data.columns:
    st.sidebar.metric(label="10Y Treasury Yield", value=f"{raw_data['TNX'].iloc[-1]:.2f}%")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Pipeline Control")

if st.sidebar.button("🔄 Refresh Live Data (Yahoo Finance)", use_container_width=True):
    with st.spinner("Downloading latest multi-asset market data..."):
        download_all_market_data(force_refresh=True)
        prepare_full_features(load_cached_market_data(), save=True)
        st.cache_data.clear()
        st.success("Market data updated successfully!")
        st.rerun()

if st.sidebar.button("🧠 Retrain All ML & DL Models", use_container_width=True):
    with st.spinner("Training Random Forest, XGBoost, LightGBM, LSTM, and Stacking Ensemble..."):
        train_and_export_all_models()
        st.cache_data.clear()
        st.success("All models retrained and serialized!")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Active Stack")
st.sidebar.caption("• Framework: PyTorch, XGBoost, LightGBM, Scikit-Learn\n• Data Source: Yahoo Finance API (2008-Present)\n• Validation: Walk-Forward Expanding TimeSeriesSplit")

# --- Top Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Market & Technicals",
    "🤖 Model Arena & Performance",
    "🔮 Future Price Forecast",
    "📊 Strategy Backtest",
    "📚 Methodology & Insights"
])

# ==========================================
# TAB 1: MARKET & TECHNICAL ANALYSIS
# ==========================================
with tab1:
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("50-Day Moving Avg", f"${feat_data['SMA_50'].iloc[-1]:.2f}")
    with col_m2:
        st.metric("200-Day Moving Avg", f"${feat_data['SMA_200'].iloc[-1]:.2f}")
    with col_m3:
        rsi_val = feat_data['RSI_14'].iloc[-1]
        rsi_state = "Overbought (>70)" if rsi_val > 70 else ("Oversold (<30)" if rsi_val < 30 else "Neutral")
        st.metric("14-Day RSI", f"{rsi_val:.1f}", rsi_state)
    with col_m4:
        st.metric("Gold / Silver Ratio", f"{feat_data['Gold_Silver_Ratio'].iloc[-1]:.2f}" if 'Gold_Silver_Ratio' in feat_data else "N/A")

    st.markdown("#### 🕯️ Interactive Gold Price & Technical Indicators")
    time_range = st.radio("Timeframe Range:", ["1 Year", "3 Years", "5 Years", "All Time (2008-Present)"], horizontal=True, index=0)
    
    if time_range == "1 Year":
        plot_df = feat_data.tail(252)
    elif time_range == "3 Years":
        plot_df = feat_data.tail(252 * 3)
    elif time_range == "5 Years":
        plot_df = feat_data.tail(252 * 5)
    else:
        plot_df = feat_data

    fig_price = go.Figure()
    
    # Check if OHLC is present
    if 'GLD_Open' in plot_df.columns:
        fig_price.add_trace(go.Candlestick(
            x=plot_df.index,
            open=plot_df['GLD_Open'],
            high=plot_df['GLD_High'],
            low=plot_df['GLD_Low'],
            close=plot_df['GLD_Close'],
            name="GLD OHLC",
            increasing_line_color='#00e676',
            decreasing_line_color='#ff5252'
        ))
    else:
        fig_price.add_trace(go.Scatter(x=plot_df.index, y=plot_df['GLD'], mode='lines', name='GLD Price', line=dict(color='#ffd700', width=2)))

    # Technical overlays
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
        st.markdown("##### 🌊 MACD (Moving Average Convergence Divergence)")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MACD'], line=dict(color='#29b6f6', width=1.5), name='MACD'))
        fig_macd.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MACD_Signal'], line=dict(color='#ff9100', width=1.5), name='Signal'))
        fig_macd.add_trace(go.Bar(x=plot_df.index, y=plot_df['MACD_Diff'], name='Histogram', marker_color='#80cbc4'))
        fig_macd.update_layout(template='plotly_dark', height=240, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_macd, use_container_width=True)

    # Cross-Asset Correlation Heatmap
    st.markdown("#### 🔥 Multi-Asset Intermarket Correlation Matrix")
    corr_assets = ['GLD', 'SPX', 'USO', 'SLV', 'EURUSD', 'DXY', 'TNX', 'VIX', 'TIP']
    available_corr = [c for c in corr_assets if c in raw_data.columns]
    corr_matrix = raw_data[available_corr].corr().round(3)
    
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Viridis",
        template="plotly_dark",
        title="Asset Correlation Heatmap (2008 - Present)"
    )
    fig_corr.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)


# ==========================================
# TAB 2: MODEL PERFORMANCE & BENCHMARK ARENA
# ==========================================
with tab2:
    st.markdown("### 🏆 Model Comparison & Walk-Forward Holdout Evaluation")
    st.caption("Evaluated on strict chronologically isolated holdout test data (most recent 15% trading days).")

    if metrics is not None:
        metrics_df = pd.DataFrame(metrics).T
        st.dataframe(
            metrics_df.style.format({
                'RMSE': '{:.3f}',
                'MAE': '{:.3f}',
                'MAPE (%)': '{:.2f}%',
                'R2_Score': '{:.4f}',
                'Directional_Accuracy (%)': '{:.2f}%'
            }).highlight_min(subset=['RMSE', 'MAE', 'MAPE (%)'], color='#1b4d3e')
              .highlight_max(subset=['R2_Score', 'Directional_Accuracy (%)'], color='#1b4d3e'),
            use_container_width=True
        )

    if test_preds is not None:
        st.markdown("#### 🎯 Actual vs Predicted Prices on Holdout Test Period")
        test_dates = pd.to_datetime(test_preds['Date'])
        actual_vals = test_preds['Actual']
        
        selected_models = st.multiselect(
            "Select Models to Display:",
            options=[k for k in test_preds.keys() if k not in ['Actual', 'Date']],
            default=['Stacking_Ensemble', 'XGBoost', 'LightGBM']
        )
        
        fig_test = go.Figure()
        fig_test.add_trace(go.Scatter(x=test_dates, y=actual_vals, mode='lines', name='Actual GLD Price', line=dict(color='#ffffff', width=2.5)))
        
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
            yaxis_title="GLD ETF Price ($)",
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_test, use_container_width=True)

    # Feature Importance
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
# TAB 3: MULTI-STEP FUTURE PRICE FORECAST
# ==========================================
with tab3:
    st.markdown("### 🔮 Multi-Horizon Future Price Projection")
    st.caption("Forecast future gold price path with volatility-expanded 95% confidence intervals.")
    
    col_fc1, col_fc2 = st.columns([1, 3])
    with col_fc1:
        st.markdown("##### Forecast Configuration")
        days_ahead = st.slider("Forecasting Horizon (Business Days):", min_value=1, max_value=30, value=14, step=1)
        
        forecast_df = forecast_future_prices(days_ahead=days_ahead)
        
        next_day_price = forecast_df.iloc[0]['Predicted_GLD']
        next_day_ret = forecast_df.iloc[0]['Expected_Return (%)']
        final_day_price = forecast_df.iloc[-1]['Predicted_GLD']
        final_day_ret = forecast_df.iloc[-1]['Expected_Return (%)']
        
        st.metric("1-Day Ahead Forecast", f"${next_day_price:.2f}", f"{next_day_ret:+.2f}%")
        st.metric(f"{days_ahead}-Day Target Price", f"${final_day_price:.2f}", f"{final_day_ret:+.2f}%")
        
    with col_fc2:
        # Chart historical context + future projection
        hist_context = raw_data['GLD'].tail(60)
        
        fig_fore = go.Figure()
        fig_fore.add_trace(go.Scatter(
            x=hist_context.index,
            y=hist_context.values,
            mode='lines',
            name='Historical Price (Last 60 Days)',
            line=dict(color='#90caf9', width=2)
        ))
        
        # Future predicted line
        future_dates = pd.to_datetime(forecast_df['Date'])
        fig_fore.add_trace(go.Scatter(
            x=future_dates,
            y=forecast_df['Predicted_GLD'],
            mode='lines+markers',
            name='AI Projected Price',
            line=dict(color='#ffd700', width=2.5)
        ))
        
        # 95% Confidence Bounds
        fig_fore.add_trace(go.Scatter(
            x=future_dates,
            y=forecast_df['Upper_Bound_95'],
            mode='lines',
            line=dict(color='rgba(255,215,0,0.2)', dash='dot'),
            name='95% Upper Bound',
            showlegend=False
        ))
        fig_fore.add_trace(go.Scatter(
            x=future_dates,
            y=forecast_df['Lower_Bound_95'],
            mode='lines',
            line=dict(color='rgba(255,215,0,0.2)', dash='dot'),
            fill='tonexty',
            fillcolor='rgba(255,215,0,0.12)',
            name='95% Confidence Band'
        ))
        
        fig_fore.update_layout(
            template='plotly_dark',
            height=450,
            xaxis_title="Date",
            yaxis_title="GLD ETF Price ($)",
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_fore, use_container_width=True)

    st.markdown("#### 📋 Detailed Day-by-Day Forecast Projections")
    st.dataframe(
        forecast_df.style.format({
            'Predicted_GLD': '${:.2f}',
            'Lower_Bound_95': '${:.2f}',
            'Upper_Bound_95': '${:.2f}',
            'Expected_Return (%)': '{:+.2f}%'
        }),
        use_container_width=True
    )


# ==========================================
# TAB 4: STRATEGY BACKTESTER
# ==========================================
with tab4:
    st.markdown("### 📊 Algorithmic Strategy Backtesting Simulator")
    st.caption("Simulates trading execution based on AI model directional signals on holdout data vs Buy & Hold.")
    
    col_bt1, col_bt2 = st.columns([1, 3])
    with col_bt1:
        st.markdown("##### Backtest Settings")
        capital = st.number_input("Initial Capital ($):", min_value=1000, max_value=1000000, value=10000, step=1000)
        allow_short = st.checkbox("Allow Short Selling (Long/Short)", value=False)
        
        bt_results = run_strategy_backtest(allow_short=allow_short, initial_capital=capital)
        bt_summary = bt_results['Summary']
        
        st.metric("Strategy Total Return", f"{bt_summary['Strategy_Total_Return (%)']:+.2f}%")
        st.metric("Benchmark Total Return", f"{bt_summary['Benchmark_Total_Return (%)']:+.2f}%")
        st.metric("Strategy Sharpe Ratio", f"{bt_summary['Strategy_Sharpe']:.2f}")
        st.metric("Max Drawdown", f"{bt_summary['Strategy_Max_Drawdown (%)']:.2f}%")
        st.metric("Trade Win Rate", f"{bt_summary['Win_Rate (%)']:.2f}%")
        
    with col_bt2:
        bt_dates = pd.to_datetime(bt_results['Equity_Curve']['Dates'])
        strat_equity = bt_results['Equity_Curve']['Strategy']
        bench_equity = bt_results['Equity_Curve']['Benchmark']
        
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(x=bt_dates, y=strat_equity, mode='lines', name='AI Strategy Equity', line=dict(color='#00e676', width=2.5)))
        fig_equity.add_trace(go.Scatter(x=bt_dates, y=bench_equity, mode='lines', name='Buy & Hold Benchmark', line=dict(color='#90caf9', width=1.8, dash='dash')))
        
        fig_equity.update_layout(
            template='plotly_dark',
            height=460,
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_equity, use_container_width=True)


# ==========================================
# TAB 5: METHODOLOGY & QUANT RESEARCH
# ==========================================
with tab5:
    st.markdown("""
    ### 📖 Quantitative Architecture & Technical Insights

    #### 1. Why Classical Kaggle Notebooks Fail in Real Trading
    - **Lookahead Bias via Random Shuffling**: Standard `train_test_split(shuffle=True)` leaks future prices into past data. In our platform, we enforce strict chronological splitting and **Walk-Forward Expanding Window Cross-Validation (`TimeSeriesSplit`)**.
    - **Contemporaneous vs. True Forecasting**: Predicting today's Gold price from today's S&P 500 close is impossible in live trading. We use strict $t, t-1, \dots, t-k$ lag features so that all information is strictly available before market close.
    - **Tree Model Non-Stationarity**: Standard decision trees cannot extrapolate prices higher than the maximum historical price seen during training. We implemented a **`ResidualPricePredictor`** that models daily price delta ($\Delta P_{t+1}$) and percentage returns, enabling tree models (XGBoost, LightGBM, Random Forest) to forecast all-time highs smoothly with $R^2 > 0.99$.

    #### 2. Multi-Asset Macro Drivers
    Gold price movements are heavily governed by macroeconomic relationships:
    - **Real Yields & 10Y Treasuries (`^TNX`, `TIP`)**: When real bond yields fall, non-yielding Gold becomes more attractive.
    - **US Dollar Index (`DXY`)**: Gold is priced in USD and exhibits a strong inverse correlation with the Dollar.
    - **Market Fear & Volatility (`^VIX`)**: Spikes in geopolitical uncertainty drive safe-haven inflows into Gold.
    - **Commodities & Precious Metals (`USO`, `SLV`)**: Intermarket ratios like Gold/Silver and Gold/Oil capture macroeconomic inflation and industrial demand cycles.
    """)
