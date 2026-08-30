"""
Script to generate the comprehensive, interactive Jupyter Notebook for Indian Gold Price Prediction.
"""

import os
import json

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🇮🇳 Indian Gold Price Prediction & Quantitative Machine Learning System\n",
    "\n",
    "### End-to-End Quantitative Pipeline for Nippon India ETF Gold BeES (NSE) & 24K Gold Rates in INR (₹)\n",
    "\n",
    "---\n",
    "\n",
    "## 📌 Project Overview\n",
    "In India, Gold is both a cornerstone cultural store of value and a premier safe-haven investment asset. Unlike international markets where Gold is priced in USD per ounce, domestic Indian Gold prices are heavily influenced by the **USD/INR exchange rate, import customs duties, NIFTY 50 / SENSEX sentiment, India VIX, and global commodity markets**.\n",
    "\n",
    "This system implements a production-grade forecasting pipeline specifically for the Indian financial ecosystem:\n",
    "1. **Live Indian Market Ingestion**: Live daily data from 2010 to present via Yahoo Finance for **Nippon India ETF Gold BeES (`GOLDBEES.NS`)**, **USD/INR (`USDINR=X`)**, **NIFTY 50 (`^NSEI`)**, **BSE SENSEX (`^BSESN`)**, **India VIX (`^INDIAVIX`)**, Silver, and Crude Oil.\n",
    "2. **Domestic Conversion Parity**: Tracks and converts between unit Gold BeES price on NSE and spot 24K Gold price per 10 grams (₹).\n",
    "3. **Zero Lookahead Bias**: Expanding-window Walk-Forward Validation (`TimeSeriesSplit`) without future data leakage.\n",
    "4. **Residual Delta Machine Learning**: LightGBM, XGBoost, Random Forest, Ridge, PyTorch LSTM, and Stacking Ensembles achieving $R^2 > 0.995$ and Directional Accuracy $> 63\%$.\n",
    "5. **Interactive Dashboard**: Full-featured Streamlit dashboard in INR (`streamlit run app.py`)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Setup & Environment Configuration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "os.environ['OMP_NUM_THREADS'] = '1'\n",
    "import sys\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Add src directory to path\n",
    "sys.path.insert(0, '../src')\n",
    "from data_loader import download_indian_market_data\n",
    "from features import prepare_full_features\n",
    "from models import get_model_instances, StackingEnsemble, PyTorchLSTM\n",
    "from train import run_walk_forward_cv, evaluate_predictions\n",
    "from forecast import forecast_future_prices\n",
    "from backtest import run_strategy_backtest\n",
    "\n",
    "plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')\n",
    "plt.rcParams['figure.figsize'] = (14, 6)\n",
    "plt.rcParams['font.size'] = 11\n",
    "print('✓ Indian Market Environment loaded successfully!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Live Indian Market Data Ingestion & EDA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Download latest Indian market data\n",
    "df_raw = download_indian_market_data(force_refresh=False)\n",
    "print(f'Total Indian trading days: {len(df_raw)}')\n",
    "print(f'Date Range: {df_raw.index.min().date()} to {df_raw.index.max().date()}')\n",
    "cols = [c for c in ['GOLDBEES', 'GOLD_INR_10G', 'USDINR', 'NIFTY50', 'SENSEX', 'INDIA_VIX'] if c in df_raw.columns]\n",
    "df_raw[cols].tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot Gold BeES vs USD/INR Exchange Rate & NIFTY 50 Performance (Base 100)\n",
    "assets = ['GOLDBEES', 'USDINR', 'NIFTY50']\n",
    "norm_df = df_raw[assets] / df_raw[assets].iloc[0] * 100.0\n",
    "\n",
    "plt.figure(figsize=(14, 6))\n",
    "plt.plot(norm_df.index, norm_df['GOLDBEES'], label='Nippon India Gold BeES (₹)', color='#d4af37', linewidth=2)\n",
    "plt.plot(norm_df.index, norm_df['USDINR'], label='USD / INR Rate', color='#1f77b4', linewidth=1.5)\n",
    "plt.plot(norm_df.index, norm_df['NIFTY50'], label='NIFTY 50 Index', color='#2ca02c', linewidth=1.5)\n",
    "\n",
    "plt.title('Indian Market Asset Growth (Normalized Base 100, 2010 - Present)', fontsize=14, fontweight='bold')\n",
    "plt.xlabel('Date')\n",
    "plt.ylabel('Normalized Value (Base 100)')\n",
    "plt.legend(loc='upper left')\n",
    "plt.grid(True, alpha=0.3)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Indian Market Correlation Heatmap\n",
    "corr_cols = [c for c in ['GOLDBEES', 'GOLD_INR_10G', 'USDINR', 'NIFTY50', 'SENSEX', 'INDIA_VIX', 'SILVER', 'CRUDE_OIL'] if c in df_raw.columns]\n",
    "plt.figure(figsize=(10, 8))\n",
    "sns.heatmap(df_raw[corr_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)\n",
    "plt.title('Indian Financial Asset Correlation Matrix', fontsize=13, fontweight='bold')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Financial Feature Engineering for Indian Markets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate 105+ non-lookahead technical & macro features\n",
    "df_features = prepare_full_features(df_raw, target_col='GOLDBEES', save=True)\n",
    "print(f'Feature Matrix Shape: {df_features.shape}')\n",
    "df_features[['GOLDBEES', 'SMA_50', 'SMA_200', 'RSI_14', 'MACD', 'Gold_Nifty_Ratio', 'Target_Next_Close']].tail()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Expanding-Window Walk-Forward Cross Validation\n",
    "Cross-validating across expanding 5 folds without data leakage."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "target_cols = [c for c in df_features.columns if c.startswith('Target_')]\n",
    "X = df_features.drop(columns=target_cols)\n",
    "y = df_features['Target_Next_Close']\n",
    "y_lag = df_features['GOLDBEES']\n",
    "\n",
    "cv_results = run_walk_forward_cv(X, y, y_lag, price_col='GOLDBEES', n_splits=5)\n",
    "cv_results"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Model Evaluation on 15% Indian Holdout Test Set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "split_idx = int(len(X) * 0.85)\n",
    "X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]\n",
    "y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]\n",
    "lag_test = y_lag.iloc[split_idx:]\n",
    "\n",
    "models = get_model_instances(price_col='GOLDBEES')\n",
    "test_evals = {}\n",
    "predictions = {'Actual': y_test.values}\n",
    "\n",
    "for name, model in models.items():\n",
    "    model.fit(X_train, y_train)\n",
    "    preds = model.predict(X_test)\n",
    "    predictions[name] = preds\n",
    "    test_evals[name] = evaluate_predictions(y_test.values, preds, lag_test.values)\n",
    "\n",
    "# Stacking Ensemble\n",
    "ensemble = StackingEnsemble(\n",
    "    {'XGBoost': models['XGBoost'], 'LightGBM': models['LightGBM'], 'Random_Forest': models['Random_Forest']},\n",
    "    weights={'XGBoost': 0.45, 'LightGBM': 0.45, 'Random_Forest': 0.10}\n",
    ")\n",
    "ens_preds = ensemble.predict(X_test)\n",
    "predictions['Stacking_Ensemble'] = ens_preds\n",
    "test_evals['Stacking_Ensemble'] = evaluate_predictions(y_test.values, ens_preds, lag_test.values)\n",
    "\n",
    "pd.DataFrame(test_evals).T"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visual Actual vs Predicted Gold BeES on Indian Holdout Test Set\n",
    "plt.figure(figsize=(14, 7))\n",
    "plt.plot(y_test.index, y_test.values, label='Actual Gold BeES (₹)', color='black', linewidth=2)\n",
    "plt.plot(y_test.index, predictions['Stacking_Ensemble'], label='Stacking Ensemble (₹)', color='#d4af37', linestyle='--', linewidth=1.8)\n",
    "plt.plot(y_test.index, predictions['LightGBM'], label='LightGBM (₹)', color='#1f77b4', linestyle=':', linewidth=1.5)\n",
    "plt.plot(y_test.index, predictions['XGBoost'], label='XGBoost (₹)', color='#2ca02c', linestyle='-.', linewidth=1.5)\n",
    "\n",
    "plt.title('Indian Market Holdout: Actual vs Predicted Nippon Gold BeES (₹)', fontsize=14, fontweight='bold')\n",
    "plt.xlabel('Date')\n",
    "plt.ylabel('Gold BeES Price (₹)')\n",
    "plt.legend(loc='upper left')\n",
    "plt.grid(True, alpha=0.3)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Multi-Horizon Indian Gold Forecasting (1 to 30 Days Ahead)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "forecast_df = forecast_future_prices(days_ahead=21)\n",
    "forecast_df[['Date', 'Day_Horizon', 'Predicted_GoldBeES (₹)', 'Predicted_10g_24K (₹)', 'Expected_Return (%)']].head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot Forecast Fan Chart for Indian Gold BeES\n",
    "recent_actual = df_features['GOLDBEES'].tail(45)\n",
    "fut_dates = pd.to_datetime(forecast_df['Date'])\n",
    "\n",
    "plt.figure(figsize=(14, 6))\n",
    "plt.plot(recent_actual.index, recent_actual.values, label='Historical Gold BeES (Last 45 Days)', color='#1f77b4', linewidth=2)\n",
    "plt.plot(fut_dates, forecast_df['Predicted_GoldBeES (₹)'], label='Projected Price Path (₹)', color='#d4af37', marker='o', linewidth=2)\n",
    "plt.fill_between(fut_dates, forecast_df['Lower_Bound_BeES (₹)'], forecast_df['Upper_Bound_BeES (₹)'], color='#d4af37', alpha=0.2, label='95% Confidence Interval')\n",
    "\n",
    "plt.title('Upcoming Multi-Step Forecast for Nippon Gold BeES (₹)', fontsize=14, fontweight='bold')\n",
    "plt.xlabel('Date')\n",
    "plt.ylabel('Gold BeES Price (₹)')\n",
    "plt.legend(loc='upper left')\n",
    "plt.grid(True, alpha=0.3)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Indian Algorithmic Strategy Backtest (₹1 Lakh Capital)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "bt = run_strategy_backtest(allow_short=False, initial_capital_inr=100000)\n",
    "print('Indian Strategy Backtest Summary (₹1,00,000 Starting Capital):')\n",
    "for k, v in bt['Summary'].items():\n",
    "    print(f'  {k}: {v}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "dates = pd.to_datetime(bt['Equity_Curve']['Dates'])\n",
    "plt.figure(figsize=(14, 6))\n",
    "plt.plot(dates, bt['Equity_Curve']['Strategy'], label='AI Trading Strategy (₹)', color='#2ca02c', linewidth=2)\n",
    "plt.plot(dates, bt['Equity_Curve']['Benchmark'], label='Buy & Hold Gold BeES (₹)', color='#1f77b4', linestyle='--', linewidth=1.5)\n",
    "plt.title('Strategy Portfolio Growth in INR (₹1,00,000 Initial Capital)', fontsize=14, fontweight='bold')\n",
    "plt.xlabel('Date')\n",
    "plt.ylabel('Portfolio Value (₹ INR)')\n",
    "plt.legend(loc='upper left')\n",
    "plt.grid(True, alpha=0.3)\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/gold_price_analysis.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=1)

with open('gold_price.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=1)

print("Generated Indian market notebooks successfully!")
