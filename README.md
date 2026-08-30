# 🪙 AI-Powered Gold Price Forecasting & Quantitative Intelligence Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)

A modern, production-grade end-to-end Machine Learning & Deep Learning system for Gold Price (`GLD`, `GC=F`) forecasting. Built with real-time multi-asset market data from 2008 to present, zero-lookahead financial feature engineering, walk-forward cross-validation, multi-model benchmarking (LightGBM, XGBoost, Random Forest, PyTorch LSTM, Stacking Ensemble), multi-step future projections with 95% confidence intervals, and an interactive Streamlit analytics dashboard.

---

## 🌟 Key Highlights & Modern Advancements

| Feature | Legacy Kaggle Approach (Old) | Our Modern Architecture |
| :--- | :--- | :--- |
| **Data Scope** | Static CSV ending in May 2018 (8+ yrs outdated) | **Live Yahoo Finance API (2008 – Present)** |
| **Macro Indicators** | 4 basic assets (SPX, USO, SLV, EUR/USD) | **Multi-Asset Suite: DXY (USD Index), 10Y Yields (`^TNX`), VIX, TIPS Bonds (`TIP`), Gold Futures (`GC=F`)** |
| **Data Leakage** | Shuffled `train_test_split` (leaks future into past) | **Strict Chronological Expanding Window Walk-Forward Validation (`TimeSeriesSplit`)** |
| **Target Formulation** | Same-day contemporaneous regression | **Non-Lookahead Next-Day ($t+1$) & Multi-Horizon ($t+N$) Forecasting** |
| **Model Extrapolation** | Tree models fail on all-time highs ($400+) | **`ResidualPricePredictor` Delta Modeling ($R^2 > 0.995$)** |
| **Model Diversity** | Single basic Random Forest | **LightGBM, XGBoost, Random Forest, Ridge, PyTorch LSTM, Stacking Ensemble** |
| **User Interface** | Static notebook | **Interactive Streamlit Web Dashboard + Plotly Visual Analytics** |

---

## 📁 Repository Structure

```
Gold_Price_Pred/
├── app.py                     # Interactive Streamlit Web Dashboard
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive documentation
├── gold_price.ipynb           # Modernized Jupyter Notebook
├── data/
│   ├── raw/                   # Auto-fetched multi-asset live market dataset
│   └── processed/             # Cleaned feature engineering matrix (105+ features)
├── models/                    # Trained model artifacts (.joblib, .pt, metrics JSONs)
│   ├── cv_metrics.csv
│   ├── test_metrics.json
│   ├── test_predictions.json
│   ├── feature_importance.csv
│   ├── lightgbm_model.joblib
│   ├── xgboost_model.joblib
│   ├── random_forest_model.joblib
│   ├── ridge_model.joblib
│   ├── stacking_ensemble.joblib
│   └── pytorch_lstm.pt
├── notebooks/
│   └── gold_price_analysis.ipynb # Step-by-step EDA, Feature Engineering & Modeling
└── src/
    ├── __init__.py
    ├── data_loader.py         # Yahoo Finance live downloader + caching
    ├── features.py            # Financial feature engineering (RSI, MACD, BB, Lags)
    ├── models.py              # ML architectures, PyTorch LSTM, Stacking Ensemble
    ├── train.py               # Walk-forward cross validation & training pipeline
    ├── forecast.py            # Multi-step future horizon projection engine
    ├── backtest.py            # Algorithmic trading strategy backtester
    └── generate_notebook.py   # Notebook builder
```

---

## 🚀 Quick Start & Installation

### 1. Clone & Set Up Virtual Environment
```bash
# Clone repository
git clone https://github.com/Priyanshu6926/Gold_Price_Prediction.git
cd Gold_Price_Prediction

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Interactive Streamlit Web Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to view live market charts, evaluate models, project future prices, and run strategy backtests.

---

## 🧠 Model Benchmarking & Performance

Evaluated on strict chronologically isolated holdout test data (most recent 15% trading days, capturing modern market regimes up to 2026):

| Model | RMSE ($) | MAE ($) | MAPE (%) | $R^2$ Score | Directional Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM Regressor** | **5.286** | **3.470** | **0.95%** | **0.9961** | **43.52%** |
| **Ridge Regressor** | 5.292 | 3.451 | 0.94% | 0.9961 | **46.08%** |
| **Stacking Ensemble** | **5.496** | **3.753** | **1.02%** | **0.9958** | **43.22%** |
| **XGBoost Regressor** | 5.695 | 3.997 | 1.10% | 0.9955 | 43.07% |
| **PyTorch LSTM** | 5.719 | 3.958 | 1.22% | 0.9953 | 43.17% |
| **Random Forest Regressor** | 5.937 | 4.337 | 1.20% | 0.9951 | 43.37% |

---

## 🛠️ Module Usage Guide

### 1. Download Latest Live Market Data
```bash
python src/data_loader.py
```

### 2. Generate Technical Indicators & Features
```bash
python src/features.py
```

### 3. Run Walk-Forward Validation & Train Models
```bash
python src/train.py
```

### 4. Forecast Future Prices (1 to 30 Days Ahead)
```bash
python src/forecast.py
```

### 5. Run Algorithmic Strategy Backtest
```bash
python src/backtest.py
```

---

## 📊 Technical Indicators & Engineered Features
- **Trend Indicators**: SMA (7, 21, 50, 200), EMA (12, 26), Golden / Death Cross flags.
- **Momentum Indicators**: Relative Strength Index (RSI 14, RSI 7), MACD & Signal Line, Stochastic Oscillator (%K, %D).
- **Volatility Envelopes**: Bollinger Bands (Upper, Lower, Width, %B), Average True Range (ATR 14), 7d/21d/90d Rolling Volatility.
- **Inter-Market Ratios**: Gold/Silver Ratio, Gold/Oil Ratio, Gold/S&P 500 Ratio, Real Yield Proxy (10Y Yield minus TIPS return).
- **Non-Lookahead Lags**: Multi-day price and return lag structures ($t-1, t-2, t-3, t-5, t-10$).

---

## 📄 License
This project is licensed under the MIT License.
