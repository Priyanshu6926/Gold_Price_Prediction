# 🇮🇳 Indian Gold Price Prediction & AI Forecasting Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![NSE India](https://img.shields.io/badge/Market-NSE%20%7C%20BSE-orange.svg)](https://www.nseindia.com/)

A modern, production-grade Quantitative Machine Learning & Deep Learning system designed specifically for the **Indian Gold Market**. The platform models **Nippon India ETF Gold BeES (`GOLDBEES.NS`)** on the National Stock Exchange (NSE) and domestic **24K Spot Gold per 10 grams (₹)**, driven by key Indian macroeconomic indicators including **USD/INR exchange rate**, **NIFTY 50**, **BSE SENSEX**, **India VIX**, and international commodity parity.

---

## 🌟 Indian Market Architecture & Innovations

| Pillar | Implementation for Indian Markets |
| :--- | :--- |
| **Target Assets** | **Nippon India ETF Gold BeES (`GOLDBEES.NS`)** on NSE & **24K Gold Rate (₹ / 10 Grams)** |
| **Domestic Macro Drivers** | **USD/INR (`USDINR=X`)**, **NIFTY 50 (`^NSEI`)**, **BSE SENSEX (`^BSESN`)**, **India VIX (`^INDIAVIX`)** |
| **Commodity Drivers** | International Gold Futures (`GC=F`), Silver (`SI=F`), Crude Oil (`CL=F`), US Dollar Index (`DXY`) |
| **Validation Framework** | **5-Fold Expanding Window Walk-Forward Time-Series Split** (Zero lookahead leakage) |
| **Extrapolation Engine** | **`ResidualPricePredictor` Delta Modeling** ($R^2 > 0.995$, Directional Accuracy $> 63\%$) |
| **Interactive Dashboard** | **Streamlit Indian Financial Dashboard (₹)** with interactive Plotly analytics |

---

## 📁 Repository Structure

```
Gold_Price_Pred/
├── app.py                     # Interactive Streamlit Web Dashboard (INR)
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive Indian Market documentation
├── gold_price.ipynb           # Modernized Indian Gold Jupyter Notebook
├── data/
│   ├── raw/                   # Live Indian & Global market data
│   └── processed/             # Cleaned feature engineering matrix (107+ features)
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
│   └── gold_price_analysis.ipynb # Comprehensive EDA, Feature Engineering & Modeling
└── src/
    ├── __init__.py
    ├── data_loader.py         # Live downloader for NSE/BSE/Yahoo Finance
    ├── features.py            # Indian technical indicators, NIFTY & USD/INR ratios
    ├── models.py              # ML architectures, PyTorch LSTM, Stacking Ensemble
    ├── train.py               # Walk-forward cross validation on Indian data
    ├── forecast.py            # Multi-step future projection engine (₹ / unit & ₹ / 10g)
    ├── backtest.py            # Algorithmic trading backtester (₹ INR Capital)
    └── generate_notebook.py   # Notebook builder
```

---

## 🚀 Quick Start & Installation

### 1. Set Up Virtual Environment & Dependencies
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Interactive Indian Streamlit Web Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to view live Indian market charts, predict future Gold BeES and 10g 24K Gold prices, and run strategy backtests.

---

## 🧠 Model Benchmarking on Indian Market Holdout Test Set

Evaluated on chronologically isolated Indian market holdout test data (most recent 15% trading days):

| Model | RMSE (₹) | MAE (₹) | MAPE (%) | $R^2$ Score | Directional Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **1.691** | **0.879** | **0.89%** | **0.9956** | **63.03%** |
| **LightGBM Regressor** | 1.720 | 0.917 | 0.91% | 0.9954 | **64.40%** |
| **Stacking Ensemble** | 1.724 | 0.904 | 0.90% | 0.9954 | **63.03%** |
| **PyTorch LSTM** | 1.795 | 0.987 | 0.99% | 0.9950 | 52.38% |
| **XGBoost Regressor** | 1.807 | 0.919 | 0.92% | 0.9949 | **63.88%** |
| **Ridge Regressor** | 2.599 | 1.890 | 1.94% | 0.9896 | 55.71% |

*Note: Metrics measured on Nippon India ETF Gold BeES (₹ per unit).*

---

## 🛠️ CLI Pipeline Usage

```bash
# 1. Fetch Latest Indian Market Data (GOLDBEES, USDINR, NIFTY 50, SENSEX, VIX)
python src/data_loader.py

# 2. Compute Domestic Technical & Macro Features
python src/features.py

# 3. Train All Models with Walk-Forward Cross Validation
python src/train.py

# 4. Generate Future Forecast (1 to 30 Days Ahead in ₹)
python src/forecast.py

# 5. Run Algorithmic Strategy Backtest in INR (₹)
python src/backtest.py
```

---

## 📄 License
This project is licensed under the MIT License.
