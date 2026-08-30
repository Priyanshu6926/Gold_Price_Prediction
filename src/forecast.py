"""
Multi-Step Future Forecasting Engine for Indian Gold Markets.
Forecasts future prices for Nippon India ETF Gold BeES (₹/unit) and Domestic Spot Gold (₹/10g)
from 1 to 30 days ahead with 95% confidence intervals.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
if CURR_DIR not in sys.path:
    sys.path.insert(0, CURR_DIR)

from features import PROCESSED_DATA_PATH, prepare_full_features
from data_loader import download_indian_market_data
from models import ResidualPricePredictor, StackingEnsemble

MODELS_DIR = os.path.join(os.path.dirname(CURR_DIR), 'models')


def load_forecasting_pipeline():
    """
    Loads saved model and feature names.
    """
    ensemble_path = os.path.join(MODELS_DIR, 'stacking_ensemble.joblib')
    xgb_path = os.path.join(MODELS_DIR, 'xgboost_model.joblib')
    features_path = os.path.join(MODELS_DIR, 'feature_names.json')
    
    if os.path.exists(ensemble_path):
        model = joblib.load(ensemble_path)
    elif os.path.exists(xgb_path):
        model = joblib.load(xgb_path)
    else:
        raise FileNotFoundError("Trained models not found. Please run train.py first.")
        
    with open(features_path, 'r') as f:
        feature_names = json.load(f)
        
    return model, feature_names


def generate_future_trading_days(start_date: pd.Timestamp, num_days: int = 30) -> list:
    """
    Generates next upcoming business days for Indian trading sessions (Mon-Fri).
    """
    future_dates = []
    curr = start_date + timedelta(days=1)
    while len(future_dates) < num_days:
        if curr.weekday() < 5:
            future_dates.append(curr)
        curr += timedelta(days=1)
    return future_dates


def forecast_future_prices(days_ahead: int = 30) -> pd.DataFrame:
    """
    Generates multi-day future Indian gold price predictions with confidence intervals.
    """
    model, feature_names = load_forecasting_pipeline()
    
    if os.path.exists(PROCESSED_DATA_PATH):
        df = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
    else:
        raw = download_indian_market_data()
        df = prepare_full_features(raw)
        
    latest_row = df.iloc[-1]
    latest_date = df.index[-1]
    current_goldbees = float(latest_row['GOLDBEES'])
    current_10g = float(latest_row['GOLD_INR_10G']) if 'GOLD_INR_10G' in latest_row else current_goldbees * 1000
    
    # Estimate conversion multiplier from Gold BeES to 10g Gold
    ratio_10g = current_10g / current_goldbees if current_goldbees > 0 else 1000.0
    
    recent_daily_vol = float(df['GOLDBEES_Ret_1d'].tail(30).std())
    if np.isnan(recent_daily_vol) or recent_daily_vol == 0:
        recent_daily_vol = 0.008
        
    future_dates = generate_future_trading_days(latest_date, num_days=days_ahead)
    
    feat_cols = [c for c in feature_names if c in df.columns]
    simulated_features = df[feat_cols].iloc[-1:].copy()
    
    forecasts = []
    simulated_price = current_goldbees
    
    for step, future_date in enumerate(future_dates, start=1):
        next_pred = float(model.predict(simulated_features)[0])
        
        # Volatility expansion
        horizon_vol = recent_daily_vol * np.sqrt(step) * next_pred
        lower_95 = max(next_pred - (1.96 * horizon_vol), 1.0)
        upper_95 = next_pred + (1.96 * horizon_vol)
        expected_ret = ((next_pred - current_goldbees) / current_goldbees) * 100.0
        
        # Equivalent 10 Grams Indian Gold Rate in INR
        pred_10g = next_pred * ratio_10g
        lower_10g = lower_95 * ratio_10g
        upper_10g = upper_95 * ratio_10g
        
        forecasts.append({
            'Date': future_date.strftime('%Y-%m-%d'),
            'Day_Horizon': step,
            'Predicted_GoldBeES (₹)': round(next_pred, 2),
            'Lower_Bound_BeES (₹)': round(lower_95, 2),
            'Upper_Bound_BeES (₹)': round(upper_95, 2),
            'Predicted_10g_24K (₹)': round(pred_10g, 0),
            'Lower_10g_24K (₹)': round(lower_10g, 0),
            'Upper_10g_24K (₹)': round(upper_10g, 0),
            'Expected_Return (%)': round(expected_ret, 2)
        })
        
        simulated_price = next_pred
        if 'GOLDBEES_Lag_1' in simulated_features.columns:
            simulated_features['GOLDBEES_Lag_1'] = simulated_price
        if 'GOLDBEES' in simulated_features.columns:
            simulated_features['GOLDBEES'] = simulated_price
            
    forecast_df = pd.DataFrame(forecasts)
    return forecast_df


if __name__ == '__main__':
    res = forecast_future_prices(days_ahead=14)
    print("\nNext 14-Day Indian Gold Price Forecast:")
    print(res[['Date', 'Day_Horizon', 'Predicted_GoldBeES (₹)', 'Predicted_10g_24K (₹)', 'Expected_Return (%)']].to_string(index=False))
