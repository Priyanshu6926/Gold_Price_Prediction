"""
Feature Engineering Module for Indian Gold Price Prediction.
Generates technical indicators for GOLDBEES (NSE) and Domestic Spot Gold (₹/10g),
intermarket ratios with NIFTY 50, SENSEX, USD/INR, Crude Oil, India VIX,
and non-lookahead lagged predictors.
"""

import os
import numpy as np
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

PROCESSED_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'processed', 'indian_gold_features.csv'
)


def compute_indian_technical_indicators(df: pd.DataFrame, target_col: str = 'GOLDBEES') -> pd.DataFrame:
    """
    Computes comprehensive technical indicators for Indian Gold asset.
    """
    data = df.copy()
    close = data[target_col]
    high = data[f'{target_col}_High'] if f'{target_col}_High' in data.columns else close
    low = data[f'{target_col}_Low'] if f'{target_col}_Low' in data.columns else close
    volume = data[f'{target_col}_Volume'] if f'{target_col}_Volume' in data.columns else pd.Series(1, index=data.index)
    
    # 1. Moving Averages
    data['SMA_7'] = SMAIndicator(close=close, window=7).sma_indicator()
    data['SMA_21'] = SMAIndicator(close=close, window=21).sma_indicator()
    data['SMA_50'] = SMAIndicator(close=close, window=50).sma_indicator()
    data['SMA_200'] = SMAIndicator(close=close, window=200).sma_indicator()
    
    data['EMA_12'] = EMAIndicator(close=close, window=12).ema_indicator()
    data['EMA_26'] = EMAIndicator(close=close, window=26).ema_indicator()
    
    # Moving Average Distance and Golden Cross
    data['SMA_Dist_50'] = (close - data['SMA_50']) / data['SMA_50']
    data['SMA_Dist_200'] = (close - data['SMA_200']) / data['SMA_200']
    data['Golden_Cross'] = (data['SMA_50'] > data['SMA_200']).astype(int)
    
    # 2. MACD
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Diff'] = macd.macd_diff()
    
    # 3. Momentum & RSI
    data['RSI_14'] = RSIIndicator(close=close, window=14).rsi()
    data['RSI_7'] = RSIIndicator(close=close, window=7).rsi()
    
    stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    data['Stoch_K'] = stoch.stoch()
    data['Stoch_D'] = stoch.stoch_signal()
    
    # 4. Volatility (Bollinger Bands & ATR)
    bb = BollingerBands(close=close, window=20, window_dev=2)
    data['BB_Upper'] = bb.bollinger_hband()
    data['BB_Lower'] = bb.bollinger_lband()
    data['BB_Width'] = bb.bollinger_wband()
    data['BB_Pct_B'] = bb.bollinger_pband()
    
    atr = AverageTrueRange(high=high, low=low, close=close, window=14)
    data['ATR_14'] = atr.average_true_range()
    
    # 5. Volume Indicators
    if f'{target_col}_Volume' in data.columns and (data[f'{target_col}_Volume'] > 0).any():
        obv = OnBalanceVolumeIndicator(close=close, volume=volume)
        data['OBV'] = obv.on_balance_volume()
        data['OBV_SMA_20'] = data['OBV'].rolling(20).mean()
        
    return data


def compute_indian_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes intermarket ratios and macro correlations tailored for the Indian financial ecosystem.
    """
    data = df.copy()
    
    # Domestic Ratios
    if 'GOLDBEES' in data.columns and 'NIFTY50' in data.columns:
        data['Gold_Nifty_Ratio'] = data['GOLDBEES'] / data['NIFTY50']
    if 'GOLDBEES' in data.columns and 'SENSEX' in data.columns:
        data['Gold_Sensex_Ratio'] = data['GOLDBEES'] / data['SENSEX']
    if 'GOLDBEES' in data.columns and 'USDINR' in data.columns:
        data['Gold_USDINR_Ratio'] = data['GOLDBEES'] / data['USDINR']
    if 'GOLD_INR_10G' in data.columns and 'SILVER_INR_1KG' in data.columns:
        data['Gold_Silver_Ratio_INR'] = data['GOLD_INR_10G'] / (data['SILVER_INR_1KG'] / 100.0)
        
    # Asset Returns and Rolling Volatilities
    assets = ['GOLDBEES', 'GOLD_INR_10G', 'USDINR', 'NIFTY50', 'SENSEX', 'INDIA_VIX', 'GC_F', 'SILVER', 'CRUDE_OIL', 'DXY']
    for asset in assets:
        if asset in data.columns:
            data[f'{asset}_Ret_1d'] = data[asset].pct_change(1)
            data[f'{asset}_Ret_5d'] = data[asset].pct_change(5)
            data[f'{asset}_Ret_21d'] = data[asset].pct_change(21)
            data[f'{asset}_Vol_21d'] = data[f'{asset}_Ret_1d'].rolling(21).std() * np.sqrt(252)
            
    return data


def create_indian_lag_features(df: pd.DataFrame, target_col: str = 'GOLDBEES', lags: list = [1, 2, 3, 5, 10, 21]) -> pd.DataFrame:
    """
    Creates lagged observations to eliminate lookahead bias.
    """
    data = df.copy()
    for lag in lags:
        data[f'{target_col}_Lag_{lag}'] = data[target_col].shift(lag)
        if f'{target_col}_Ret_1d' in data.columns:
            data[f'{target_col}_Ret_Lag_{lag}'] = data[f'{target_col}_Ret_1d'].shift(lag)
            
    macro_lags = ['USDINR_Ret_1d', 'NIFTY50_Ret_1d', 'SENSEX_Ret_1d', 'INDIA_VIX_Ret_1d', 'CRUDE_OIL_Ret_1d', 'RSI_14', 'MACD', 'BB_Pct_B']
    for feat in macro_lags:
        if feat in data.columns:
            data[f'{feat}_Lag_1'] = data[feat].shift(1)
            
    return data


def construct_indian_target_variables(df: pd.DataFrame, target_col: str = 'GOLDBEES') -> pd.DataFrame:
    """
    Constructs forward-looking targets for Indian Gold forecasting.
    """
    data = df.copy()
    data['Target_Next_Close'] = data[target_col].shift(-1)
    data['Target_Next_Return'] = np.log(data[target_col].shift(-1) / data[target_col])
    data['Target_Next_Direction'] = (data['Target_Next_Close'] > data[target_col]).astype(int)
    
    # 10g Gold target parallel construction
    if 'GOLD_INR_10G' in data.columns:
        data['Target_Next_Close_10G'] = data['GOLD_INR_10G'].shift(-1)
        
    return data


def prepare_full_features(df: pd.DataFrame, target_col: str = 'GOLDBEES', save: bool = True) -> pd.DataFrame:
    """
    Orchestrates the full Indian market feature engineering pipeline.
    """
    print(f"[Info] Computing Technical Indicators for {target_col}...")
    featured = compute_indian_technical_indicators(df, target_col=target_col)
    
    print("[Info] Computing Indian Intermarket Macro Features (NIFTY, SENSEX, USDINR, VIX)...")
    featured = compute_indian_macro_features(featured)
    
    print("[Info] Creating Non-Lookahead Lagged Features...")
    featured = create_indian_lag_features(featured, target_col=target_col)
    
    print("[Info] Constructing Forward Targets...")
    featured = construct_indian_target_variables(featured, target_col=target_col)
    
    clean_df = featured.dropna().copy()
    print(f"[Success] Indian Feature Matrix ready: {clean_df.shape}")
    
    if save:
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        clean_df.to_csv(PROCESSED_DATA_PATH)
        print(f"[Success] Saved processed features to {PROCESSED_DATA_PATH}")
        
    return clean_df


if __name__ == '__main__':
    from data_loader import download_indian_market_data
    raw = download_indian_market_data()
    features = prepare_full_features(raw)
    print("\nFeature Columns Sample (First 20):", list(features.columns[:20]))
    print("Total features:", len(features.columns))
