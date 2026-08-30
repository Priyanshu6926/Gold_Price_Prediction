"""
Feature Engineering Module for Gold Price Prediction.
Generates technical indicators, inter-market ratios, momentum & volatility features,
and time-lagged predictors without lookahead bias.
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
    'data', 'processed', 'gold_features.csv'
)


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes comprehensive technical indicators for GLD.
    """
    data = df.copy()
    close = data['GLD']
    high = data['GLD_High'] if 'GLD_High' in data.columns else close
    low = data['GLD_Low'] if 'GLD_Low' in data.columns else close
    volume = data['GLD_Volume'] if 'GLD_Volume' in data.columns else pd.Series(1, index=data.index)
    
    # 1. Moving Averages
    data['SMA_7'] = SMAIndicator(close=close, window=7).sma_indicator()
    data['SMA_21'] = SMAIndicator(close=close, window=21).sma_indicator()
    data['SMA_50'] = SMAIndicator(close=close, window=50).sma_indicator()
    data['SMA_200'] = SMAIndicator(close=close, window=200).sma_indicator()
    
    data['EMA_12'] = EMAIndicator(close=close, window=12).ema_indicator()
    data['EMA_26'] = EMAIndicator(close=close, window=26).ema_indicator()
    
    # Trend Cross Signals
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
    
    # 5. Volume (if available)
    if 'GLD_Volume' in data.columns and (data['GLD_Volume'] > 0).any():
        obv = OnBalanceVolumeIndicator(close=close, volume=volume)
        data['OBV'] = obv.on_balance_volume()
        data['OBV_SMA_20'] = data['OBV'].rolling(20).mean()
        
    return data


def compute_intermarket_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cross-asset ratios, relative performance, and macro interactions.
    """
    data = df.copy()
    
    # Intermarket Ratios
    if 'SLV' in data.columns:
        data['Gold_Silver_Ratio'] = data['GLD'] / data['SLV']
    if 'USO' in data.columns:
        data['Gold_Oil_Ratio'] = data['GLD'] / data['USO']
    if 'SPX' in data.columns:
        data['Gold_SPX_Ratio'] = data['GLD'] / data['SPX']
    if 'DXY' in data.columns:
        data['Gold_DXY_Ratio'] = data['GLD'] / data['DXY']
        
    # Real Yield Proxy
    if 'TNX' in data.columns and 'TIP' in data.columns:
        data['Real_Yield_Proxy'] = data['TNX'] - data['TIP'].pct_change(252) * 100
        
    # Daily Returns
    asset_cols = ['GLD', 'SPX', 'USO', 'SLV', 'EURUSD', 'DXY', 'TNX', 'VIX', 'TIP']
    for col in asset_cols:
        if col in data.columns:
            data[f'{col}_Ret_1d'] = data[col].pct_change(1)
            data[f'{col}_Ret_5d'] = data[col].pct_change(5)
            data[f'{col}_Ret_21d'] = data[col].pct_change(21)
            data[f'{col}_Vol_21d'] = data[f'{col}_Ret_1d'].rolling(21).std() * np.sqrt(252)
            
    return data


def create_lag_features(df: pd.DataFrame, target_col: str = 'GLD', lags: list = [1, 2, 3, 5, 10, 21]) -> pd.DataFrame:
    """
    Creates lagged observations to enable pure time-series forecasting without lookahead bias.
    """
    data = df.copy()
    
    # Lag primary target price and returns
    for lag in lags:
        data[f'{target_col}_Lag_{lag}'] = data[target_col].shift(lag)
        data[f'{target_col}_Ret_Lag_{lag}'] = data[f'{target_col}_Ret_1d'].shift(lag) if f'{target_col}_Ret_1d' in data.columns else data[target_col].pct_change().shift(lag)
        
    # Lag key external macro features by 1 day (so t prediction uses t-1 exogenous close)
    macro_features = ['SPX_Ret_1d', 'USO_Ret_1d', 'SLV_Ret_1d', 'DXY_Ret_1d', 'VIX_Ret_1d', 'RSI_14', 'MACD', 'BB_Pct_B']
    for feat in macro_features:
        if feat in data.columns:
            data[f'{feat}_Lag_1'] = data[feat].shift(1)
            
    return data


def construct_target_variables(df: pd.DataFrame, target_col: str = 'GLD') -> pd.DataFrame:
    """
    Constructs clean forward-looking targets:
    - Target_Next_Close: Gold Price at t+1
    - Target_Next_Return: Log return at t+1
    - Target_Next_Direction: Binary 1 (Up) / 0 (Down) at t+1
    """
    data = df.copy()
    data['Target_Next_Close'] = data[target_col].shift(-1)
    data['Target_Next_Return'] = np.log(data[target_col].shift(-1) / data[target_col])
    data['Target_Next_Direction'] = (data['Target_Next_Close'] > data[target_col]).astype(int)
    
    # Multi-step targets for direct forecasting horizons
    for h in [5, 10, 21]:
        data[f'Target_Next_Close_{h}d'] = data[target_col].shift(-h)
        
    return data


def prepare_full_features(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Orchestrates full feature engineering pipeline.
    """
    print("[Info] Computing Technical Indicators...")
    featured = compute_technical_indicators(df)
    
    print("[Info] Computing Intermarket & Macro Ratios...")
    featured = compute_intermarket_features(featured)
    
    print("[Info] Creating Non-Lookahead Lagged Features...")
    featured = create_lag_features(featured)
    
    print("[Info] Constructing Target Variables...")
    featured = construct_target_variables(featured)
    
    # Drop warm-up rows (from 200-day moving average and lags) and the last rows where future target is NaN
    initial_shape = featured.shape
    clean_df = featured.dropna().copy()
    print(f"[Success] Feature matrix ready: {clean_df.shape} (from {initial_shape} raw rows)")
    
    if save:
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        clean_df.to_csv(PROCESSED_DATA_PATH)
        print(f"[Success] Saved processed features to {PROCESSED_DATA_PATH}")
        
    return clean_df


if __name__ == '__main__':
    from data_loader import download_all_market_data
    raw = download_all_market_data()
    features = prepare_full_features(raw)
    print("\nFeature Columns Sample (First 20):", list(features.columns[:20]))
    print("Total features:", len(features.columns))
