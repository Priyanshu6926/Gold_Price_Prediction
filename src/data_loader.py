"""
Data Loader Module for Indian Gold Market Prediction.
Fetches live historical data from Yahoo Finance across Indian and Global assets:
- Nippon India ETF Gold BeES (GOLDBEES.NS) & Spot Gold per 10g in INR
- USD/INR Currency Rate (USDINR=X)
- NIFTY 50 (^NSEI) & BSE SENSEX (^BSESN)
- India VIX (^INDIAVIX)
- MCX / International Gold Futures (GC=F) & Silver (SI=F)
- Crude Oil (CL=F) & US Dollar Index (DX-Y.NYB)
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# Indian & Macro tickers to download
INDIAN_TICKERS = {
    'GOLDBEES': 'GOLDBEES.NS',  # Nippon India ETF Gold BeES (NSE)
    'USDINR': 'USDINR=X',      # USD to INR Exchange Rate
    'NIFTY50': '^NSEI',        # NIFTY 50 Index (NSE)
    'SENSEX': '^BSESN',        # BSE SENSEX Index (BSE)
    'INDIA_VIX': '^INDIAVIX',  # India Volatility Index
    'GC_F': 'GC=F',            # International Gold Futures ($/oz)
    'SILVER': 'SI=F',          # International Silver Futures ($/oz)
    'CRUDE_OIL': 'CL=F',       # Crude Oil Futures ($/bbl)
    'DXY': 'DX-Y.NYB'          # US Dollar Index
}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'indian_gold_market_data.csv')


def download_indian_market_data(start_date: str = '2010-01-01', force_refresh: bool = False) -> pd.DataFrame:
    """
    Downloads Indian and global market data, aligns on trading dates,
    and computes domestic Gold INR price per 10 grams (24K).
    """
    os.makedirs(os.path.join(DATA_DIR, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'processed'), exist_ok=True)
    
    if os.path.exists(RAW_DATA_PATH) and not force_refresh:
        print(f"[Info] Loading cached Indian market data from {RAW_DATA_PATH}")
        df = pd.read_csv(RAW_DATA_PATH, index_col=0, parse_dates=True)
        return df

    print(f"[Info] Fetching live Indian and global financial data from Yahoo Finance (2010 to present)...")
    merged_data = pd.DataFrame()
    
    for name, symbol in INDIAN_TICKERS.items():
        try:
            ticker_df = yf.download(symbol, start=start_date, auto_adjust=False, progress=False)
            if isinstance(ticker_df.columns, pd.MultiIndex):
                ticker_df.columns = [col[0] for col in ticker_df.columns]
                
            if ticker_df.empty or 'Close' not in ticker_df.columns:
                print(f"  [Warning] Empty data for {name} ({symbol})")
                continue

            ticker_df = ticker_df.dropna(subset=['Close'])
            ticker_df.index = pd.to_datetime(ticker_df.index).tz_localize(None)

            if name == 'GOLDBEES':
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in ticker_df.columns:
                        merged_data[f'GOLDBEES_{col}'] = ticker_df[col]
                merged_data['GOLDBEES'] = ticker_df['Close']
            else:
                merged_data[name] = ticker_df['Close']
                
            print(f"  ✓ {name} ({symbol}): {len(ticker_df)} rows")
        except Exception as e:
            print(f"  ✗ Failed fetching {name} ({symbol}): {e}")

    # Forward fill non-trading days across domestic and international holidays
    merged_data = merged_data.sort_index()
    merged_data = merged_data.ffill().bfill()
    
    # Compute Indian Spot Gold Price per 10 Grams (24 Karat) in INR
    # Formula: (Spot Gold USD/oz * USDINR / 31.1034768) * 10 * 1.06 (customs/landing parity factor)
    if 'GC_F' in merged_data.columns and 'USDINR' in merged_data.columns:
        merged_data['GOLD_INR_10G'] = (merged_data['GC_F'] * merged_data['USDINR'] / 31.1034768) * 10.0 * 1.06
        merged_data['SILVER_INR_1KG'] = (merged_data['SILVER'] * merged_data['USDINR'] / 31.1034768) * 1000.0 * 1.06
        
    # Primary Indian target is GOLDBEES (Nippon Gold ETF in INR) and GOLD_INR_10G
    if 'GOLDBEES' in merged_data.columns:
        merged_data = merged_data.dropna(subset=['GOLDBEES'])
        
    merged_data.to_csv(RAW_DATA_PATH)
    print(f"[Success] Data saved to {RAW_DATA_PATH} with shape: {merged_data.shape}")
    print(f"Date range: {merged_data.index.min().strftime('%Y-%m-%d')} to {merged_data.index.max().strftime('%Y-%m-%d')}")
    
    return merged_data


if __name__ == '__main__':
    df = download_indian_market_data(force_refresh=True)
    print("\nTail (Latest Indian Market Rates):")
    cols_to_show = [c for c in ['GOLDBEES', 'GOLD_INR_10G', 'USDINR', 'NIFTY50', 'SENSEX', 'INDIA_VIX'] if c in df.columns]
    print(df[cols_to_show].tail())
