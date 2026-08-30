"""
Data Loader Module for Gold Price Prediction.
Fetches live historical data from Yahoo Finance across multiple asset classes
(Equities, Commodities, Forex, Bond Yields, Volatility) and provides local caching.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# Default tickers to download
TICKERS = {
    'GLD': 'GLD',          # SPDR Gold Shares (Target ETF)
    'GC_F': 'GC=F',        # Gold Futures Continuous Contract
    'SPX': '^GSPC',        # S&P 500 Index
    'USO': 'USO',          # United States Oil Fund ETF
    'SLV': 'SLV',          # iShares Silver Trust ETF
    'EURUSD': 'EURUSD=X',  # EUR / USD Exchange Rate
    'DXY': 'DX-Y.NYB',     # US Dollar Index
    'TNX': '^TNX',         # CBOE 10-Year Treasury Yield Note
    'VIX': '^VIX',         # CBOE Volatility Index
    'TIP': 'TIP'           # iShares TIPS Bond ETF (Inflation Expectations)
}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'gold_market_data.csv')
LEGACY_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'gld_price_data.csv')


def download_all_market_data(start_date: str = '2008-01-01', force_refresh: bool = False) -> pd.DataFrame:
    """
    Downloads multi-asset financial data, merges them on trading dates,
    and returns a unified DataFrame.
    """
    os.makedirs(os.path.join(DATA_DIR, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'processed'), exist_ok=True)
    
    if os.path.exists(RAW_DATA_PATH) and not force_refresh:
        print(f"[Info] Loading cached market data from {RAW_DATA_PATH}")
        df = pd.read_csv(RAW_DATA_PATH, index_col=0, parse_dates=True)
        return df

    print(f"[Info] Fetching latest live data from Yahoo Finance (2008 to present)...")
    ticker_symbols = list(TICKERS.values())
    
    try:
        raw_download = yf.download(
            tickers=ticker_symbols,
            start=start_date,
            auto_adjust=False,
            group_by='ticker',
            progress=False
        )
    except Exception as e:
        print(f"[Error] Batch download failed: {e}. Falling back to single ticker downloads...")
        raw_download = None

    merged_data = pd.DataFrame()
    
    # Invert mapping: symbol -> name
    sym_to_name = {v: k for k, v in TICKERS.items()}
    
    for name, symbol in TICKERS.items():
        try:
            if raw_download is not None and symbol in raw_download:
                ticker_df = raw_download[symbol].copy()
            else:
                ticker_df = yf.download(symbol, start=start_date, auto_adjust=False, progress=False)
                if isinstance(ticker_df.columns, pd.MultiIndex):
                    ticker_df.columns = [col[0] for col in ticker_df.columns]
                    
            if ticker_df.empty or 'Close' not in ticker_df.columns:
                continue

            ticker_df = ticker_df.dropna(how='all')
            ticker_df.index = pd.to_datetime(ticker_df.index).tz_localize(None)

            if name == 'GLD':
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in ticker_df.columns:
                        merged_data[f'GLD_{col}'] = ticker_df[col]
                merged_data['GLD'] = ticker_df['Close']
            else:
                merged_data[name] = ticker_df['Close']
                
            print(f"  ✓ {name} ({symbol}): {len(ticker_df)} rows")
        except Exception as e:
            print(f"  ✗ Failed fetching {name} ({symbol}): {e}")

    # Forward fill non-trading days / slight holiday mismatches across international assets
    merged_data = merged_data.sort_index()
    merged_data = merged_data.ffill().bfill()
    
    # Drop rows where GLD is missing
    if 'GLD' in merged_data.columns:
        merged_data = merged_data.dropna(subset=['GLD'])
        
    # Save to raw storage
    merged_data.to_csv(RAW_DATA_PATH)
    print(f"[Success] Data saved to {RAW_DATA_PATH} with shape: {merged_data.shape}")
    print(f"Date range: {merged_data.index.min().strftime('%Y-%m-%d')} to {merged_data.index.max().strftime('%Y-%m-%d')}")
    
    return merged_data


def load_legacy_data() -> pd.DataFrame:
    """
    Loads the original 2008-2018 benchmark dataset for backwards compatibility and baseline comparisons.
    """
    if os.path.exists(LEGACY_DATA_PATH):
        df = pd.read_csv(LEGACY_DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        return df
    raise FileNotFoundError(f"Legacy dataset not found at {LEGACY_DATA_PATH}")


if __name__ == '__main__':
    df = download_all_market_data(force_refresh=True)
    print("\nDataset Summary:")
    print(df.info())
    print("\nTail (Most Recent Data):")
    print(df.tail())
