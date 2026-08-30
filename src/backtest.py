"""
Algorithmic Trading Strategy Backtester for Gold Price Predictions.
Simulates long/cash and long/short strategies based on ML directional signals
and compares against the standard Buy & Hold benchmark.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
if CURR_DIR not in sys.path:
    sys.path.insert(0, CURR_DIR)

MODELS_DIR = os.path.join(os.path.dirname(CURR_DIR), 'models')


def run_strategy_backtest(allow_short: bool = False, initial_capital: float = 10000.0) -> dict:
    """
    Simulates ML-driven algorithmic trading on holdout test data.
    """
    preds_path = os.path.join(MODELS_DIR, 'test_predictions.json')
    if not os.path.exists(preds_path):
        raise FileNotFoundError("Test predictions not found. Run train.py first.")
        
    with open(preds_path, 'r') as f:
        data = json.load(f)
        
    dates = pd.to_datetime(data['Date'])
    actual_prices = np.array(data['Actual'])
    pred_prices = np.array(data['Stacking_Ensemble'])
    
    # Calculate baseline actual returns: r_t = (P_t - P_{t-1}) / P_{t-1}
    actual_returns = np.zeros(len(actual_prices))
    actual_returns[1:] = (actual_prices[1:] - actual_prices[:-1]) / actual_prices[:-1]
    
    # Trading Signals: 1 if predicted next price > current price else 0 (or -1 if short)
    signals = np.zeros(len(actual_prices))
    for i in range(len(actual_prices) - 1):
        if pred_prices[i] > actual_prices[i]:
            signals[i + 1] = 1.0
        else:
            signals[i + 1] = -1.0 if allow_short else 0.0
            
    # Strategy Daily Return
    strat_returns = signals * actual_returns
    
    # Cumulative Curves
    equity_curve_strategy = initial_capital * np.cumprod(1.0 + strat_returns)
    equity_curve_benchmark = initial_capital * np.cumprod(1.0 + actual_returns)
    
    # Performance Statistics
    total_strat_ret = ((equity_curve_strategy[-1] - initial_capital) / initial_capital) * 100.0
    total_bench_ret = ((equity_curve_benchmark[-1] - initial_capital) / initial_capital) * 100.0
    
    n_days = len(actual_prices)
    years = max(n_days / 252.0, 0.1)
    
    cagr_strat = ((equity_curve_strategy[-1] / initial_capital) ** (1.0 / years) - 1.0) * 100.0
    cagr_bench = ((equity_curve_benchmark[-1] / initial_capital) ** (1.0 / years) - 1.0) * 100.0
    
    strat_vol = np.std(strat_returns) * np.sqrt(252) * 100.0
    bench_vol = np.std(actual_returns) * np.sqrt(252) * 100.0
    
    rf = 0.02  # 2% risk-free
    sharpe_strat = (cagr_strat / 100.0 - rf) / (strat_vol / 100.0) if strat_vol > 0 else 0.0
    sharpe_bench = (cagr_bench / 100.0 - rf) / (bench_vol / 100.0) if bench_vol > 0 else 0.0
    
    # Max Drawdown
    def compute_max_drawdown(equity_series):
        peak = np.maximum.accumulate(equity_series)
        drawdown = (equity_series - peak) / peak
        return float(np.min(drawdown) * 100.0)
        
    mdd_strat = compute_max_drawdown(equity_curve_strategy)
    mdd_bench = compute_max_drawdown(equity_curve_benchmark)
    
    # Win rate on active trading days
    active_days = strat_returns != 0
    win_rate = float(np.mean(strat_returns[active_days] > 0) * 100.0) if np.sum(active_days) > 0 else 0.0
    
    results = {
        'Summary': {
            'Strategy_Total_Return (%)': round(total_strat_ret, 2),
            'Benchmark_Total_Return (%)': round(total_bench_ret, 2),
            'Strategy_CAGR (%)': round(cagr_strat, 2),
            'Benchmark_CAGR (%)': round(cagr_bench, 2),
            'Strategy_Sharpe': round(sharpe_strat, 2),
            'Benchmark_Sharpe': round(sharpe_bench, 2),
            'Strategy_Max_Drawdown (%)': round(mdd_strat, 2),
            'Benchmark_Max_Drawdown (%)': round(mdd_bench, 2),
            'Win_Rate (%)': round(win_rate, 2),
            'Total_Test_Days': n_days
        },
        'Equity_Curve': {
            'Dates': [d.strftime('%Y-%m-%d') for d in dates],
            'Strategy': equity_curve_strategy.tolist(),
            'Benchmark': equity_curve_benchmark.tolist(),
            'Signals': signals.tolist()
        }
    }
    return results


if __name__ == '__main__':
    res = run_strategy_backtest(allow_short=False)
    print("\nBacktest Summary:")
    print(json.dumps(res['Summary'], indent=2))
