"""
Machine Learning and Deep Learning Model Architectures for Indian Gold Price Forecasting.
Includes Residual/Delta Wrappers for Tree Models, Ridge, PyTorch LSTM/GRU, and Stacking Ensemble.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
from typing import Tuple, Dict, Any


class ResidualPricePredictor(BaseEstimator, RegressorMixin):
    """
    Financial Time-Series Regressor that predicts price delta (P_{t+1} - P_t).
    Enables tree models (XGBoost, LightGBM, Random Forest) to handle non-stationary
    trending regimes and all-time-high price extrapolations in INR without bias.
    """
    def __init__(self, base_estimator: Any, price_col: str = 'GOLDBEES'):
        self.base_estimator = base_estimator
        self.price_col = price_col

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if isinstance(X, pd.DataFrame) and self.price_col in X.columns:
            current_prices = X[self.price_col]
        else:
            current_prices = X[:, 0] if isinstance(X, np.ndarray) else X[self.price_col]
            
        y_delta = y - current_prices
        self.base_estimator.fit(X, y_delta)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame) and self.price_col in X.columns:
            current_prices = X[self.price_col].values
        else:
            current_prices = X[:, 0] if isinstance(X, np.ndarray) else X[self.price_col].values
            
        pred_delta = self.base_estimator.predict(X)
        return current_prices + pred_delta

    @property
    def feature_importances_(self):
        if hasattr(self.base_estimator, 'feature_importances_'):
            return self.base_estimator.feature_importances_
        return None


class PyTorchLSTM(nn.Module):
    """
    Recurrent Neural Network with LSTM cells for financial sequence forecasting.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(PyTorchLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        x = self.fc1(last_step)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)
        return out.squeeze(-1)


def get_model_instances(price_col: str = 'GOLDBEES') -> Dict[str, Any]:
    """
    Returns configured model instances wrapped with ResidualPricePredictor.
    """
    rf_base = RandomForestRegressor(
        n_estimators=120,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1
    )
    xgb_base = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,
        tree_method='hist'
    )
    lgb_base = lgb.LGBMRegressor(
        n_estimators=150,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
        n_jobs=1
    )
    ridge_base = Ridge(alpha=10.0)

    models = {
        'Random_Forest': ResidualPricePredictor(rf_base, price_col=price_col),
        'XGBoost': ResidualPricePredictor(xgb_base, price_col=price_col),
        'LightGBM': ResidualPricePredictor(lgb_base, price_col=price_col),
        'Ridge': ResidualPricePredictor(ridge_base, price_col=price_col)
    }
    return models


class StackingEnsemble:
    """
    Weighted / Meta-Ensemble combining best Gradient Boosters and Tree models.
    """
    def __init__(self, models: Dict[str, Any], weights: Dict[str, float] = None):
        self.models = models
        if weights is None:
            self.weights = {k: 1.0 / len(models) for k in models.keys()}
        else:
            total = sum(weights.values())
            self.weights = {k: v / total for k, v in weights.items()}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        for name, model in self.models.items():
            model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(X))
        for name, model in self.models.items():
            preds += self.weights[name] * model.predict(X)
        return preds


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts 2D tabular features into 3D sequential sliding windows for LSTM/GRU.
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y[i + seq_len])
    return np.array(X_seq), np.array(y_seq)
