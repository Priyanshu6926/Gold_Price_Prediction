"""
Training and Walk-Forward Validation Engine for Indian Gold Market Prediction.
Evaluates models across NSE GOLDBEES and Domestic Spot Gold in INR.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# Ensure src directory is in sys.path
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
if CURR_DIR not in sys.path:
    sys.path.insert(0, CURR_DIR)

from features import PROCESSED_DATA_PATH, prepare_full_features
from data_loader import download_indian_market_data
from models import get_model_instances, StackingEnsemble, PyTorchLSTM, create_sequences

MODELS_DIR = os.path.join(os.path.dirname(CURR_DIR), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, y_lag: np.ndarray) -> float:
    """
    Measures the % of correct directional predictions (Up vs Down move).
    """
    actual_dir = (y_true > y_lag).astype(int)
    pred_dir = (y_pred > y_lag).astype(int)
    return float(np.mean(actual_dir == pred_dir) * 100.0)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_lag: np.ndarray) -> dict:
    """
    Computes regression and financial metrics.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)
    r2 = float(r2_score(y_true, y_pred))
    da = calculate_directional_accuracy(y_true, y_pred, y_lag)
    
    return {
        'RMSE (₹)': round(rmse, 4),
        'MAE (₹)': round(mae, 4),
        'MAPE (%)': round(mape, 2),
        'R2_Score': round(r2, 4),
        'Directional_Accuracy (%)': round(da, 2)
    }


def prepare_feature_target_split(df: pd.DataFrame, target_col: str = 'Target_Next_Close', base_price_col: str = 'GOLDBEES'):
    """
    Extracts feature matrix X, target y, and lag baseline.
    """
    target_cols = [c for c in df.columns if c.startswith('Target_')]
    y = df[target_col].copy()
    X = df.drop(columns=target_cols).copy()
    y_lag = df[base_price_col].copy()
    return X, y, y_lag


def run_walk_forward_cv(X: pd.DataFrame, y: pd.Series, y_lag: pd.Series, price_col: str = 'GOLDBEES', n_splits: int = 5):
    """
    Performs expanding-window TimeSeriesSplit cross validation on Indian market data.
    """
    print(f"\n[Info] Running Indian Market Walk-Forward Cross Validation ({n_splits} folds)...", flush=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    models = get_model_instances(price_col=price_col)
    fold_metrics = {name: [] for name in models.keys()}
    fold_metrics['Stacking_Ensemble'] = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        lag_te = y_lag.iloc[test_idx]
        
        trained_fold_models = {}
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            metrics = evaluate_predictions(y_te.values, preds, lag_te.values)
            fold_metrics[name].append(metrics)
            trained_fold_models[name] = model
            
        ensemble = StackingEnsemble(
            {'XGBoost': trained_fold_models['XGBoost'],
             'LightGBM': trained_fold_models['LightGBM'],
             'Random_Forest': trained_fold_models['Random_Forest']},
            weights={'XGBoost': 0.45, 'LightGBM': 0.45, 'Random_Forest': 0.10}
        )
        ens_preds = ensemble.predict(X_te)
        ens_metrics = evaluate_predictions(y_te.values, ens_preds, lag_te.values)
        fold_metrics['Stacking_Ensemble'].append(ens_metrics)
        print(f"  Fold {fold+1}/{n_splits} complete.", flush=True)
        
    cv_summary = {}
    for name, list_metrics in fold_metrics.items():
        avg_metrics = {}
        for metric_name in list_metrics[0].keys():
            avg_metrics[metric_name] = round(float(np.mean([m[metric_name] for m in list_metrics])), 4)
        cv_summary[name] = avg_metrics
        
    print("\n--- Indian Market Walk-Forward Cross Validation Summary ---", flush=True)
    cv_df = pd.DataFrame(cv_summary).T
    print(cv_df.to_string(), flush=True)
    return cv_df


def train_pytorch_lstm(X_train: pd.DataFrame, y_train: pd.Series, 
                       X_test: pd.DataFrame, y_test: pd.Series, 
                       price_col: str = 'GOLDBEES',
                       seq_len: int = 20, epochs: int = 30, lr: float = 0.001):
    """
    Trains PyTorch LSTM sequence forecaster on delta changes.
    """
    print(f"\n[Info] Training PyTorch Deep Learning LSTM Model for Indian Market ({price_col})...", flush=True)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    y_delta_train = y_train - X_train[price_col]
    y_delta_test = y_test - X_test[price_col]
    
    X_tr_s = scaler_x.fit_transform(X_train)
    X_te_s = scaler_x.transform(X_test)
    
    y_tr_s = scaler_y.fit_transform(y_delta_train.values.reshape(-1, 1)).flatten()
    y_te_s = scaler_y.transform(y_delta_test.values.reshape(-1, 1)).flatten()
    
    X_tr_seq, y_tr_seq = create_sequences(X_tr_s, y_tr_s, seq_len=seq_len)
    X_te_seq, y_te_seq = create_sequences(X_te_s, y_te_s, seq_len=seq_len)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_tr_seq), torch.FloatTensor(y_tr_seq))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PyTorchLSTM(input_dim=X_train.shape[1], hidden_dim=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
    model.eval()
    with torch.no_grad():
        test_inputs = torch.FloatTensor(X_te_seq).to(device)
        norm_preds = model(test_inputs).cpu().numpy()
        pred_deltas = scaler_y.inverse_transform(norm_preds.reshape(-1, 1)).flatten()
        
    price_test_aligned = X_test[price_col].values[seq_len:]
    preds_price = price_test_aligned + pred_deltas
    y_test_aligned = y_test.values[seq_len:]
    
    lstm_metrics = evaluate_predictions(y_test_aligned, preds_price, price_test_aligned)
    print(f"  ✓ PyTorch LSTM Test Results: {lstm_metrics}", flush=True)
    
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'pytorch_lstm.pt'))
    joblib.dump(scaler_x, os.path.join(MODELS_DIR, 'scaler_x_lstm.joblib'))
    joblib.dump(scaler_y, os.path.join(MODELS_DIR, 'scaler_y_lstm.joblib'))
    
    return model, lstm_metrics, preds_price


def train_and_export_all_models(target_name: str = 'GOLDBEES'):
    """
    End-to-end training pipeline on full and train/test splits for Indian Gold.
    """
    if not os.path.exists(PROCESSED_DATA_PATH):
        raw = download_indian_market_data()
        df = prepare_full_features(raw, target_col=target_name)
    else:
        df = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
        
    X, y, y_lag = prepare_feature_target_split(df, target_col='Target_Next_Close', base_price_col=target_name)
    
    # 1. Walk-Forward Cross Validation
    cv_summary_df = run_walk_forward_cv(X, y, y_lag, price_col=target_name, n_splits=5)
    cv_summary_df.to_csv(os.path.join(MODELS_DIR, 'cv_metrics.csv'))
    
    # 2. Train-Test Split (Chronological 85% Train, 15% Holdout Test)
    split_idx = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    lag_test = y_lag.iloc[split_idx:]
    
    with open(os.path.join(MODELS_DIR, 'feature_names.json'), 'w') as f:
        json.dump(list(X.columns), f)
        
    models = get_model_instances(price_col=target_name)
    test_metrics = {}
    test_predictions = {'Actual': y_test.values.tolist(), 'Date': [d.strftime('%Y-%m-%d') for d in y_test.index]}
    
    print("\n[Info] Training Final Indian Market Models on 85% Train and Evaluating on 15% Holdout Test...", flush=True)
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate_predictions(y_test.values, preds, lag_test.values)
        test_metrics[name] = metrics
        test_predictions[name] = preds.tolist()
        joblib.dump(model, os.path.join(MODELS_DIR, f'{name.lower()}_model.joblib'))
        print(f"  ✓ {name}: RMSE={metrics['RMSE (₹)']}, MAE={metrics['MAE (₹)']}, R2={metrics['R2_Score']}, DirAcc={metrics['Directional_Accuracy (%)']}%", flush=True)
        
    # Stacking Ensemble
    ensemble = StackingEnsemble(
        {'XGBoost': models['XGBoost'],
         'LightGBM': models['LightGBM'],
         'Random_Forest': models['Random_Forest']},
        weights={'XGBoost': 0.45, 'LightGBM': 0.45, 'Random_Forest': 0.10}
    )
    ens_preds = ensemble.predict(X_test)
    ens_metrics = evaluate_predictions(y_test.values, ens_preds, lag_test.values)
    test_metrics['Stacking_Ensemble'] = ens_metrics
    test_predictions['Stacking_Ensemble'] = ens_preds.tolist()
    joblib.dump(ensemble, os.path.join(MODELS_DIR, 'stacking_ensemble.joblib'))
    print(f"  ✓ Stacking_Ensemble: RMSE={ens_metrics['RMSE (₹)']}, MAE={ens_metrics['MAE (₹)']}, R2={ens_metrics['R2_Score']}, DirAcc={ens_metrics['Directional_Accuracy (%)']}%", flush=True)
    
    # 3. Train PyTorch LSTM
    lstm_model, lstm_metrics, lstm_preds = train_pytorch_lstm(X_train, y_train, X_test, y_test, price_col=target_name, seq_len=20)
    test_metrics['PyTorch_LSTM'] = lstm_metrics
    
    padded_lstm = [None] * 20 + lstm_preds.tolist()
    test_predictions['PyTorch_LSTM'] = padded_lstm
    
    # 4. Feature Importance extraction
    if hasattr(models['XGBoost'], 'feature_importances_') and models['XGBoost'].feature_importances_ is not None:
        feature_imp = pd.Series(models['XGBoost'].feature_importances_, index=X.columns).sort_values(ascending=False)
        feature_imp.head(30).to_csv(os.path.join(MODELS_DIR, 'feature_importance.csv'))
    
    with open(os.path.join(MODELS_DIR, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)
        
    with open(os.path.join(MODELS_DIR, 'test_predictions.json'), 'w') as f:
        json.dump(test_predictions, f)
        
    print(f"\n[Success] All Indian Market models successfully saved to {MODELS_DIR}", flush=True)
    return test_metrics


if __name__ == '__main__':
    train_and_export_all_models()
