#!/usr/bin/env python3
"""
GPU-Accelerated Training Script for Smart Draft Assistant
=========================================================
Trains XGBoost on GPU with the full 103K match dataset.
"""
import sys
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, f1_score
from pathlib import Path
import json
import joblib

ML_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ML_ROOT / "datasets" / "processed"
MODEL_DIR = ML_ROOT / "models"

def main():
    print("=" * 60)
    print("GPU TRAINING - Smart Draft Assistant")
    print("=" * 60)
    
    print("\nLoading data...")
    df = pd.read_parquet(DATA_DIR / "draft_features.parquet")
    print(f"Loaded {len(df)} samples with {len(df.columns)} features")
    
    exclude_cols = ['match_id', 'radiant_win', 'patch', 'patch_version']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ('int64', 'float64', 'int32', 'float32')]
    
    print(f"Dropped non-numeric columns: {[c for c in df.columns if c not in feature_cols and c not in exclude_cols]}")
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['radiant_win'].values.astype(np.int32)
    
    print(f"Features: {len(feature_cols)}")
    print(f"Labels: {np.sum(y == 1)} radiant wins, {np.sum(y == 0)} dire wins")
    print(f"Balance: {np.mean(y):.3f} radiant win rate")
    
    # 5-fold cross validation
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    fold_scores = []
    
    # GPU-optimized XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 10,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'device': 'cuda',
        'tree_method': 'hist',
        'verbosity': 1,
    }
    
    print(f"\nXGBoost version: {xgb.__version__}")
    print(f"Parameters: {params}")
    
    start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, 'val')],
            early_stopping_rounds=100,
            verbose_eval=100,
        )
        
        val_preds = model.predict(dval)
        oof_preds[val_idx] = val_preds
        
        auc = roc_auc_score(y_val, val_preds)
        acc = accuracy_score(y_val, (val_preds > 0.5).astype(int))
        f1 = f1_score(y_val, (val_preds > 0.5).astype(int))
        
        fold_scores.append({'auc': auc, 'accuracy': acc, 'f1': f1, 'best_round': model.best_iteration})
        print(f"Fold {fold + 1} - AUC: {auc:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}, Best round: {model.best_iteration}")
    
    total_time = time.time() - start_time
    
    # Overall metrics
    overall_auc = roc_auc_score(y, oof_preds)
    overall_acc = accuracy_score(y, (oof_preds > 0.5).astype(int))
    overall_f1 = f1_score(y, (oof_preds > 0.5).astype(int))
    overall_ll = log_loss(y, oof_preds)
    
    print(f"\n{'=' * 60}")
    print(f"RESULTS (5-fold CV, {len(X)} samples)")
    print(f"{'=' * 60}")
    print(f"ROC-AUC:  {overall_auc:.4f}")
    print(f"Accuracy: {overall_acc:.4f}")
    print(f"F1 Score: {overall_f1:.4f}")
    print(f"Log Loss: {overall_ll:.4f}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"Avg fold time: {total_time/n_folds:.1f}s")
    
    # Per-fold summary
    print(f"\nFold details:")
    for i, fs in enumerate(fold_scores):
        print(f"  Fold {i+1}: AUC={fs['auc']:.4f} Acc={fs['accuracy']:.4f} F1={fs['f1']:.4f} rounds={fs['best_round']}")
    
    # Train final model on all data
    print(f"\nTraining final model on all {len(X)} samples...")
    dfull = xgb.DMatrix(X, label=y, feature_names=feature_cols)
    final_rounds = int(np.mean([fs['best_round'] for fs in fold_scores]))
    
    final_model = xgb.train(
        params,
        dfull,
        num_boost_round=final_rounds,
        verbose_eval=50,
    )
    
    # Feature importance
    importance = final_model.get_score(importance_type='gain')
    imp_list = [(k, v) for k, v in importance.items()]
    imp_list.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 20 features:")
    for name, gain in imp_list[:20]:
        print(f"  {name}: {gain:.2f}")
    
    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = MODEL_DIR / "draft_model_gpu.json"
    final_model.save_model(str(model_path))
    print(f"\nModel saved: {model_path}")
    
    # Save metadata
    meta = {
        'model_type': 'xgboost_gpu',
        'features': len(feature_cols),
        'training_samples': len(X),
        'cv_folds': n_folds,
        'metrics': {
            'roc_auc': float(overall_auc),
            'accuracy': float(overall_acc),
            'f1': float(overall_f1),
            'log_loss': float(overall_ll),
        },
        'fold_scores': fold_scores,
        'params': params,
        'best_rounds': final_rounds,
        'feature_cols': feature_cols,
        'training_time_s': total_time,
    }
    
    meta_path = MODEL_DIR / "training_meta_gpu.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved: {meta_path}")
    
    print(f"\nDone! ROC-AUC: {overall_auc:.4f}")


if __name__ == "__main__":
    main()
