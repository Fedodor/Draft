#!/usr/bin/env python3
"""
Enhanced Training Script - Filtered + Tuned
============================================
Filters to recent patches, tunes hyperparameters, evaluates properly.
"""
import sys
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, f1_score, brier_score_loss
from pathlib import Path
import json

ML_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ML_ROOT / "datasets" / "processed"
MODEL_DIR = ML_ROOT / "models"

RECENT_PATCHES = {"7.37", "7.38", "7.39", "7.40", "7.41"}

def train_and_evaluate(X, y, feature_cols, params, n_folds=5, early_stopping=100, num_boost_round=3000, label=""):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    fold_scores = []
    
    start = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
        
        model = xgb.train(
            params, dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, 'val')],
            early_stopping_rounds=early_stopping,
            verbose_eval=False,
        )
        
        val_preds = model.predict(dval)
        oof_preds[val_idx] = val_preds
        
        auc = roc_auc_score(y_val, val_preds)
        acc = accuracy_score(y_val, (val_preds > 0.5).astype(int))
        f1 = f1_score(y_val, (val_preds > 0.5).astype(int))
        brier = brier_score_loss(y_val, val_preds)
        
        fold_scores.append({'auc': auc, 'accuracy': acc, 'f1': f1, 'brier': brier, 'best_round': model.best_iteration})
    
    elapsed = time.time() - start
    
    overall_auc = roc_auc_score(y, oof_preds)
    overall_acc = accuracy_score(y, (oof_preds > 0.5).astype(int))
    overall_f1 = f1_score(y, (oof_preds > 0.5).astype(int))
    overall_ll = log_loss(y, oof_preds)
    overall_brier = brier_score_loss(y, oof_preds)
    
    print(f"\n{'=' * 50}")
    print(f"{label} ({len(X)} samples, {len(feature_cols)} features)")
    print(f"{'=' * 50}")
    print(f"ROC-AUC:   {overall_auc:.4f}")
    print(f"Accuracy:  {overall_acc:.4f}")
    print(f"F1 Score:  {overall_f1:.4f}")
    print(f"Log Loss:  {overall_ll:.4f}")
    print(f"Brier:     {overall_brier:.4f}")
    print(f"Time:      {elapsed:.1f}s")
    fold_aucs = [s['auc'] for s in fold_scores]
    fold_str = ", ".join("%.4f" % a for a in fold_aucs)
    print(f"\nFold AUCs: [{fold_str}]")
    print(f"Fold std:  {np.std([s['auc'] for s in fold_scores]):.4f}")
    
    return overall_auc, overall_acc, overall_f1, overall_ll, overall_brier, fold_scores, elapsed


def main():
    print("=" * 60)
    print("ENHANCED TRAINING - Smart Draft Assistant")
    print("=" * 60)
    
    df = pd.read_parquet(DATA_DIR / "draft_features.parquet")
    print(f"Loaded {len(df)} samples")
    
    exclude_cols = ['match_id', 'radiant_win', 'patch', 'patch_version']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ('int64', 'float64', 'int32', 'float32')]
    
    X_all = df[feature_cols].values.astype(np.float32)
    y_all = df['radiant_win'].values.astype(np.int32)
    
    print(f"Total features: {len(feature_cols)}")
    print(f"Total samples: {len(X_all)}")
    
    # Patch distribution
    if 'patch' in df.columns:
        print(f"\nPatch distribution:")
        for p, c in df['patch'].value_counts().head(15).items():
            print(f"  {p}: {c}")
    
    # ============================================================
    # EXP 1: All patches baseline
    # ============================================================
    base_params = {
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
        'verbosity': 0,
    }
    
    print("\n" + "=" * 60)
    print("EXP 1: All patches baseline")
    print("=" * 60)
    auc1, acc1, f11, ll1, brier1, folds1, t1 = train_and_evaluate(
        X_all, y_all, feature_cols, base_params, label="All patches (baseline)")
    
    # ============================================================
    # EXP 2: Recent patches only (7.37+)
    # ============================================================
    print("\n" + "=" * 60)
    print("EXP 2: Recent patches only (7.37+)")
    print("=" * 60)
    
    if 'patch' in df.columns:
        recent_mask = df['patch'].isin(RECENT_PATCHES)
        df_recent = df[recent_mask]
        print(f"Recent patches: {recent_mask.sum()} / {len(df)} samples ({recent_mask.mean()*100:.1f}%)")
        print(f"Recent patch distribution:")
        for p, c in df_recent['patch'].value_counts().items():
            print(f"  {p}: {c}")
    else:
        df_recent = df
        print("No patch column found, using all data")
    
    X_recent = df_recent[feature_cols].values.astype(np.float32)
    y_recent = df_recent['radiant_win'].values.astype(np.int32)
    
    auc2, acc2, f12, ll2, brier2, folds2, t2 = train_and_evaluate(
        X_recent, y_recent, feature_cols, base_params, label="Recent patches (7.37+)")
    
    # ============================================================
    # EXP 3: Recent patches with tuned params
    # ============================================================
    print("\n" + "=" * 60)
    print("EXP 3: Recent patches + tuned hyperparams")
    print("=" * 60)
    
    tuned_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.03,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'min_child_weight': 15,
        'gamma': 0.2,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'scale_pos_weight': len(y_recent[y_recent == 0]) / max(1, len(y_recent[y_recent == 1])),
        'device': 'cuda',
        'tree_method': 'hist',
        'verbosity': 0,
    }
    
    auc3, acc3, f13, ll3, brier3, folds3, t3 = train_and_evaluate(
        X_recent, y_recent, feature_cols, tuned_params, num_boost_round=5000,
        early_stopping=150, label="Recent patches (tuned)")
    
    # ============================================================
    # EXP 4: More trees, lower learning rate
    # ============================================================
    print("\n" + "=" * 60)
    print("EXP 4: Deep search (low LR, more trees)")
    print("=" * 60)
    
    deep_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 10,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'min_child_weight': 20,
        'gamma': 0.3,
        'reg_alpha': 1.0,
        'reg_lambda': 3.0,
        'max_bin': 512,
        'device': 'cuda',
        'tree_method': 'hist',
        'verbosity': 0,
    }
    
    auc4, acc4, f14, ll4, brier4, folds4, t4 = train_and_evaluate(
        X_recent, y_recent, feature_cols, deep_params, num_boost_round=10000,
        early_stopping=200, label="Deep search")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Experiment':<40} {'AUC':>8} {'Acc':>8} {'F1':>8} {'Brier':>8}")
    print("-" * 72)
    print(f"{'All patches (baseline)':<40} {auc1:>8.4f} {acc1:>8.4f} {f11:>8.4f} {brier1:>8.4f}")
    print(f"{'Recent patches (7.37+)':<40} {auc2:>8.4f} {acc2:>8.4f} {f12:>8.4f} {brier2:>8.4f}")
    print(f"{'Recent patches (tuned)':<40} {auc3:>8.4f} {acc3:>8.4f} {f13:>8.4f} {brier3:>8.4f}")
    print(f"{'Deep search':<40} {auc4:>8.4f} {acc4:>8.4f} {f14:>8.4f} {brier4:>8.4f}")
    
    # Pick best
    results = [
        ('all_baseline', auc1, base_params, feature_cols, X_all, y_all, folds1, t1),
        ('recent_base', auc2, base_params, feature_cols, X_recent, y_recent, folds2, t2),
        ('recent_tuned', auc3, tuned_params, feature_cols, X_recent, y_recent, folds3, t3),
        ('deep_search', auc4, deep_params, feature_cols, X_recent, y_recent, folds4, t4),
    ]
    
    best = max(results, key=lambda x: x[1])
    name, best_auc, best_params, best_features, X_best, y_best, best_folds, best_time = best
    
    print(f"\nBest: {name} (AUC: {best_auc:.4f})")
    
    # Train final model on best config
    print(f"\nTraining final model ({name})...")
    dfull = xgb.DMatrix(X_best, label=y_best, feature_names=best_features)
    final_rounds = int(np.mean([f['best_round'] for f in best_folds]))
    
    final_model = xgb.train(
        best_params, dfull,
        num_boost_round=final_rounds,
        verbose_eval=0,
    )
    
    # Feature importance
    importance = final_model.get_score(importance_type='gain')
    imp_list = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 25 features:")
    for name_f, gain in imp_list[:25]:
        print(f"  {name_f}: {gain:.2f}")
    
    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    final_model.save_model(str(MODEL_DIR / "draft_model_gpu.json"))
    
    meta = {
        'model_type': 'xgboost_gpu',
        'features': len(best_features),
        'training_samples': len(X_best),
        'total_matches': len(df),
        'patch_filter': list(RECENT_PATCHES) if name != 'all_baseline' else 'all',
        'metrics': {
            'roc_auc': float(best_auc),
        },
        'fold_scores': best_folds,
        'params': {k: v for k, v in best_params.items() if k != 'device'},
        'best_rounds': final_rounds,
        'feature_cols': best_features,
    }
    
    with open(MODEL_DIR / "training_meta_gpu.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nModel saved: {MODEL_DIR / 'draft_model_gpu.json'}")
    print(f"Done! Best ROC-AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
