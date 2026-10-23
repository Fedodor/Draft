import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, f1_score, brier_score_loss
from pathlib import Path
import json, time

ML_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ML_ROOT / "datasets" / "processed"
MODEL_DIR = ML_ROOT / "models"

df = pd.read_parquet(DATA_DIR / "draft_features.parquet")

exclude_cols = ['match_id', 'radiant_win', 'patch', 'patch_version']
feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ('int64', 'float64', 'int32', 'float32')]

RECENT = {"7.37", "7.38", "7.39", "7.40", "7.41"}
SCENARIOS = [
    ("All patches", df),
    ("7.37+", df[df['patch'].isin(RECENT)]),
    ("7.39 only", df[df['patch'] == '7.39']),
    ("7.38+", df[df['patch'].isin({"7.38", "7.39", "7.40", "7.41"})]),
    ("7.40+", df[df['patch'].isin({"7.40", "7.41"})]),
]

best_auc = 0
best_scenario = None
results = []

for name, sub_df in SCENARIOS:
    if len(sub_df) < 1000:
        print(f"Skipping {name} ({len(sub_df)} samples)")
        continue
    
    X = sub_df[feature_cols].values.astype(np.float32)
    y = sub_df['radiant_win'].values.astype(np.int32)
    
    print(f"\n--- {name} ({len(X)} samples) ---")
    
    # Try multiple param configs
    configs = [
        ("deep", {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 10, 'learning_rate': 0.01, 'subsample': 0.7, 'colsample_bytree': 0.6, 'min_child_weight': 20, 'gamma': 0.3, 'reg_alpha': 1.0, 'reg_lambda': 3.0, 'max_bin': 512, 'device': 'cuda', 'tree_method': 'hist', 'verbosity': 0}),
        ("wide", {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 4, 'learning_rate': 0.02, 'subsample': 0.8, 'colsample_bytree': 0.5, 'min_child_weight': 30, 'gamma': 0.5, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'max_bin': 256, 'device': 'cuda', 'tree_method': 'hist', 'verbosity': 0}),
        ("balanced", {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 7, 'learning_rate': 0.015, 'subsample': 0.75, 'colsample_bytree': 0.65, 'min_child_weight': 25, 'gamma': 0.4, 'reg_alpha': 1.5, 'reg_lambda': 4.0, 'max_bin': 512, 'device': 'cuda', 'tree_method': 'hist', 'verbosity': 0}),
    ]
    
    for cfg_name, params in configs:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(X))
        fold_aucs = []
        t0 = time.time()
        
        for fold, (tr, va) in enumerate(skf.split(X, y)):
            dtrain = xgb.DMatrix(X[tr], label=y[tr], feature_names=feature_cols)
            dval = xgb.DMatrix(X[va], label=y[va], feature_names=feature_cols)
            model = xgb.train(params, dtrain, num_boost_round=10000, evals=[(dval, 'val')], early_stopping_rounds=200, verbose_eval=False)
            preds = model.predict(dval)
            oof[va] = preds
            fold_aucs.append(roc_auc_score(y[va], preds))
        
        auc = roc_auc_score(y, oof)
        acc = accuracy_score(y, (oof > 0.5).astype(int))
        brier = brier_score_loss(y, oof)
        elapsed = time.time() - t0
        
        fold_str = ", ".join("%.4f" % a for a in fold_aucs)
        print("  %s: AUC=%.4f Acc=%.4f Brier=%.4f folds=[%s] std=%.4f (%.0fs)" % (cfg_name, auc, acc, brier, fold_str, np.std(fold_aucs), elapsed))
        
        if auc > best_auc:
            best_auc = auc
            best_scenario = (name, cfg_name, params, sub_df, feature_cols, fold_aucs)

print(f"\n{'='*60}")
print(f"BEST: {best_scenario[0]} / {best_scenario[1]} = AUC {best_auc:.4f}")
print(f"{'='*60}")

# Train final model on best
name, cfg_name, params, sub_df, fcols, folds = best_scenario
X = sub_df[fcols].values.astype(np.float32)
y = sub_df['radiant_win'].values.astype(np.int32)

dfull = xgb.DMatrix(X, label=y, feature_names=fcols)
# Use avg best round estimate
model = xgb.train(params, dfull, num_boost_round=500, verbose_eval=0)

importance = model.get_score(importance_type='gain')
imp_list = sorted(importance.items(), key=lambda x: x[1], reverse=True)
print(f"\nTop 20 features:")
for fname, gain in imp_list[:20]:
    print(f"  {fname}: {gain:.2f}")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
model.save_model(str(MODEL_DIR / "draft_model_gpu.json"))
meta = {
    'model_type': 'xgboost_gpu',
    'features': len(fcols),
    'training_samples': len(X),
    'scenario': name,
    'config': cfg_name,
    'metrics': {'roc_auc': float(best_auc)},
    'fold_aucs': fold_aucs,
    'params': {k: v for k, v in params.items() if k != 'device'},
    'feature_cols': fcols,
}
with open(MODEL_DIR / "training_meta_gpu.json", 'w') as f:
    json.dump(meta, f, indent=2)
print(f"\nSaved model to {MODEL_DIR}")
