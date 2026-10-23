#!/usr/bin/env python3
"""
Enhanced Model Trainer for Smart Draft Assistant
================================================

Trains XGBoost/LightGBM models on processed draft features with:
- Time-based validation splits
- Multiple metrics (ROC-AUC, LogLoss, Brier, Calibration)
- Feature importance analysis
- ONNX export for inference
- Model versioning and experiment tracking

Usage:
    python trainer.py --data ml/datasets/processed/draft_features.parquet --model-dir ml/models
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import sys

ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split

from training.splits import resolve_time_column, time_split

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    from onnxmltools import convert_xgboost, convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

warnings.filterwarnings('ignore')


DEFAULT_MODEL_CONFIG = {
    "xgboost": {
        "n_estimators": 2000,
        "max_depth": 8,
        "learning_rate": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": ["auc", "logloss"],
        "early_stopping_rounds": 150,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 1,
    },
    "lightgbm": {
        "n_estimators": 2000,
        "max_depth": 8,
        "learning_rate": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "early_stopping_rounds": 150,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
}


class ModelTrainer:
    def __init__(
        self,
        data_path: Path,
        model_dir: Path,
        model_type: str = "xgboost",
        config: Dict = None
    ):
        self.data_path = data_path
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_type = model_type
        self.config = config or DEFAULT_MODEL_CONFIG.get(model_type, DEFAULT_MODEL_CONFIG["xgboost"])
        self.model = None
        self.feature_names = None
        self.training_history = {}
    
    def load_data(self) -> pd.DataFrame:
        df = pd.read_parquet(self.data_path)
        print(f"Loaded data: {df.shape[0]} samples, {df.shape[1]} features")
        
        # Print basic stats
        if 'radiant_win' in df.columns:
            print(f"Radiant win rate: {df['radiant_win'].mean():.3f}")
        if 'patch' in df.columns:
            print(f"Patches: {df['patch'].value_counts().head(10).to_dict()}")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare feature matrix and labels, excluding non-feature columns"""
        exclude_cols = [
            'match_id', 'radiant_win', 'patch', 'patch_version', 'duration',
            'avg_mmr', 'game_mode', 'lobby_type'
        ]
        
        # Exclude pick order columns (data leakage - future info)
        pick_cols = [c for c in df.columns if c.startswith(('rad_pick_', 'dire_pick_'))]
        exclude_cols.extend(pick_cols)
        
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df[feature_cols].fillna(0).values
        y = df['radiant_win'].astype(int).values
        
        self.feature_names = feature_cols
        return X, y, feature_cols
    
    def time_based_split(self, df: pd.DataFrame, test_size: float = 0.2):
        """Split chronologically by start_time (falling back to match_id)."""
        if 'radiant_win' not in df.columns:
            raise SystemExit("Data has no 'radiant_win' column to split on")
        try:
            column, used_start_time = resolve_time_column(df)
        except SystemExit:
            print("Warning: no time column, using random split")
            return train_test_split(
                df, test_size=test_size, random_state=42,
                stratify=df['radiant_win'],
            )
        if not used_start_time:
            print("Warning: start_time missing; using match_id for the chronological split")

        df_sorted = df.sort_values(column, kind='mergesort')
        split_idx = int(len(df_sorted) * (1 - test_size))

        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]

        print(f"Time-based split by {column}:")
        print(f"  Train: {len(train_df)} matches ({train_df[column].min():,} - {train_df[column].max():,})")
        print(f"  Test:  {len(test_df)} matches ({test_df[column].min():,} - {test_df[column].max():,})")
        print(f"  Train radiant WR: {train_df['radiant_win'].mean():.3f}")
        print(f"  Test radiant WR:  {test_df['radiant_win'].mean():.3f}")

        return train_df, test_df

    def three_way_split(
        self,
        df: pd.DataFrame,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
    ):
        """Chronological train/validation/test split.

        Validation is reserved for early stopping; the test partition is
        evaluated once after model selection.
        """
        result = time_split(df, train_frac, val_frac, test_frac)
        if not result.used_start_time:
            print("Warning: start_time missing; using match_id for the chronological split")
        print(f"Three-way chronological split by {result.time_column}: {result.sizes()}")
        for name, bounds in result.boundaries.items():
            print(f"  {name}: {bounds['min']:,} - {bounds['max']:,}")
        return result
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the model"""
        print(f"\nTraining {self.model_type.upper()} model...")
        print(f"Train samples: {len(X_train):,}, Val samples: {len(X_val):,}")
        print(f"Features: {len(self.feature_names):,}")
        
        if self.model_type == "lightgbm" and LGB_AVAILABLE:
            self.model = lgb.LGBMClassifier(**self.config)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(self.config.get('early_stopping_rounds', 150), verbose=False),
                          lgb.log_evaluation(period=50)]
            )
        else:
            self.model = xgb.XGBClassifier(**self.config)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=50
            )
        
        # Store training history
        if hasattr(self.model, 'evals_result_'):
            self.training_history = self.model.evals_result_
        
        return self.model
    
    def evaluate(self, X_test, y_test) -> Dict:
        """Evaluate model with comprehensive metrics"""
        y_pred = self.model.predict_proba(X_test)[:, 1]
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Core metrics
        auc = roc_auc_score(y_test, y_pred)
        logloss = log_loss(y_test, y_pred)
        brier = brier_score_loss(y_test, y_pred)
        
        # Accuracy at threshold
        accuracy = (y_pred_binary == y_test).mean()
        
        # Precision/Recall at 0.5
        tp = ((y_pred_binary == 1) & (y_test == 1)).sum()
        fp = ((y_pred_binary == 1) & (y_test == 0)).sum()
        fn = ((y_pred_binary == 0) & (y_test == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calibration
        calibration = self._calibration_curve(y_test, y_pred)
        
        # Binned accuracy
        binned_accuracy = self._binned_accuracy(y_test, y_pred)
        
        metrics = {
            'roc_auc': float(auc),
            'log_loss': float(logloss),
            'brier_score': float(brier),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'calibration': calibration,
            'binned_accuracy': binned_accuracy,
        }
        
        print(f"\n{'='*55}")
        print(f"EVALUATION METRICS")
        print(f"{'='*55}")
        print(f"ROC-AUC:     {auc:.4f}")
        print(f"LogLoss:     {logloss:.4f}")
        print(f"Brier Score: {brier:.4f}")
        print(f"Accuracy:    {accuracy:.4f}")
        print(f"Precision:   {precision:.4f}")
        print(f"Recall:      {recall:.4f}")
        print(f"F1:          {f1:.4f}")
        print(f"\nCalibration (predicted -> actual winrate):")
        for bin_info in calibration:
            print(f"  {bin_info['bin_min']:.1f}-{bin_info['bin_max']:.1f}: "
                  f"pred={bin_info['predicted']:.3f}, actual={bin_info['actual']:.3f}, "
                  f"n={bin_info['count']}")
        
        return metrics
    
    def _calibration_curve(self, y_true, y_pred, n_bins: int = 10) -> List[Dict]:
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred, bins) - 1
        
        results = []
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                avg_pred = y_pred[mask].mean()
                avg_true = y_true[mask].mean()
                results.append({
                    'bin_min': float(bins[i]),
                    'bin_max': float(bins[i+1]),
                    'predicted': float(avg_pred),
                    'actual': float(avg_true),
                    'count': int(mask.sum())
                })
        return results
    
    def _binned_accuracy(self, y_true, y_pred, n_bins: int = 10) -> List[Dict]:
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred, bins) - 1
        
        results = []
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                results.append({
                    'bin': i,
                    'pred_range': f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                    'accuracy': float((y_true[mask] == (y_pred[mask] > 0.5)).mean()),
                    'count': int(mask.sum())
                })
        return results
    
    def save_model(self, name: str = "draft_model"):
        """Save model in multiple formats with metadata"""
        # Joblib (primary format)
        joblib_path = self.model_dir / f"{name}.joblib"
        joblib.dump(self.model, joblib_path)
        print(f"Saved joblib model to {joblib_path}")
        
        # Feature names
        with open(self.model_dir / f"{name}_features.json", 'w') as f:
            json.dump(self.feature_names, f)
        
        # Model config
        with open(self.model_dir / f"{name}_config.json", 'w') as f:
            json.dump({
                'model_type': self.model_type,
                'model_config': self.config,
                'feature_count': len(self.feature_names),
                'feature_names': self.feature_names,
                'training_date': datetime.utcnow().isoformat(),
            }, f, indent=2)
        
        # ONNX export
        if ONNX_AVAILABLE:
            try:
                initial_type = [('float_input', FloatTensorType([None, len(self.feature_names)]))]
                if self.model_type == "lightgbm":
                    onnx_model = convert_lightgbm(self.model, initial_types=initial_type)
                else:
                    onnx_model = convert_xgboost(self.model, initial_types=initial_type)
                onnx_path = self.model_dir / f"{name}.onnx"
                with open(onnx_path, 'wb') as f:
                    f.write(onnx_model.SerializeToString())
                print(f"Saved ONNX model to {onnx_path}")
            except Exception as e:
                print(f"ONNX conversion failed: {e}")
        
        # Save training history
        if self.training_history:
            with open(self.model_dir / f"{name}_history.json", 'w') as f:
                # Convert numpy arrays to lists
                serializable = {}
                for k, v in self.training_history.items():
                    serializable[k] = {k2: [float(x) for x in v2] for k2, v2 in v.items()}
                json.dump(serializable, f, indent=2)
    
    def feature_importance(self, top_n: int = 50) -> pd.DataFrame:
        """Get feature importance as DataFrame"""
        if self.model is None:
            return pd.DataFrame()
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        df = pd.DataFrame({
            'rank': range(1, len(indices) + 1),
            'feature': [self.feature_names[i] for i in indices],
            'importance': importances[indices]
        })
        
        print(f"\nTop {top_n} Feature Importances:")
        for _, row in df.head(20).iterrows():
            print(f"  {row['rank']:2d}. {row['feature']}: {row['importance']:.5f}")
        
        return df


def train_model(
    data_path: Path,
    model_dir: Path,
    model_type: str = "xgboost",
    test_size: float = 0.1,
    time_split: bool = True,
    config: Dict = None,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Tuple[ModelTrainer, Dict]:
    """High-level training function.

    Early stopping uses the validation partition only; the test partition is
    evaluated once, after training, and never influences model selection.
    """
    trainer = ModelTrainer(data_path, model_dir, model_type, config)

    df = trainer.load_data()

    if time_split:
        result = trainer.three_way_split(df, train_frac, val_frac, test_size)
        train_df, val_df, test_df = result.train, result.val, result.test
    else:
        train_df, holdout_df = train_test_split(
            df, test_size=val_frac + test_size, random_state=42,
            stratify=df['radiant_win'],
        )
        val_df, test_df = train_test_split(
            holdout_df, test_size=test_size / (val_frac + test_size),
            random_state=42, stratify=holdout_df['radiant_win'],
        )

    X_train, y_train, _ = trainer.prepare_features(train_df)
    X_val, y_val, _ = trainer.prepare_features(val_df)
    X_test, y_test, _ = trainer.prepare_features(test_df)

    # Train with early stopping on validation.
    trainer.train(X_train, y_train, X_val, y_val)

    val_metrics = trainer.evaluate(X_val, y_val)
    print("\n--- Held-out test evaluation (single pass) ---")
    metrics = trainer.evaluate(X_test, y_test)
    metrics['validation'] = val_metrics

    # Feature importance
    trainer.feature_importance(top_n=50)

    # Save
    model_name = f"draft_model_{trainer.model_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    trainer.save_model(model_name)

    return trainer, metrics


def train_per_patch(
    data_path: Path,
    model_dir: Path,
    model_type: str = "xgboost",
    test_size: float = 0.2,
    config: Dict = None
) -> Dict[str, Dict]:
    """Train a separate model for each patch (patch-aware)."""
    df = pd.read_parquet(data_path)
    if 'patch' not in df.columns:
        raise SystemExit("No 'patch' column; cannot train per-patch models")

    df['_patch_str'] = df['patch'].astype(str)
    results = {}
    for patch in sorted(df['_patch_str'].unique()):
        sub = df[df['_patch_str'] == patch].copy()
        if len(sub) < 500:
            print(f"Patch {patch}: skipping ({len(sub)} samples)")
            continue
        # Chronological split within patch
        if 'match_id' in sub.columns:
            sub = sub.sort_values('match_id')
        split_idx = int(len(sub) * (1 - test_size))
        train_df, test_df = sub.iloc[:split_idx], sub.iloc[split_idx:]

        trainer = ModelTrainer(data_path, model_dir, model_type, config)
        trainer.data_path = data_path
        X_train, y_train, feats = trainer.prepare_features(train_df)
        X_test, y_test, _ = trainer.prepare_features(test_df)
        trainer.feature_names = feats
        trainer.train(X_train, y_train, X_test, y_test)
        metrics = trainer.evaluate(X_test, y_test)
        results[patch] = metrics
        trainer.save_model(f"draft_model_{trainer.model_type}_patch_{patch.replace('.', '_')}")

    print(f"\n{'='*55}")
    print("PER-PATCH SUMMARY")
    print(f"{'='*55}")
    for patch, m in sorted(results.items()):
        print(f"  Patch {patch}: AUC={m['roc_auc']:.4f} acc={m['accuracy']:.4f} n_val={m.get('_n', '?')}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train draft prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--data", type=str, default=str(ML_ROOT / "datasets" / "processed" / "draft_features.parquet"),
                        help="Processed features parquet file")
    parser.add_argument("--model-dir", type=str, default=str(ML_ROOT / "models"),
                        help="Model output directory")
    parser.add_argument("--model-type", type=str, default="xgboost",
                        choices=["xgboost", "lightgbm"],
                        help="Model type to train")
    parser.add_argument("--test-size", type=float, default=0.1,
                        help="Test split ratio")
    parser.add_argument("--train-frac", type=float, default=0.8,
                        help="Train split ratio (time-based only)")
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="Validation split ratio (used for early stopping)")
    parser.add_argument("--time-split", action="store_true", default=True,
                        help="Use time-based split (chronological)")
    parser.add_argument("--no-time-split", action="store_false", dest="time_split",
                        help="Use random split instead of time-based")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model config JSON")
    parser.add_argument("--per-patch", action="store_true",
                        help="Train a separate model for each patch (patch-aware)")

    args = parser.parse_args()
    
    # Load custom config if provided
    config = None
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = json.load(f)
    
    print("="*60)
    print(f"MODEL TRAINER - {args.model_type.upper()}")
    print("="*60)
    print(f"Data: {args.data}")
    print(f"Model dir: {args.model_dir}")
    print(f"Test size: {args.test_size}")
    print(f"Time split: {args.time_split}")
    print(f"Per-patch: {args.per_patch}")
    print()
    
    if args.per_patch:
        train_per_patch(
            data_path=Path(args.data),
            model_dir=Path(args.model_dir),
            model_type=args.model_type,
            test_size=args.test_size,
            config=config,
        )
        return
    
    try:
        trainer, metrics = train_model(
            data_path=Path(args.data),
            model_dir=Path(args.model_dir),
            model_type=args.model_type,
            test_size=args.test_size,
            time_split=args.time_split,
            config=config,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
        )
        
        print(f"\n{'='*55}")
        print("TRAINING COMPLETE")
        print(f"{'='*55}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"LogLoss: {metrics['log_loss']:.4f}")
        print(f"Brier:   {metrics['brier_score']:.4f}")
        print(f"Model saved to: {args.model_dir}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    from datetime import datetime
    main()