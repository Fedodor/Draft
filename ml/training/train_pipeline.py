#!/usr/bin/env python3
"""
Smart Draft Assistant - Automated ML Training Pipeline
======================================================

This script automates the entire ML training process:
1. Collects match data from OpenDota API
2. Processes raw matches into ML-ready features
3. Trains XGBoost model with time-based validation
4. Exports model to ONNX for inference
5. Generates training report

Usage:
    python train_pipeline.py --matches 5000 --patch 7.41 --api-key YOUR_KEY
    python train_pipeline.py --matches 10000 --pro-only --output-dir ml/models/v2
"""

import argparse
import asyncio
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split

try:
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Import enhanced modules
import sys
from pathlib import Path
ml_root = Path(__file__).parent.parent
sys.path.insert(0, str(ml_root))
sys.path.insert(0, str(ml_root / "training"))
sys.path.insert(0, str(ml_root / "feature-engineering"))
from settings import get_opendota_key
from data_collector import OpenDotaCollector
from processor import FeatureProcessor
from training.trainer import ModelTrainer

# Suppress warnings
warnings.filterwarnings('ignore')


async def run_pipeline(args):
    """Run the complete training pipeline"""
    
    print("="*60)
    print("SMART DRAFT ASSISTANT - ML TRAINING PIPELINE")
    print("="*60)
    print(f"Started at: {datetime.utcnow().isoformat()}")
    
    # Setup directories
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    model_dir = Path(args.model_dir)
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # STEP 1: DATA COLLECTION
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 1: DATA COLLECTION")
    print(f"{'='*60}")
    
    async with OpenDotaCollector(
        api_key=args.api_key,
        rate_limit=args.rate_limit,
        max_concurrent=args.max_concurrent
    ) as collector:
        allpick_target = args.allpick_matches
        if allpick_target is None and args.include_ranked:
            allpick_target = max(1000, args.matches // 2)
            print(f"NOTE: --allpick-matches not set, defaulting to {allpick_target}")
        if allpick_target is None:
            allpick_target = 0

        metadata = await collector.collect_matches_paginated(
            target_count=args.matches,
            output_dir=raw_dir,
            patches=args.patches,
            include_pro=args.include_pro,
            include_ranked=args.include_ranked,
            min_mmr=args.min_mmr,
            allpick_target=allpick_target,
            allpick_mmr=args.allpick_mmr,
            resume=not args.no_resume
        )
    
    if len(metadata.get("match_ids", [])) < args.min_matches:
        print(f"WARNING: Only collected {len(metadata.get('match_ids', []))} matches (min: {args.min_matches})")
        if not args.force:
            print("Use --force to continue anyway")
            return 1
    
    # -------------------------------------------------------------------------
    # STEP 2: FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 2: FEATURE ENGINEERING")
    print(f"{'='*60}")
    
    processor = FeatureProcessor(
        raw_dir=raw_dir,
        output_dir=processed_dir,
        include_bans=not args.no_bans,
        include_semantics=not args.no_semantics,
        include_meta=not args.no_meta
    )
    df = processor.process_all(limit=args.limit)
    
    if len(df) < args.min_matches:
        print(f"WARNING: Only {len(df)} valid matches after processing (min: {args.min_matches})")
        if not args.force:
            return 1
    
    # -------------------------------------------------------------------------
    # STEP 3: MODEL TRAINING
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 3: MODEL TRAINING")
    print(f"{'='*60}")
    
    trainer = ModelTrainer(
        processed_dir / "draft_features.parquet",
        model_dir,
        model_type=args.model_type,
        config=args.model_config
    )
    
    # Load and split data
    df = trainer.load_data()
    
    if args.time_split:
        train_df, test_df = trainer.time_based_split(df, test_size=args.test_size)
    else:
        train_df, test_df = train_test_split(
            df, test_size=args.test_size, random_state=42, 
            stratify=df['radiant_win']
        )
    
    X_train, y_train, _ = trainer.prepare_features(train_df)
    X_test, y_test, _ = trainer.prepare_features(test_df)
    
    # Train
    trainer.train(X_train, y_train, X_test, y_test)
    
    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)
    
    # Feature importance
    fi_df = trainer.feature_importance(top_n=50)
    
    # Save model
    model_name = f"draft_model_{trainer.model_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    trainer.save_model(model_name)
    
    # -------------------------------------------------------------------------
    # STEP 4: GENERATE REPORT
    # -------------------------------------------------------------------------
    if True:  # DEFAULT_CONFIG['output']['generate_report']:
        print(f"\n{'='*60}")
        print("STEP 4: GENERATING REPORT")
        print(f"{'='*60}")
        
        report = {
            "pipeline_version": "1.0",
            "run_date": datetime.utcnow().isoformat(),
            "config": {
                "matches_requested": args.matches,
                "matches_collected": len(metadata.get('match_ids', [])),
                "matches_processed": len(df),
                "patches": args.patches,
                "include_pro": args.include_pro,
                "include_ranked": args.include_ranked,
                "min_mmr": args.min_mmr,
                "time_split": args.time_split,
                "test_size": args.test_size,
            },
            "model": model_name,
            "metrics": metrics,
            "top_features": fi_df.to_dict('records') if not fi_df.empty else [],
            "data_stats": {
                "total_matches": len(df),
                "radiant_winrate": float(df['radiant_win'].mean()),
                "avg_duration": float(df['duration'].mean()),
                "patches": df['patch'].value_counts().head(10).to_dict() if 'patch' in df.columns else {},
            }
        }
        
        report_path = model_dir / f"{model_name}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {report_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"LogLoss: {metrics['log_loss']:.4f}")
    print(f"Brier:   {metrics['brier_score']:.4f}")
    print(f"Model saved to: {model_dir}")
    print(f"Finished at: {datetime.utcnow().isoformat()}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Smart Draft Assistant - Automated ML Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data collection args
    parser.add_argument("--matches", type=int, default=5000,
                        help="Target number of matches to collect")
    parser.add_argument("--min-matches", type=int, default=1000,
                        help="Minimum matches required to proceed")
    parser.add_argument("--patches", nargs="+", default=["7.41", "7.40", "7.39", "7.38", "7.37", "7.36", "7.35"],
                        help="Patches to collect from (newest first)")
    parser.add_argument("--include-pro", action=argparse.BooleanOptionalAction, default=True,
                        help="Include professional matches")
    parser.add_argument("--include-ranked", action=argparse.BooleanOptionalAction, default=True,
                        help="Include ranked public matches (captain's mode + All Pick)")
    parser.add_argument("--min-mmr", type=int, default=4000,
                        help="Minimum average MMR for captain's mode matches")
    parser.add_argument("--allpick-matches", type=int, default=None,
                        help="Target number of All Pick matches for hero matchup stats "
                             "(0 = skip; default = matches//2 when ranked enabled)")
    parser.add_argument("--allpick-mmr", type=int, default=3000,
                        help="Minimum average MMR for All Pick matches")
    parser.add_argument("--rate-limit", type=float, default=1.0,
                        help="Requests per second to OpenDota API")
    parser.add_argument("--max-concurrent", type=int, default=3,
                        help="Max concurrent detail requests")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenDota API key for higher rate limits")
    parser.add_argument("--no-resume", action="store_true",
                        help="Don't resume from existing data")
    parser.add_argument("--force", action="store_true",
                        help="Continue even if minimum matches not met")
    
    # Paths (defaults anchored to ML root so CWD does not affect location)
    parser.add_argument("--raw-dir", type=str, default=str(ml_root / "datasets" / "raw"),
                        help="Raw data directory")
    parser.add_argument("--processed-dir", type=str, default=str(ml_root / "datasets" / "processed"),
                        help="Processed data directory")
    parser.add_argument("--model-dir", type=str, default=str(ml_root / "models"),
                        help="Model output directory")
    
    # Feature args
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of matches to process")
    parser.add_argument("--no-bans", action="store_true",
                        help="Exclude ban features")
    parser.add_argument("--no-semantics", action="store_true",
                        help="Exclude semantic features")
    parser.add_argument("--no-meta", action="store_true",
                        help="Exclude meta features")
    
    # Training args
    parser.add_argument("--model-type", type=str, default="xgboost",
                        choices=["xgboost", "lightgbm"],
                        help="Model type to train")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test split ratio")
    parser.add_argument("--time-split", action="store_true", default=True,
                        help="Use time-based split (chronological)")
    parser.add_argument("--no-time-split", action="store_false", dest="time_split",
                        help="Use random split instead of time-based")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model config JSON")
    
    args = parser.parse_args()
    args.api_key = args.api_key or get_opendota_key()
    
    # Load model config if provided (default first so a bad path cannot leave it unset)
    args.model_config = {
        "n_estimators": 2000,
        "max_depth": 8,
        "learning_rate": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 150,
        "random_state": 42,
        "n_jobs": -1,
    }
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            parser.error(f"--config file not found: {config_path}")
        with open(config_path) as f:
            args.model_config.update(json.load(f))
    
    print("="*60)
    print(f"SMART DRAFT ASSISTANT - ML TRAINING PIPELINE")
    print("="*60)
    print(f"Matches: {args.matches}")
    print(f"Patches: {args.patches}")
    print(f"Pro: {args.include_pro}")
    print(f"Ranked: {args.include_ranked}")
    print(f"Min MMR: {args.min_mmr}")
    print(f"Model: {args.model_type}")
    print()
    
    # Run pipeline
    try:
        exit_code = asyncio.run(run_pipeline(args))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()