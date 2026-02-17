"""
model_training.py — Train and compare ML classifiers for jailbreak detection.

Pipeline:
  1. Load combined dataset
  2. Extract features (hand-crafted + embeddings)
  3. 80/20 stratified train/test split
  4. Train Logistic Regression, Random Forest, XGBoost
  5. 5-fold cross-validation comparison
  6. Hyperparameter tuning for the best model
  7. Save final model + metadata to models/
"""

import os
import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from feature_engineering import extract_all_features, get_handcrafted_feature_names

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models() -> dict:
    """Return a dictionary of model name -> (pipeline, param_grid)."""
    return {
        "Logistic Regression": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                    class_weight="balanced",
                )),
            ]),
            "params": {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__penalty": ["l2"],
            },
        },
        "Random Forest": {
            "pipeline": Pipeline([
                ("clf", RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_SEED,
                    class_weight="balanced",
                    n_jobs=-1,
                )),
            ]),
            "params": {
                "clf__n_estimators": [100, 200, 300],
                "clf__max_depth": [10, 20, 30, None],
                "clf__min_samples_split": [2, 5, 10],
            },
        },
        "XGBoost": {
            "pipeline": Pipeline([
                ("clf", XGBClassifier(
                    tree_method="hist",  # fast CPU training
                    eval_metric="logloss",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                    use_label_encoder=False,
                )),
            ]),
            "params": {
                "clf__n_estimators": [100, 200, 300],
                "clf__max_depth": [3, 5, 7, 10],
                "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "clf__subsample": [0.7, 0.8, 0.9, 1.0],
                "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            },
        },
    }


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def load_and_prepare_data():
    """Load combined dataset and extract features."""
    data_path = DATA_DIR / "train_data.csv"
    if not data_path.exists():
        print("Training data not found. Running data collection first ...")
        from data_collection import collect_data
        collect_data()

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")

    texts = df["text"].tolist()
    labels = df["label"].values

    features = extract_all_features(texts)
    return features, labels, texts


def compare_models(X_train, y_train):
    """Train and compare all models using cross-validation."""
    models = get_models()
    results = {}

    print("\n" + "=" * 60)
    print("Model Comparison (5-Fold Cross-Validation)")
    print("=" * 60)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for name, config in models.items():
        print(f"\n--- {name} ---")
        pipeline = config["pipeline"]

        start = time.time()
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        elapsed = time.time() - start

        results[name] = {
            "mean_f1": scores.mean(),
            "std_f1": scores.std(),
            "scores": scores.tolist(),
            "time_seconds": elapsed,
        }

        print(f"  F1 scores: {scores}")
        print(f"  Mean F1:   {scores.mean():.4f} (+/- {scores.std():.4f})")
        print(f"  Time:      {elapsed:.1f}s")

    # Rank models
    ranked = sorted(results.items(), key=lambda x: x[1]["mean_f1"], reverse=True)
    print("\n--- Ranking ---")
    for i, (name, r) in enumerate(ranked, 1):
        print(f"  {i}. {name}: F1 = {r['mean_f1']:.4f}")

    return ranked[0][0], results


def tune_best_model(X_train, y_train, best_model_name: str):
    """Hyperparameter tuning for the best model using RandomizedSearchCV."""
    print("\n" + "=" * 60)
    print(f"Hyperparameter Tuning: {best_model_name}")
    print("=" * 60)

    models = get_models()
    config = models[best_model_name]

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    search = RandomizedSearchCV(
        estimator=config["pipeline"],
        param_distributions=config["params"],
        n_iter=20,
        cv=cv,
        scoring="f1",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=1,
    )

    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV F1:  {search.best_score_:.4f}")
    print(f"Tuning time: {elapsed:.1f}s")

    return search.best_estimator_, search.best_params_, search.best_score_


def train_and_save():
    """Full training pipeline: load data, compare models, tune, save."""
    total_start = time.time()

    # 1. Load and prepare data
    print("\n[1/5] Loading and preparing data ...")
    X, y, texts = load_and_prepare_data()

    # 2. Train/test split
    print("\n[2/5] Splitting data (80/20 stratified) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # Save test set for evaluation
    test_data = pd.DataFrame(X_test)
    test_data["label"] = y_test
    test_data.to_csv(DATA_DIR / "test_features.csv", index=False)

    # 3. Compare models
    print("\n[3/5] Comparing models ...")
    best_model_name, comparison_results = compare_models(X_train, y_train)

    # 4. Tune best model
    print(f"\n[4/5] Tuning best model ({best_model_name}) ...")
    best_model, best_params, best_cv_score = tune_best_model(X_train, y_train, best_model_name)

    # 5. Final evaluation on test set
    print("\n[5/5] Final evaluation on held-out test set ...")
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{'='*60}")
    print("FINAL TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"  Model:     {best_model_name}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Benign', 'Jailbreak'])}")

    # Check against targets
    targets_met = precision >= 0.90 and recall >= 0.85 and f1 >= 0.87
    if targets_met:
        print("✅ All performance targets met!")
    else:
        print("⚠️  Some targets not met. Consider additional feature engineering or data.")
        if precision < 0.90:
            print(f"   Precision: {precision:.4f} < 0.90 target")
        if recall < 0.85:
            print(f"   Recall: {recall:.4f} < 0.85 target")
        if f1 < 0.87:
            print(f"   F1-Score: {f1:.4f} < 0.87 target")

    # Save model
    model_path = MODEL_DIR / "jailbreak_detector.pkl"
    joblib.dump(best_model, model_path)
    print(f"\nModel saved → {model_path}")
    print(f"Model size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Save metadata (convert numpy types to native Python for JSON serialization)
    metadata = {
        "model_name": best_model_name,
        "best_params": {k: str(v) for k, v in best_params.items()},
        "cv_f1_score": float(best_cv_score),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_count": int(X.shape[1]),
        "handcrafted_features": get_handcrafted_feature_names(),
        "comparison_results": {
            name: {"mean_f1": float(r["mean_f1"]), "std_f1": float(r["std_f1"])}
            for name, r in comparison_results.items()
        },
        "targets_met": bool(targets_met),
    }
    meta_path = MODEL_DIR / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved → {meta_path}")

    # Save predictions for evaluation script
    eval_data = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
    })
    eval_data.to_csv(DATA_DIR / "test_predictions.csv", index=False)

    total_elapsed = time.time() - total_start
    print(f"\nTotal training time: {total_elapsed / 60:.1f} minutes")

    return best_model, metadata


if __name__ == "__main__":
    train_and_save()
