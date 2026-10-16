"""Backup / fixed training script (does NOT overwrite original files).

Addresses professor feedback:
1. Drops leaky rule-based column Failed_Transaction_Count_7d (==4 -> 100% fraud).
2. Real hyperparameter tuning via RandomizedSearchCV (5-fold, PR-AUC).
3. Threshold tuned on VALIDATION split, reported once on TEST split.

Usage:
  python train_model_v2.py synthetic_fraud_dataset.csv
Creates:
  model_leakfree.joblib, metrics_leakfree.json
"""
import sys
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix,
                             average_precision_score)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

CAT_COLS = ["Transaction_Type", "Device_Type", "Location",
            "Merchant_Category", "Card_Type", "Authentication_Method"]

# Professor feedback: this column encodes the label (4 -> always fraud).
LEAKY_COLS = ["Failed_Transaction_Count_7d"]

MIN_PRECISION = 0.75


def prepare(df: pd.DataFrame, drop_leaky: bool = True) -> pd.DataFrame:
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Hour"] = df["Timestamp"].dt.hour
    df["Day"] = df["Timestamp"].dt.day
    drop = ["Transaction_ID", "User_ID", "Timestamp", "Risk_Score"]
    if drop_leaky:
        drop += LEAKY_COLS
    return df.drop(columns=drop, errors="ignore")


def tune_threshold(y_true, probs, min_precision=MIN_PRECISION):
    """Lowest threshold with precision >= floor, maximizing recall (on validation)."""
    best_thresh, best_rec = 0.20, -1.0
    for thresh in np.arange(0.01, 0.50, 0.01):
        pred = (probs >= thresh).astype(int)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        if prec >= min_precision and rec > best_rec:
            best_rec, best_thresh = rec, float(thresh)
    return best_thresh


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "synthetic_fraud_dataset.csv"
    df = prepare(pd.read_csv(path), drop_leaky=True)
    X, y = df.drop(columns=["Fraud_Label"]), df["Fraud_Label"]
    num_cols = [c for c in X.columns if c not in CAT_COLS]
    print(f"Features ({len(X.columns)}): {list(X.columns)}")
    print(f"Leaky cols dropped: {LEAKY_COLS}")

    # 60/20/20 split: train / validation (threshold) / test (final report)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=42, stratify=y_tmp)
    print(f"Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}")

    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("num", StandardScaler(), num_cols)])

    base = ImbPipeline([
        ("prep", prep),
        ("smote", SMOTE(random_state=42)),
        ("clf", XGBClassifier(random_state=42, eval_metric="logloss"))])

    param_dist = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [3, 5, 7, 9],
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
    }

    search = RandomizedSearchCV(
        base, param_distributions=param_dist, n_iter=10,
        scoring="average_precision",  # PR-AUC, right metric for imbalance
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        random_state=42, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    print(f"Best PR-AUC: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    model = search.best_estimator_

    # Threshold on VALIDATION only
    val_probs = model.predict_proba(X_val)[:, 1]
    threshold = tune_threshold(y_val, val_probs)
    print(f"Chosen threshold (val, prec>={MIN_PRECISION}): {threshold:.2f}")

    # Final report on TEST only (untouched until now)
    probs = model.predict_proba(X_test)[:, 1]
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    metrics = {
        "threshold": threshold,
        "min_precision_floor": MIN_PRECISION,
        "dropped_leaky_cols": LEAKY_COLS,
        "best_params": search.best_params_,
        "best_cv_pr_auc": float(search.best_score_),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "pr_auc": average_precision_score(y_test, probs),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(y_test)),
    }

    joblib.dump({"model": model, "threshold": threshold,
                 "features": list(X.columns)}, "model_leakfree.joblib")
    json.dump(metrics, open("metrics_leakfree.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))
    print("Saved: model_leakfree.joblib, metrics_leakfree.json (original files untouched)")
