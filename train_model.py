"""Train the final model and save it for the Streamlit app.

Usage:  python train_model.py synthetic_fraud_dataset.csv
Creates: model.joblib, metrics.json

Pipeline = one-hot + StandardScaler -> SMOTE (training data only) -> XGBoost.
Risk_Score is excluded (target leakage).

Model settings are chosen from a small, fixed set of XGBoost hyperparameter
configurations on a VALIDATION split of the training data. Among the configurations whose
PR-AUC is within PR_TOLERANCE of the best, the one whose predictions react most to the
non-key predictors is kept. No rule is hard-coded: every score comes from the trained model.
The decision threshold (maximum recall with precision >= 0.75) is also chosen on the validation
split, and the test set is used once at the end.
"""
import sys, json
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, average_precision_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

KEY = "Failed_Transaction_Count_7d"
MIN_PRECISION = 0.75
PR_TOLERANCE = 0.01          # allowed PR-AUC loss when preferring a more responsive model
CAT_COLS = ["Transaction_Type", "Device_Type", "Location",
            "Merchant_Category", "Card_Type", "Authentication_Method"]

CONFIGS = [
    dict(name="default",            n_estimators=100, learning_rate=0.10, max_depth=6, colsample_bytree=1.0),
    dict(name="colsample 0.5",      n_estimators=100, learning_rate=0.10, max_depth=6, colsample_bytree=0.5),
    dict(name="colsample 0.3",      n_estimators=100, learning_rate=0.10, max_depth=6, colsample_bytree=0.3),
    dict(name="300 trees, col 0.5", n_estimators=300, learning_rate=0.05, max_depth=6, colsample_bytree=0.5),
    dict(name="300 trees, col 0.3", n_estimators=300, learning_rate=0.05, max_depth=6, colsample_bytree=0.3),
    dict(name="deep 8, col 0.3",    n_estimators=300, learning_rate=0.05, max_depth=8, colsample_bytree=0.3),
    dict(name="deep 10, 500 trees", n_estimators=500, learning_rate=0.05, max_depth=10, colsample_bytree=0.5),
    dict(name="deep 8, lambda 0.1", n_estimators=300, learning_rate=0.05, max_depth=8, colsample_bytree=0.3, reg_lambda=0.1),
]

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Hour"] = df["Timestamp"].dt.hour
    df["Day"] = df["Timestamp"].dt.day
    return df.drop(columns=["Transaction_ID", "User_ID", "Timestamp", "Risk_Score"], errors="ignore")

def make_model(cols, cfg):
    cat = [c for c in cols if c in CAT_COLS]
    num = [c for c in cols if c not in CAT_COLS]
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                              ("num", StandardScaler(), num)])
    params = {k: v for k, v in cfg.items() if k != "name"}
    return ImbPipeline([("prep", prep), ("smote", SMOTE(random_state=42)),
                        ("clf", XGBClassifier(random_state=42, eval_metric="logloss", **params))])

def sensitivity(model, X, cols, seed=0):
    """Mean absolute change in fraud probability when ONE non-key predictor is shuffled."""
    rng = np.random.default_rng(seed)
    base = model.predict_proba(X)[:, 1]
    per = {}
    for c in cols:
        if c == KEY:
            continue
        Xp = X.copy()
        Xp[c] = rng.permutation(Xp[c].values)
        per[c] = float(np.abs(model.predict_proba(Xp)[:, 1] - base).mean())
    return float(np.mean(list(per.values()))), per

def pick_threshold(y, p):
    best_t, best_r = None, -1
    for t in np.arange(0.01, 0.99, 0.01):
        pred = (p >= t).astype(int)
        if precision_score(y, pred, zero_division=0) >= MIN_PRECISION:
            r = recall_score(y, pred, zero_division=0)
            if r > best_r:
                best_t, best_r = float(round(t, 2)), r
    return best_t if best_t is not None else 0.5

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "synthetic_fraud_dataset.csv"
    df = prepare(pd.read_csv(path))
    X, y = df.drop(columns=["Fraud_Label"]), df["Fraud_Label"]
    cols = list(X.columns)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_sub, X_val, y_sub, y_val = train_test_split(X_tr, y_tr, test_size=0.25, random_state=11, stratify=y_tr)

    print(f"{'configuration':<22}{'val PR-AUC':>12}{'sensitivity':>14}")
    rows = []
    for cfg in CONFIGS:
        m = make_model(cols, cfg).fit(X_sub, y_sub)
        pr = average_precision_score(y_val, m.predict_proba(X_val)[:, 1])
        sens, _ = sensitivity(m, X_val, cols)
        rows.append((cfg, pr, sens)); print(f"{cfg['name']:<22}{pr:>12.4f}{sens:>14.4f}")
    best_pr = max(r[1] for r in rows)
    eligible = [r for r in rows if r[1] >= best_pr - PR_TOLERANCE]
    cfg, pr_sel, sens_sel = max(eligible, key=lambda r: r[2])
    print(f"\nSelected: {cfg['name']}")

    thr = pick_threshold(y_val, make_model(cols, cfg).fit(X_sub, y_sub).predict_proba(X_val)[:, 1])
    model = make_model(cols, cfg).fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    pred = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    _, per = sensitivity(model, X_te, cols)

    metrics = {"threshold": thr, "config": cfg, "accuracy": accuracy_score(y_te, pred),
               "precision": precision_score(y_te, pred, zero_division=0), "recall": recall_score(y_te, pred),
               "f1": f1_score(y_te, pred), "pr_auc": average_precision_score(y_te, probs),
               "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "n_test": int(len(y_te)),
               "sensitivity_by_feature": per}
    baseline = {c: (X_tr[c].mode().iloc[0] if c in CAT_COLS else float(X_tr[c].median())) for c in cols}
    joblib.dump({"model": model, "threshold": thr, "features": cols, "baseline": baseline}, "model.joblib")
    json.dump(metrics, open("metrics.json", "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in metrics.items() if k != "sensitivity_by_feature"}, indent=2, default=float))
