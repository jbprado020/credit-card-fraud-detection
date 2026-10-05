"""Train the final model from the notebook and save it for the Streamlit app.

Usage:  python train_model.py synthetic_fraud_dataset.csv
Creates: model.joblib, metrics.json
Pipeline = one-hot + StandardScaler -> SMOTE (train only) -> XGBoost (default settings)
Risk_Score is excluded (target leakage); the decision threshold is 0.20.
"""
import sys, json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, average_precision_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

THRESHOLD = 0.20
CAT_COLS = ["Transaction_Type", "Device_Type", "Location",
            "Merchant_Category", "Card_Type", "Authentication_Method"]

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Hour"] = df["Timestamp"].dt.hour
    df["Day"] = df["Timestamp"].dt.day
    return df.drop(columns=["Transaction_ID", "User_ID", "Timestamp", "Risk_Score"],
                   errors="ignore")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "synthetic_fraud_dataset.csv"
    df = prepare(pd.read_csv(path))
    X, y = df.drop(columns=["Fraud_Label"]), df["Fraud_Label"]
    num_cols = [c for c in X.columns if c not in CAT_COLS]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("num", StandardScaler(), num_cols)])
    model = ImbPipeline([
        ("prep", prep),
        ("smote", SMOTE(random_state=42)),
        ("clf", XGBClassifier(n_estimators=100, learning_rate=0.1,
                              random_state=42, eval_metric="logloss"))])
    model.fit(X_tr, y_tr)

    probs = model.predict_proba(X_te)[:, 1]
    pred = (probs >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    metrics = {
        "threshold": THRESHOLD,
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "pr_auc": average_precision_score(y_te, probs),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_test": int(len(y_te)),
    }
    joblib.dump({"model": model, "threshold": THRESHOLD,
                 "features": list(X.columns)}, "model.joblib")
    json.dump(metrics, open("metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))
