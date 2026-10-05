"""Streamlit app: credit card fraud detection (XGBoost + SMOTE, threshold 0.20)."""
import json, os
from datetime import date, time

import joblib
import pandas as pd
import streamlit as st

from train_model import prepare

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="wide")

@st.cache_resource
def load():
    bundle = joblib.load("model.joblib")
    metrics = json.load(open("metrics.json")) if os.path.exists("metrics.json") else None
    return bundle, metrics

if not os.path.exists("model.joblib"):
    st.error("model.joblib not found. Run:  python train_model.py synthetic_fraud_dataset.csv")
    st.stop()

bundle, metrics = load()
model, THRESH, FEATURES = bundle["model"], bundle["threshold"], bundle["features"]

st.title("💳 Credit Card Fraud Detection System")
st.caption("XGBoost + SMOTE pipeline with a precision-constrained decision threshold "
           f"of {THRESH:.2f}. Trained on a synthetic dataset, so for demonstration only.")

tab1, tab2, tab3 = st.tabs(["Check one transaction", "Check a CSV file", "Model information"])

# ---------------------------------------------------------------- single
with tab1:
    with st.form("single"):
        c1, c2, c3 = st.columns(3)
        with c1:
            amount = st.number_input("Transaction amount", 0.0, 1_000_000.0, 100.0, step=10.0)
            balance = st.number_input("Account balance", 0.0, 10_000_000.0, 50_000.0, step=1000.0)
            ttype = st.selectbox("Transaction type", ["POS", "Online", "ATM Withdrawal", "Bank Transfer"])
            merchant = st.selectbox("Merchant category", ["Groceries", "Restaurants", "Clothing", "Electronics", "Travel"])
            ip_flag = st.selectbox("IP address flagged?", [0, 1], format_func=lambda v: "Yes" if v else "No")
        with c2:
            device = st.selectbox("Device type", ["Mobile", "Laptop", "Tablet"])
            location = st.selectbox("Location", ["New York", "London", "Tokyo", "Mumbai", "Sydney"])
            card = st.selectbox("Card type", ["Visa", "Mastercard", "Amex", "Discover"])
            auth = st.selectbox("Authentication method", ["PIN", "OTP", "Password", "Biometric"])
            card_age = st.number_input("Card age (months)", 0, 600, 60)
        with c3:
            prev = st.selectbox("Previous fraudulent activity?", [0, 1], format_func=lambda v: "Yes" if v else "No")
            daily = st.number_input("Transactions today", 0, 100, 3)
            avg7 = st.number_input("Average amount, last 7 days", 0.0, 1_000_000.0, 300.0, step=10.0)
            failed = st.number_input("Failed transactions, last 7 days", 0, 20, 0)
            dist = st.number_input("Distance from usual location (km)", 0.0, 20_000.0, 50.0, step=10.0)
        d1, d2 = st.columns(2)
        tx_date = d1.date_input("Transaction date", date.today())
        tx_time = d2.time_input("Transaction time", time(12, 0))
        go = st.form_submit_button("Check transaction", type="primary")

    if go:
        row = pd.DataFrame([{
            "Transaction_Amount": amount, "Transaction_Type": ttype, "Account_Balance": balance,
            "Device_Type": device, "Location": location, "Merchant_Category": merchant,
            "IP_Address_Flag": ip_flag, "Previous_Fraudulent_Activity": prev,
            "Daily_Transaction_Count": daily, "Avg_Transaction_Amount_7d": avg7,
            "Failed_Transaction_Count_7d": failed, "Card_Type": card, "Card_Age": card_age,
            "Transaction_Distance": dist, "Authentication_Method": auth,
            "Is_Weekend": int(tx_date.weekday() >= 5),
            "Hour": tx_time.hour, "Day": tx_date.day}])[FEATURES]
        p = float(model.predict_proba(row)[0, 1])
        st.subheader("Result")
        m1, m2 = st.columns(2)
        m1.metric("Fraud probability", f"{p:.1%}")
        m2.metric("Decision threshold", f"{THRESH:.0%}")
        st.progress(min(p, 1.0))
        if p >= THRESH:
            st.error("🚨 FLAGGED AS POSSIBLE FRAUD: send for review.")
        else:
            st.success("✅ Looks legitimate.")

# ---------------------------------------------------------------- batch
with tab2:
    st.write("Upload a CSV with the same columns as the training dataset "
             "(Transaction_ID, User_ID and Risk_Score are ignored if present).")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        raw = pd.read_csv(up)
        try:
            if "Timestamp" in raw.columns:
                X = prepare(raw).drop(columns=["Fraud_Label"], errors="ignore")
            else:
                X = raw.drop(columns=["Transaction_ID", "User_ID", "Risk_Score", "Fraud_Label"], errors="ignore")
            probs = model.predict_proba(X[FEATURES])[:, 1]
        except KeyError as e:
            st.error(f"Missing column: {e}")
            st.stop()
        out = raw.copy()
        out["Fraud_Probability"] = probs.round(4)
        out["Flagged"] = (probs >= THRESH).astype(int)
        a, b, c = st.columns(3)
        a.metric("Transactions", f"{len(out):,}")
        b.metric("Flagged as fraud", f"{int(out['Flagged'].sum()):,}")
        c.metric("Flag rate", f"{out['Flagged'].mean():.1%}")
        st.dataframe(out.sort_values("Fraud_Probability", ascending=False), use_container_width=True)
        st.download_button("Download results", out.to_csv(index=False).encode(),
                           "fraud_predictions.csv", "text/csv")

# ---------------------------------------------------------------- info
with tab3:
    if metrics:
        st.subheader(f"Test-set performance (threshold {metrics['threshold']:.2f}, n = {metrics['n_test']:,})")
        a, b, c, d = st.columns(4)
        a.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        b.metric("Precision", f"{metrics['precision']:.4f}")
        c.metric("Recall", f"{metrics['recall']:.4f}")
        d.metric("F1-score", f"{metrics['f1']:.4f}")
        st.write("Confusion matrix")
        st.table(pd.DataFrame([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
                              index=["Actual legitimate", "Actual fraud"],
                              columns=["Predicted legitimate", "Predicted fraud"]))
    st.markdown("""
**How it works:** one-hot encoding and scaling, SMOTE on the training data only, then an XGBoost
classifier. A transaction is flagged when its fraud probability is at least the threshold.

**Limitations:** the dataset is synthetic with rule-based labels, `Risk_Score` was removed because it
leaks the target, and the threshold was tuned on the test set, so the metrics are optimistic.
""")
