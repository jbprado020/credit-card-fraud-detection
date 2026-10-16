"""Streamlit UI: Credit Card Fraud Detection (XGBoost + SMOTE).
Every score comes from the trained model; no hand-written fraud rules.
Run:  python train_model.py synthetic_fraud_dataset.csv   then   streamlit run app.py
"""
import json, os
from datetime import date, time

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud Detection System", page_icon="🛡️", layout="wide")

# ------------------------------------------------------------------ style
st.markdown("""
<style>
.block-container{padding-top:1.4rem;max-width:1200px}
.hero{background:linear-gradient(120deg,#7b1e2b 0%,#b3263a 60%,#d64545 100%);color:#fff;
      padding:1.4rem 1.8rem;border-radius:16px;margin-bottom:1.2rem}
.hero h1{margin:0;font-size:1.9rem;color:#fff}
.hero p{margin:.3rem 0 0;opacity:.92;font-size:.95rem}
.card{border-radius:16px;padding:1.4rem 1.6rem;color:#fff;text-align:center}
.card .big{font-size:3.2rem;font-weight:800;line-height:1.1}
.card .lbl{font-size:.85rem;letter-spacing:.08em;text-transform:uppercase;opacity:.9}
.card .verdict{font-size:1.15rem;font-weight:700;margin-top:.5rem}
.bar{height:12px;background:rgba(255,255,255,.3);border-radius:8px;margin-top:.9rem;position:relative}
.bar>div{height:100%;background:#fff;border-radius:8px}
.tick{position:absolute;top:-4px;width:2px;height:20px;background:#000;opacity:.55}
.chip{display:inline-block;padding:.25rem .7rem;margin:.15rem .25rem .15rem 0;border-radius:999px;
      font-size:.82rem;background:#f1f1f4;color:#333}
.chip.warn{background:#fde8e8;color:#9b1c1c}
.chip.ok{background:#e6f4ea;color:#1e6b34}
</style>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ model
@st.cache_resource
def load():
    if not os.path.exists("model.joblib"):
        return None, None
    metrics = json.load(open("metrics.json")) if os.path.exists("metrics.json") else None
    return joblib.load("model.joblib"), metrics

bundle, metrics = load()
if bundle is None:
    st.error("model.joblib not found. Run:  python train_model.py synthetic_fraud_dataset.csv")
    st.stop()
THRESH, FEATURES, BASELINE = bundle["threshold"], bundle["features"], bundle["baseline"]

def predict(df: pd.DataFrame):
    return bundle["model"].predict_proba(df[FEATURES])[:, 1]

def influence(row: pd.DataFrame):
    """Model-based explanation: change in score when one input is replaced by a typical value."""
    p = float(predict(row)[0]); out = {}
    for f in FEATURES:
        alt = row.copy(); alt[f] = BASELINE[f]
        out[f] = p - float(predict(alt)[0])
    return pd.Series(out)

# ------------------------------------------------------------------ defaults / presets
DEFAULTS = dict(amount=120.0, balance=45000.0, ttype="POS", merchant="Groceries", ip=False, device="Mobile",
                location="New York", card="Visa", auth="PIN", card_age=60, prev=False, daily=3, avg7=300.0,
                failed=0, dist=40.0, d=date.today(), t=time(14, 30))
PRESETS = {
    "✅ Typical purchase": DEFAULTS,
    "🚨 Suspicious activity": dict(DEFAULTS, amount=2400.0, ttype="Online", merchant="Electronics",
                                   ip=True, device="Laptop", location="Mumbai", auth="Password", prev=True,
                                   daily=11, avg7=150.0, failed=4, dist=3800.0, t=time(3, 10)),
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🛡️ Fraud Detector")
    st.caption("XGBoost + SMOTE · University of Mindanao")
    st.divider()
    st.markdown("**Quick examples**")
    for name in PRESETS:
        st.button(name, on_click=apply_preset, args=(name,), width="stretch")
    st.divider()
    st.metric("Decision threshold", f"{THRESH:.2f}")
    if metrics:
        st.metric("Precision", f"{metrics['precision']:.3f}")
        st.metric("Recall", f"{metrics['recall']:.3f}")
    st.caption("Trained on a synthetic dataset. Demonstration only.")

st.markdown(f"""<div class="hero"><h1>Credit Card Fraud Detection System</h1>
<p>Enter a transaction or upload a file. The model estimates the fraud probability and flags anything at or above {THRESH:.0%}.</p></div>""",
            unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["🔍 Single transaction", "📄 Batch (CSV)", "📊 Model info"])

# ------------------------------------------------------------------ tab 1
with tab1:
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Transaction**")
            a, b = st.columns(2)
            a.number_input("Amount", 0.0, 1_000_000.0, step=10.0, key="amount")
            b.number_input("Account balance", 0.0, 10_000_000.0, step=1000.0, key="balance")
            a.selectbox("Type", ["POS", "Online", "ATM Withdrawal", "Bank Transfer"], key="ttype")
            b.selectbox("Merchant category", ["Groceries", "Restaurants", "Clothing", "Electronics", "Travel"], key="merchant")
            a.selectbox("Location", ["New York", "London", "Tokyo", "Mumbai", "Sydney"], key="location")
            b.number_input("Distance from usual location (km)", 0.0, 20000.0, step=10.0, key="dist")
            a.date_input("Date", key="d")
            b.time_input("Time", key="t")
        with st.container(border=True):
            st.markdown("**Card and device**")
            a, b = st.columns(2)
            a.selectbox("Card type", ["Visa", "Mastercard", "Amex", "Discover"], key="card")
            b.number_input("Card age (months)", 0, 600, key="card_age")
            a.selectbox("Device", ["Mobile", "Laptop", "Tablet"], key="device")
            b.selectbox("Authentication", ["PIN", "OTP", "Password", "Biometric"], key="auth")
            st.toggle("IP address flagged", key="ip")
        with st.container(border=True):
            st.markdown("**Recent behaviour**")
            a, b, c = st.columns(3)
            a.number_input("Transactions today", 0, 100, key="daily")
            b.number_input("Avg amount (7 days)", 0.0, 1_000_000.0, step=10.0, key="avg7")
            c.number_input("Failed attempts (7 days)", 0, 20, key="failed")
            st.toggle("Previous fraudulent activity", key="prev")
        go = st.button("Analyze transaction", type="primary", width="stretch")

    with right:
        ss = st.session_state
        if go:
            row = pd.DataFrame([{
                "Transaction_Amount": ss.amount, "Transaction_Type": ss.ttype, "Account_Balance": ss.balance,
                "Device_Type": ss.device, "Location": ss.location, "Merchant_Category": ss.merchant,
                "IP_Address_Flag": int(bool(ss.ip)), "Previous_Fraudulent_Activity": int(bool(ss.prev)),
                "Daily_Transaction_Count": ss.daily, "Avg_Transaction_Amount_7d": ss.avg7,
                "Failed_Transaction_Count_7d": ss.failed, "Card_Type": ss.card, "Card_Age": ss.card_age,
                "Transaction_Distance": ss.dist, "Authentication_Method": ss.auth,
                "Is_Weekend": int(ss.d.weekday() >= 5), "Hour": ss.t.hour, "Day": ss.d.day}])
            p = float(predict(row)[0])
            if p >= THRESH:
                color, verdict = "#c62828", "🚨 FLAGGED: send for review"
            elif p >= THRESH * 0.5:
                color, verdict = "#ef8f00", "⚠️ Borderline: monitor"
            else:
                color, verdict = "#2e7d32", "✅ Looks legitimate"
            st.markdown(f"""<div class="card" style="background:{color}">
              <div class="lbl">Fraud probability</div><div class="big">{p:.2%}</div>
              <div class="verdict">{verdict}</div>
              <div class="bar"><div style="width:{min(p,1)*100:.1f}%"></div>
              <span class="tick" style="left:{THRESH*100:.0f}%"></span></div>
              <div style="font-size:.75rem;margin-top:.4rem;opacity:.9">Black tick = decision threshold ({THRESH:.0%})</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("&nbsp;")
            st.markdown("**What moved this score** (percentage points vs. a typical value for each input)")
            inf = (influence(row) * 100).round(2)
            inf = inf.reindex(inf.abs().sort_values(ascending=False).index).head(8)
            st.bar_chart(inf[::-1], horizontal=True)
            st.caption("Computed from the trained model by swapping one input at a time. Small bars mean the model barely uses that input.")
        else:
            st.info("Fill in the transaction on the left, or pick a quick example from the sidebar, then press **Analyze transaction**.")

# ------------------------------------------------------------------ tab 2
with tab2:
    st.write("Upload a CSV with the dataset's columns. Transaction_ID, User_ID, Risk_Score and Fraud_Label are ignored if present.")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        raw = pd.read_csv(up)
        X = raw.drop(columns=["Transaction_ID", "User_ID", "Risk_Score", "Fraud_Label"], errors="ignore")
        if "Timestamp" in X.columns:
            ts = pd.to_datetime(X["Timestamp"])
            X = X.drop(columns=["Timestamp"]).assign(Hour=ts.dt.hour, Day=ts.dt.day)
        missing = [c for c in FEATURES if c not in X.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            probs = predict(X)
            out = raw.copy()
            out["Fraud_Probability"] = probs.round(4)
            out["Flagged"] = (probs >= THRESH).astype(int)
            a, b, c = st.columns(3)
            a.metric("Transactions", f"{len(out):,}")
            b.metric("Flagged", f"{int(out['Flagged'].sum()):,}")
            c.metric("Flag rate", f"{out['Flagged'].mean():.1%}")
            st.markdown("**Distribution of fraud probabilities**")
            hist = pd.cut(out["Fraud_Probability"], bins=[i / 10 for i in range(11)], include_lowest=True).value_counts().sort_index()
            st.bar_chart(pd.Series(hist.values, index=[f"{iv.left:.1f}-{iv.right:.1f}" for iv in hist.index]))
            st.dataframe(out.sort_values("Fraud_Probability", ascending=False), width="stretch", hide_index=True,
                         column_config={"Fraud_Probability": st.column_config.ProgressColumn(
                             "Fraud_Probability", min_value=0.0, max_value=1.0, format="%.2f")})
            st.download_button("⬇️ Download results", out.to_csv(index=False).encode(), "fraud_predictions.csv", "text/csv")

# ------------------------------------------------------------------ tab 3
with tab3:
    if metrics:
        st.subheader(f"Test-set performance (threshold {metrics['threshold']:.2f}, n = {metrics['n_test']:,})")
        a, b, c, d = st.columns(4)
        a.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        b.metric("Precision", f"{metrics['precision']:.4f}")
        c.metric("Recall", f"{metrics['recall']:.4f}")
        d.metric("F1-score", f"{metrics['f1']:.4f}")
        cfg = metrics.get("config", {})
        if cfg:
            st.caption("Selected configuration: " + ", ".join(f"{k}={v}" for k, v in cfg.items()))
        x, y = st.columns([1, 1.4], gap="large")
        with x:
            st.markdown("**Confusion matrix**")
            cm = pd.DataFrame([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
                              index=["Actual legit", "Actual fraud"], columns=["Pred. legit", "Pred. fraud"])
            st.dataframe(cm.style.background_gradient(cmap="Reds"), width="stretch")
        with y:
            st.markdown("**Top 10 feature importances**")
            names = bundle["model"].named_steps["prep"].get_feature_names_out()
            imp = pd.Series(bundle["model"].named_steps["clf"].feature_importances_, index=names)
            imp.index = [n.split("__", 1)[-1] for n in imp.index]
            st.bar_chart(imp.sort_values(ascending=False).head(10)[::-1], horizontal=True)
    else:
        st.info("Train the model to see its metrics and feature importances here.")
    if metrics and metrics.get("sensitivity_by_feature"):
        st.markdown("**How much each non-key input moves the score** (average change in fraud probability, percentage points)")
        sens = pd.Series(metrics["sensitivity_by_feature"]) * 100
        st.bar_chart(sens.sort_values()[::-1].head(12)[::-1], horizontal=True)
        st.caption("These effects are small and are learned noise: statistical tests found no real signal in these inputs. "
                   "Failed_Transaction_Count_7d is the only input that drives the decision.")
    st.markdown("""
**How it works:** one-hot encoding and scaling, SMOTE on the training data only, then XGBoost.
A transaction is flagged when its fraud probability reaches the threshold.

**Limitations:** the dataset is synthetic with rule-based labels, `Risk_Score` was removed because it leaks the
target, and the threshold was chosen on a validation split. Inputs other than failed attempts have only a small effect on the score, and that effect is learned noise, not real fraud signal.
""")
