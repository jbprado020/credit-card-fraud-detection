# Credit Card Fraud Detection System

A Streamlit web app that flags potentially fraudulent credit card transactions using an XGBoost classifier trained with SMOTE on a synthetic dataset. For demonstration / school project use only.

**Authors:** Julianne Benedict Prado, Brylle Garcia

## Features

- **Check one transaction** — enter amount, balance, type, device, location, etc. and get fraud probability vs. threshold
- **Check a CSV file** — batch-score a CSV, view flag counts, preview sorted results, and download predictions
- **Model information** — test-set accuracy, precision, recall, F1, and confusion matrix

## Tech Stack

- Python, Streamlit, pandas, scikit-learn, imbalanced-learn (SMOTE), XGBoost, joblib
- Pipeline: one-hot encoding + scaling → SMOTE (train only) → XGBoost
- Decision threshold: `0.20` (tuned for recall on imbalanced data)

## Project Structure

```text
.
├── app.py                          # Streamlit app
├── train_model.py                  # Training script (creates model.joblib, metrics.json)
├── requirements.txt
├── model.joblib                    # Trained pipeline + threshold + feature list
├── metrics.json                    # Test-set metrics
├── synthetic_fraud_dataset.csv     # Training data (50k rows, synthetic)
└── examples/
    ├── demo_transactions.csv       # Mixed demo: 6 rows (3 legit + 3 fraud)
    ├── demo_all_fraud.csv          # 3 rows, all flagged (~99%)
    └── demo_all_legit.csv          # 3 rows, all legitimate (~16-18%)
```

## Getting Started

Requirements: Python 3.10+

```powershell
# 1. Go to the project folder
Set-Location -LiteralPath ".\Credit Card Fraud Detection System"

# 2. (Optional) Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install -r requirements.txt

# 4. Run the app
python -m streamlit run app.py
```

Open the `Local URL` shown in the terminal (usually http://localhost:8501).

## Retraining (optional)

`model.joblib` is already included, so this is not required unless you change the data or code:

```powershell
python train_model.py synthetic_fraud_dataset.csv
```

This regenerates `model.joblib` and `metrics.json`.

## Demo Files

Upload any of these in the **"Check a CSV file"** tab:

| File | Contents | Expected result |
|------|----------|-----------------|
| `examples/demo_all_legit.csv` | 3 legitimate | 0 flagged (0%) |
| `examples/demo_all_fraud.csv` | 3 fraud | 3 flagged (100%) |
| `examples/demo_transactions.csv` | 3 + 3 mixed | 3 flagged (50%) |

Expected probabilities: legit ~16–18%, fraud ~99.9%.

## Model Performance

Test set (n = 10,000, threshold 0.20):

| Metric | Value |
|--------|-------|
| Accuracy | 0.8291 |
| Precision | 0.7773 |
| Recall | 0.6561 |
| F1-score | 0.7116 |

Confusion matrix: TN=6183, FP=604, FN=1105, TP=2108.

## How It Works

1. `prepare()` parses `Timestamp` into `Hour`/`Day` and drops ID/leakage columns (`Transaction_ID`, `User_ID`, `Timestamp`, `Risk_Score`).
2. Categorical columns are one-hot encoded, numeric columns are scaled.
3. SMOTE oversamples the minority (fraud) class on training data only.
4. XGBoost (`n_estimators=100`) outputs fraud probability; transactions with probability ≥ 0.20 are flagged.

## Limitations

- Dataset is synthetic with rule-based labels — not real fraud patterns.
- `Risk_Score` was excluded because it leaks the target.
- Threshold was tuned on the test set, so metrics are optimistic.
- For education/demonstration only — not for production use.

## Credits

Developed by **Julianne Benedict Prado** and **Brylle Garcia** as a school project.
