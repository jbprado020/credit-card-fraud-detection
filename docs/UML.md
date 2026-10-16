# UML Diagrams — Credit Card Fraud Detection System

Authors: Julianne Benedict Prado, Brylle Garcia

GitHub renders the Mermaid diagrams below automatically.

## 1. Use Case Diagram

```mermaid
flowchart LR
  User((User))
  Dev((Developer))
  UC1([Check one transaction])
  UC2([Check a CSV file])
  UC3([View model information])
  UC4([Download predictions])
  UC5([Retrain model])
  User --> UC1
  User --> UC2
  User --> UC3
  User --> UC4
  Dev --> UC5
  UC5 -. updates .-> UC3
```

## 2. Class Diagram

```mermaid
classDiagram
  class TrainScript {
    +THRESHOLD float
    +CAT_COLS list
    +prepare(df) DataFrame
    +main() void
  }
  class DataPreprocessor {
    +ColumnTransformer transformer
    +OneHotEncoder catEncoder
    +StandardScaler numScaler
    +fit(X)
    +transform(X)
  }
  class FraudPipeline {
    +DataPreprocessor prep
    +SMOTE smote
    +XGBClassifier clf
    +fit(X_train, y_train)
    +predict_proba(X) array
  }
  class StreamlitApp {
    +model FraudPipeline
    +THRESH float
    +FEATURES list
    +metrics dict
    +load() bundle
    +checkSingleTransaction(row) float
    +checkBatchCSV(df) DataFrame
    +showModelInfo() void
  }
  class TransactionData {
    +Transaction_Amount float
    +Transaction_Type string
    +Timestamp datetime
    +Account_Balance float
    +Device_Type string
    +Location string
    +Merchant_Category string
    +IP_Address_Flag int
    +Previous_Fraudulent_Activity int
    +Daily_Transaction_Count int
    +Avg_Transaction_Amount_7d float
    +Failed_Transaction_Count_7d int
    +Card_Type string
    +Card_Age int
    +Transaction_Distance float
    +Authentication_Method string
    +Is_Weekend int
    +Hour int
    +Day int
  }
  TrainScript --> DataPreprocessor : builds
  TrainScript --> FraudPipeline : trains & saves to model.joblib
  StreamlitApp --> FraudPipeline : loads from model.joblib
  StreamlitApp --> TransactionData : validates & scores
```

## 3. Sequence Diagram — Single Transaction Check

```mermaid
sequenceDiagram
  actor User
  participant UI as StreamlitApp (app.py)
  participant Prep as DataPreprocessor
  participant Pipe as FraudPipeline (XGBoost)
  User->>UI: Fill form + Submit
  UI->>Prep: Build 1-row DataFrame[FEATURES]
  Prep-->>UI: Encoded + scaled vector
  UI->>Pipe: predict_proba(row)
  Pipe-->>UI: fraud probability p
  UI->>UI: Compare p >= THRESH (0.20)
  UI-->>User: Show FLAGGED (p) or Legitimate (p)
```

## 4. Sequence Diagram — Batch CSV Check

```mermaid
sequenceDiagram
  actor User
  participant UI as StreamlitApp (app.py)
  participant Prep as prepare() (train_model.py)
  participant Pipe as FraudPipeline
  User->>UI: Upload CSV
  UI->>Prep: prepare(raw) → Hour/Day, drop IDs
  Prep-->>UI: X[FEATURES]
  UI->>Pipe: predict_proba(X)
  Pipe-->>UI: probs[]
  UI->>UI: Flagged = probs >= 0.20
  UI-->>User: Table + metrics + Download CSV
```

## 5. Activity Diagram — Training

```mermaid
flowchart TD
  A[Load synthetic_fraud_dataset.csv] --> B[prepare: parse Timestamp to Hour/Day, drop IDs + Risk_Score]
  B --> C[Split 80/20 stratified]
  C --> D[OneHot encode categoricals + Scale numerics]
  D --> E[SMOTE oversample train only]
  E --> F[Train XGBoost n_estimators=100]
  F --> G[Evaluate on test at threshold 0.20]
  G --> H[Save model.joblib + metrics.json]
```
