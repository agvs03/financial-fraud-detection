# Financial Fraud Detection

[![Python application](https://github.com/agvs03/financial-fraud-detection/actions/workflows/python-app.yml/badge.svg)](https://github.com/agvs03/financial-fraud-detection/actions/workflows/python-app.yml)
[![Pipeline](https://github.com/agvs03/financial-fraud-detection/actions/workflows/pipeline.yml/badge.svg)](https://github.com/agvs03/financial-fraud-detection/actions/workflows/pipeline.yml)
[![CodeQL](https://github.com/agvs03/financial-fraud-detection/actions/workflows/codeql.yml/badge.svg)](https://github.com/agvs03/financial-fraud-detection/actions/workflows/codeql.yml)
[![Docker Image CI](https://github.com/agvs03/financial-fraud-detection/actions/workflows/docker-image.yml/badge.svg)](https://github.com/agvs03/financial-fraud-detection/actions/workflows/docker-image.yml)
[![Python 3.10–3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end deep-learning system for detecting fraudulent credit-card
transactions on a highly imbalanced dataset (284,807 transactions, 0.172%
fraud). A TensorFlow/Keras artificial neural network is trained with SMOTE
oversampling and a validation-tuned decision threshold, benchmarked against
Logistic Regression, Random Forest, and XGBoost, and served through a FastAPI
endpoint that scores transactions pulled from PostgreSQL.

---

## Table of contents

- [Highlights](#highlights)
- [Results](#results)
- [Model architecture](#model-architecture)
- [Project structure](#project-structure)
- [Dataset](#dataset)
- [Setup](#setup)
- [Training](#training)
- [Serving the API](#serving-the-api)
- [Design notes](#design-notes)
- [Testing](#testing)

---

## Highlights

- **ANN** (Keras) reaching **~87% recall** on the fraud class with **PR-AUC ~0.85**,
  competitive with gradient-boosted trees on this tabular PCA data.
- **SMOTE** oversampling applied *only* to the training fold — no leakage into
  validation/test (this is why the numbers here are honest, not inflated — see
  [A note on "99% recall" claims](#a-note-on-inflated-recall-claims)).
- **Threshold optimization**: the decision boundary is chosen on validation
  scores (max recall subject to a precision floor), not left at the naive 0.5.
- **Baselines**: Logistic Regression, Random Forest, XGBoost, all trained and
  evaluated on identical splits for an honest comparison.
- **Full evaluation suite**: precision/recall/F1, confusion matrices, ROC-AUC,
  PR-AUC, and rendered PR/ROC curves.
- **EDA notebook** with correlation heatmaps, class-conditional distributions,
  and Mann–Whitney U hypothesis testing with effect sizes.
- **Production-style serving**: FastAPI + SQLAlchemy + PostgreSQL, Dockerized
  with `docker-compose`.

---

## Results

> These are **real numbers from an actual run** on the Kaggle ULB dataset
> (seed 42, 20% stratified test split ≈ 56,961 transactions, ~98 frauds).
> Figures vary slightly with hardware and TensorFlow version; the authoritative
> numbers for *your* run are written to `reports/metrics.json` and printed by
> `src/train.py`. Threshold tuned per-model on validation to maximize recall
> while keeping precision ≥ 0.50.

Positive class = **fraud**.

| Model                | Precision | Recall  | F1     | ROC-AUC | PR-AUC |
|----------------------|-----------|---------|--------|---------|--------|
| **ANN (this repo)**  | 0.394     | 0.867   | 0.541  | 0.966   | **0.853** |
| Logistic Regression  | 0.541     | 0.867   | **0.667** | 0.973 | 0.717  |
| Random Forest        | 0.451     | **0.898** | 0.601 | 0.969   | **0.868** |
| XGBoost              | 0.442     | 0.888   | 0.590  | **0.982** | 0.863  |

**Reading the table honestly.** Because the classes are ~580:1, **accuracy is
useless** (every model exceeds 0.999) — **recall**, **F1**, and **PR-AUC** are
what matter. Takeaways:

- The **ANN catches ~87% of fraud** (≈85 of 98 test frauds) and its **PR-AUC of
  0.853** is competitive with the tree ensembles — a real improvement from an
  earlier, smaller network that scored 0.72.
- On this **tabular PCA data, gradient-boosted trees are hard to beat**: Random
  Forest edges everyone on recall (0.898) and PR-AUC (0.868), and Logistic
  Regression happens to win F1 at this operating point. The ANN is *competitive,
  not dominant* — which is the honest result for a dataset of this shape, and we
  report it as such rather than cherry-picking.
- The ANN's precision (0.394) sits a little under the 0.50 validation floor
  because the threshold is tuned on a small validation fold (~79 frauds) and
  doesn't transfer perfectly to test — expected small-sample behavior.

Prefer a different trade-off? One line in [`config.yaml`](config.yaml)
(`threshold.min_precision`) shifts every model along its precision/recall curve.

Generated figures (after a run) land in `reports/figures/`:

- `precision_recall_curves.png` — all four models overlaid
- `roc_curves.png` — all four models overlaid
- `confusion_<model>.png` — per-model confusion matrix at its tuned threshold

### A note on inflated recall claims

You will find many fraud-detection tutorials reporting **96–99% recall**. Most
of them are measuring the wrong thing. The common mistake is applying **SMOTE to
the entire dataset *before* splitting** — so the test set is packed with
synthetic, balanced, easy-to-classify samples, and the reported recall reflects
performance on fabricated data, not real fraud.

This project applies SMOTE **only to the training fold** (see
[`src/data_preprocessing.py`](src/data_preprocessing.py)) and evaluates on the
untouched, real, 0.17%-fraud test set. That is why the recall here is ~87% and
not 99% — the numbers are honest.

It is also possible to *force* 96%+ recall by dropping the decision threshold
toward zero, but on this data that collapses precision below 1% (you flag nearly
every transaction as fraud). A recall figure is meaningless without the
precision it was bought at.

---

## Model architecture

A wide-then-narrow fully-connected network (~15k parameters). The inputs are
already PCA components, so depth buys little; capacity plus regularization
(dropout + L2 + batch norm) and class weighting do the heavy lifting.

```
                Input: 30 features  (Time, V1..V28, Amount)
                         │
        ┌────────────────▼────────────────┐
        │  Dense(128, ReLU) + L2(1e-4)     │
        │  BatchNorm → Dropout(0.25)       │
        └────────────────┬────────────────┘
        ┌────────────────▼────────────────┐
        │  Dense(64, ReLU) + L2(1e-4)      │
        │  BatchNorm → Dropout(0.25)       │
        └────────────────┬────────────────┘
        ┌────────────────▼────────────────┐
        │  Dense(32, ReLU) + L2(1e-4)      │
        │  BatchNorm → Dropout(0.25)       │
        └────────────────┬────────────────┘
                         │
                Dense(1, Sigmoid)  →  P(fraud) ∈ [0, 1]
                         │
          threshold (tuned on validation) → {legit, fraud}
```

- **Loss**: binary cross-entropy
- **Optimizer**: Adam (lr 8e-4), `ReduceLROnPlateau` on validation PR-AUC
- **Imbalance handling**: SMOTE (train fold, to ~10% fraud) **+** class weights
  `{0: 1, 1: 2}` (a light residual weight — a heavier one collapsed precision)
- **Regularization**: dropout 0.25, L2 1e-4, batch norm, early stopping
  (patience 15) restoring best weights on `val_auprc`
- **Training metrics tracked**: recall, precision, ROC-AUC, PR-AUC

All hyperparameters live in [`config.yaml`](config.yaml) — no need to edit code
to retune.

---

## Project structure

```
financial-fraud-detection/
├── config.yaml                 # all hyperparameters & paths
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml          # PostgreSQL + API
├── .env.example
│
├── data/                       # creditcard.csv goes here (gitignored)
├── models/                     # trained artifacts (gitignored)
├── reports/
│   ├── metrics.json            # written by training
│   └── figures/                # PR/ROC curves, confusion matrices
│
├── notebooks/
│   └── eda.ipynb               # full EDA + hypothesis testing
│
├── src/
│   ├── config.py               # YAML → attribute-access config
│   ├── data_preprocessing.py   # split + RobustScaler + SMOTE
│   ├── evaluate.py             # threshold tuning + metrics + plots
│   ├── train.py                # end-to-end training entry point
│   ├── utils.py                # seeds, IO helpers
│   └── models/
│       ├── ann.py              # Keras ANN builder
│       └── baselines.py        # LogReg / RF / XGBoost factories
│
├── api/
│   ├── main.py                 # FastAPI app + routes
│   ├── predict.py              # model/scaler/threshold loader + inference
│   ├── database.py             # SQLAlchemy PostgreSQL access
│   ├── schemas.py              # Pydantic request/response models
│   └── config.py              # API settings (env-driven)
│
├── db/
│   ├── schema.sql              # transactions + fraud_scores tables
│   └── load_csv.py             # bulk-load the CSV into PostgreSQL
│
├── scripts/
│   └── make_sample_data.py     # synthetic CSV for smoke tests
│
└── tests/
    └── test_api.py             # API tests with stubbed model/db
```

---

## Dataset

The **repository does not ship any data** — `data/` is empty by design (the
raw CSV is large and carries its own license, so it's git-ignored). You supply
the data yourself. There are three ways to do that.

### Option A — the reference Kaggle dataset (recommended)

This project was built around the
[**Credit Card Fraud Detection**](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
dataset from the Machine Learning Group at ULB.

- 284,807 transactions, 492 frauds (0.172%)
- Features `V1`..`V28` are PCA transforms of the original (confidential)
  features; `Time` (seconds since first transaction) and `Amount` are raw;
  `Class` is the label (1 = fraud).

**Download with the Kaggle CLI** (needs a free Kaggle account):

```bash
# 1. Install the CLI
pip install kaggle

# 2. Create an API token: kaggle.com -> Account -> "Create New API Token".
#    This downloads kaggle.json. Put it where the CLI expects it:
#      Windows:  %USERPROFILE%\.kaggle\kaggle.json
#      macOS/Linux:  ~/.kaggle/kaggle.json   (then: chmod 600 ~/.kaggle/kaggle.json)

# 3. Download + unzip straight into data/
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
```

Or download `creditcard.csv` manually from the Kaggle page and drop it at
`data/creditcard.csv`. Either way the pipeline finds it automatically.

### Option B — your own dataset

You can point the pipeline at **any** binary-classification CSV. Nothing about
the training code is hard-wired to the Kaggle column names — the ANN sizes its
input layer from your data, and the columns to scale and the label name are
read from [`config.yaml`](config.yaml).

To use your own file:

1. Put your CSV anywhere under `data/` (e.g. `data/my_transactions.csv`).
2. Edit `config.yaml`:
   ```yaml
   data:
     raw_path: "data/my_transactions.csv"
     target_col: "is_fraud"        # <- name of YOUR binary label column (0/1)
   preprocessing:
     robust_scale_cols: ["amount", "account_age_days"]  # raw-scale numeric cols
   ```
3. Every other column is fed to the models as-is, so make sure the CSV is
   already numeric (encode categoricals beforehand) and free of an index/id
   column you don't want as a feature.
4. Run `python -m src.train` — splits, SMOTE, scaling, the ANN, all three
   baselines, threshold tuning, and evaluation adapt to your schema
   automatically.

> **Serving your own schema.** The training side is schema-agnostic, but the
> FastAPI layer is wired to the Kaggle columns for convenience. If you serve a
> different schema, update `feature_order`/`robust_scale_cols` in
> [`api/config.py`](api/config.py), the `TransactionFeatures` model in
> [`api/schemas.py`](api/schemas.py), and the table in
> [`db/schema.sql`](db/schema.sql) to match your columns.

### Option C — synthetic data (no account, just to see it run)

Generate a schema-compatible fake CSV to exercise the whole pipeline end to end.
**The resulting metrics are meaningless** — this only proves the code runs:

```bash
python scripts/make_sample_data.py --rows 30000 --out data/creditcard.csv
```

---

## Setup

Requires **Python 3.10 or 3.11** (TensorFlow does not yet publish wheels for
3.12+). Check with `python --version` first.

```bash
# 1. Clone
git clone https://github.com/<your-username>/financial-fraud-detection.git
cd financial-fraud-detection

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Windows (cmd):         .venv\Scripts\activate.bat
# macOS/Linux:           source .venv/bin/activate

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Copy the env template (only needed if you use the PostgreSQL API)
cp .env.example .env          # Windows: copy .env.example .env
```

Then add a dataset (see [Dataset](#dataset)) and you're ready to train.

**Verify the install without any data or database:**

```bash
pytest        # API tests stub the model + DB, so they run on a fresh clone
```

---

## Training

Once a dataset is in place (the path set in `config.yaml → data.raw_path`,
default `data/creditcard.csv`):

```bash
python -m src.train
```

This will:

1. Build stratified train/val/test splits, RobustScale `Amount`/`Time`, and
   SMOTE-resample the training fold.
2. Train the ANN and tune its decision threshold on validation.
3. Train Logistic Regression, Random Forest, and XGBoost on the same data.
4. Evaluate every model on the untouched test set.
5. Save artifacts to `models/` (`ann_fraud.keras`, `scaler.joblib`,
   `threshold.json`), metrics to `reports/metrics.json`, and figures to
   `reports/figures/`.

Explore the data first with the notebook:

```bash
jupyter notebook notebooks/eda.ipynb
```

---

## Serving the API

The API loads the trained ANN, scaler, and tuned threshold, and scores
transactions either from an inline JSON body or by id from PostgreSQL.

### Quickest path — Docker Compose

Brings up PostgreSQL (schema auto-applied) and the API together:

```bash
docker compose up --build
# API:      http://localhost:8000
# Swagger:  http://localhost:8000/docs
```

Mount your trained `models/` directory (compose already does this) so the
container can load the artifacts.

### Local (without Docker)

```bash
# 1. Start PostgreSQL and apply the schema
psql "$FRAUD_DATABASE_URL" -f db/schema.sql

# 2. (optional) bulk-load transactions to score by id
python -m db.load_csv --csv data/creditcard.csv

# 3. Run the service
uvicorn api.main:app --reload
```

### Endpoints

| Method | Path                    | Description                                   |
|--------|-------------------------|-----------------------------------------------|
| GET    | `/health`               | Model + database status                       |
| POST   | `/score`                | Score an inline transaction payload           |
| GET    | `/score/{id}`           | Fetch a transaction from PostgreSQL and score |
| POST   | `/score/batch`          | Score many transaction ids in one call        |

**Example — inline scoring:**

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"Time": 40000, "Amount": 149.62,
       "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.37, "V5": -0.33,
       "V6": 0.46, "V7": 0.23, "V8": 0.09, "V9": 0.36, "V10": 0.09,
       "V11": -0.55, "V12": -0.61, "V13": -0.99, "V14": -0.31, "V15": 1.46,
       "V16": -0.47, "V17": 0.20, "V18": 0.02, "V19": 0.40, "V20": 0.25,
       "V21": -0.01, "V22": 0.27, "V23": -0.11, "V24": 0.06, "V25": 0.12,
       "V26": -0.18, "V27": 0.13, "V28": -0.02}'
```

```json
{
  "transaction_id": null,
  "fraud_probability": 0.0123,
  "is_fraud": false,
  "threshold": 0.47
}
```

**Example — score a stored transaction:**

```bash
curl http://localhost:8000/score/42
```

---

## Design notes

- **Why SMOTE only on train?** Oversampling before splitting leaks synthetic
  minority samples into validation/test, inflating metrics. `build_dataset`
  splits first, then resamples the training fold only.
- **Why tune the threshold?** At 0.172% prevalence, 0.5 is arbitrary. The
  business cost of a missed fraud far exceeds a false alarm, so the threshold is
  chosen to maximize recall while keeping precision above a floor
  (`config.yaml → threshold.min_precision`). Each baseline gets the same
  treatment for fairness.
- **Why RobustScaler on `Amount`/`Time`?** They contain heavy-tailed outliers;
  a RobustScaler (median/IQR) is less distorted by them than standardization.
- **Why class weights *and* SMOTE?** SMOTE rebalances to ~10% fraud (not 50%,
  which over-synthesizes); residual imbalance is mopped up with class weights.
- **Separation of concerns.** `src/config.py` (training) and `api/config.py`
  (serving) are independent so the API image doesn't drag in training-only
  behavior, and the scaler/threshold are persisted artifacts rather than
  hard-coded.

---

## Testing

```bash
pytest
```

The API tests stub the predictor and database, so they run fast and need
neither a trained model nor a live PostgreSQL instance.

---

## License

MIT — see `LICENSE`. The dataset is distributed by ULB/Kaggle under its own
terms; review those before redistribution.
