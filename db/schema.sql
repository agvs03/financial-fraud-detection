-- PostgreSQL schema for the transactions the API scores.
-- One column per model feature (Time, V1..V28, Amount) plus a surrogate id.
-- `label` is optional and only populated for backtesting / monitoring.

CREATE TABLE IF NOT EXISTS transactions (
    id      BIGSERIAL PRIMARY KEY,
    "Time"  DOUBLE PRECISION NOT NULL,
    "V1"    DOUBLE PRECISION NOT NULL,
    "V2"    DOUBLE PRECISION NOT NULL,
    "V3"    DOUBLE PRECISION NOT NULL,
    "V4"    DOUBLE PRECISION NOT NULL,
    "V5"    DOUBLE PRECISION NOT NULL,
    "V6"    DOUBLE PRECISION NOT NULL,
    "V7"    DOUBLE PRECISION NOT NULL,
    "V8"    DOUBLE PRECISION NOT NULL,
    "V9"    DOUBLE PRECISION NOT NULL,
    "V10"   DOUBLE PRECISION NOT NULL,
    "V11"   DOUBLE PRECISION NOT NULL,
    "V12"   DOUBLE PRECISION NOT NULL,
    "V13"   DOUBLE PRECISION NOT NULL,
    "V14"   DOUBLE PRECISION NOT NULL,
    "V15"   DOUBLE PRECISION NOT NULL,
    "V16"   DOUBLE PRECISION NOT NULL,
    "V17"   DOUBLE PRECISION NOT NULL,
    "V18"   DOUBLE PRECISION NOT NULL,
    "V19"   DOUBLE PRECISION NOT NULL,
    "V20"   DOUBLE PRECISION NOT NULL,
    "V21"   DOUBLE PRECISION NOT NULL,
    "V22"   DOUBLE PRECISION NOT NULL,
    "V23"   DOUBLE PRECISION NOT NULL,
    "V24"   DOUBLE PRECISION NOT NULL,
    "V25"   DOUBLE PRECISION NOT NULL,
    "V26"   DOUBLE PRECISION NOT NULL,
    "V27"   DOUBLE PRECISION NOT NULL,
    "V28"   DOUBLE PRECISION NOT NULL,
    "Amount" DOUBLE PRECISION NOT NULL,
    label   SMALLINT,            -- 0/1 ground truth, nullable
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Scores written back by the service, for auditing and drift monitoring.
CREATE TABLE IF NOT EXISTS fraud_scores (
    id             BIGSERIAL PRIMARY KEY,
    transaction_id BIGINT REFERENCES transactions(id),
    probability    DOUBLE PRECISION NOT NULL,
    is_fraud       BOOLEAN NOT NULL,
    threshold      DOUBLE PRECISION NOT NULL,
    scored_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_fraud_scores_txn
    ON fraud_scores (transaction_id);
