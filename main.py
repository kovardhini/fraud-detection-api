"""
Fraud detection batch scoring API.

Replicates the feature engineering from the training notebook
(credit_card_fraud_detection_catboost_v3) exactly, so a CSV in the same
shape as the original train/test files gets scored consistently with
what the 5-fold CatBoost models were trained on.

Endpoints:
  GET  /                -> upload-a-CSV frontend page
  GET  /health            -> health check
  GET  /config           -> the model's training config (flag rate, AUC, etc.)
  POST /predict_csv       -> upload a CSV, get back a scored CSV (file download)
  POST /predict_json      -> upload a CSV, get back scored rows as JSON (used by the frontend)

Required input CSV columns (same as the original test.csv):
  transaction_id, user_id, device_id, timestamp, amount,
  hours_since_prev_txn, merchant_category, country, channel
"""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "models"
CONFIG_PATH = APP_DIR / "final_config.json"

REQUIRED_COLUMNS = [
    "transaction_id", "user_id", "device_id", "timestamp", "amount",
    "hours_since_prev_txn", "merchant_category", "country", "channel",
]
CAT_FEATURES = ["merchant_category", "country", "channel"]

app = FastAPI(title="Fraud Detection API", version="1.0")

# ---- loaded once at startup ----
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

FEATURES = CONFIG["features"]          # exact 75 columns, in training order
FLAG_RATE = CONFIG["flag_rate"]        # e.g. 0.035
SCORE_CUTOFF = CONFIG["score_cutoff_on_test"]

MODELS = []
for path in sorted(MODELS_DIR.glob("catboost_fold*.cbm")):
    m = CatBoostClassifier()
    m.load_model(str(path))
    MODELS.append(m)

if not MODELS:
    raise RuntimeError(f"No .cbm model files found in {MODELS_DIR}")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduces CELL 3 of the training notebook, minus the train/test split
    (batch scoring has no label and is scored as a single pool).
    """
    pool = df.copy()
    pool = pool.sort_values(["user_id", "timestamp", "transaction_id"]).reset_index(drop=True)

    def gu(col):
        return pool.groupby("user_id")[col]

    # ---------- time features ----------
    pool["time_hour"] = ((pool["timestamp"] // 3600) % 24).astype(int)
    pool["time_day"] = (pool["timestamp"] // 86400).astype(int)
    pool["time_weekday"] = (pool["time_day"] % 7).astype(int)
    pool["hour_sin"] = np.sin(2 * np.pi * pool["time_hour"] / 24)
    pool["hour_cos"] = np.cos(2 * np.pi * pool["time_hour"] / 24)

    # ---------- basic transforms ----------
    pool["log_amount"] = np.log1p(pool["amount"].clip(lower=0))
    pool["log_gap"] = np.log1p(pool["hours_since_prev_txn"].clip(lower=0))

    # ---------- user history / sequence features ----------
    pool["u_n"] = gu("amount").transform("size")
    pool["u_rank"] = gu("amount").cumcount()
    pool["u_rank_pct"] = pool["u_rank"] / (pool["u_n"] - 1).clip(lower=1)

    pool["u_prev_ts_gap"] = gu("timestamp").diff()
    pool["u_next_ts_gap"] = -gu("timestamp").diff(-1)
    pool["u_prev_amount"] = gu("amount").shift(1)
    pool["u_next_amount"] = gu("amount").shift(-1)
    pool["amt_diff_prev"] = pool["amount"] - pool["u_prev_amount"]
    pool["amt_ratio_prev"] = pool["amount"] / pool["u_prev_amount"].replace(0, np.nan)
    pool["u_roll3_mean"] = gu("amount").transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    pool["amt_ratio_roll3"] = pool["amount"] / pool["u_roll3_mean"].replace(0, np.nan)

    pool["gap_mismatch"] = pool["hours_since_prev_txn"] - pool["u_prev_ts_gap"]
    pool["gap_mismatch_abs"] = pool["gap_mismatch"].abs()
    pool["gap_matches"] = (pool["gap_mismatch_abs"] < 1).astype(int)

    # ---------- user amount statistics ----------
    pool["u_amount_mean"] = gu("amount").transform("mean")
    pool["u_amount_std"] = gu("amount").transform("std")
    pool["u_amount_median"] = gu("amount").transform("median")
    pool["u_amount_max"] = gu("amount").transform("max")
    pool["u_amount_ratio"] = pool["amount"] / pool["u_amount_mean"].replace(0, np.nan)
    pool["u_amount_z"] = (pool["amount"] - pool["u_amount_mean"]) / pool["u_amount_std"].replace(0, np.nan)
    pool["u_amount_pct"] = gu("amount").rank(pct=True)
    pool["amount_vs_user_max"] = pool["amount"] / pool["u_amount_max"].replace(0, np.nan)

    # ---------- user gap statistics ----------
    pool["u_gap_median"] = gu("hours_since_prev_txn").transform("median")
    pool["u_gap_std"] = gu("hours_since_prev_txn").transform("std")
    pool["gap_ratio_user"] = pool["hours_since_prev_txn"] / pool["u_gap_median"].replace(0, np.nan)
    pool["gap_z_user"] = (pool["hours_since_prev_txn"] - pool["u_gap_median"]) / pool["u_gap_std"].replace(0, np.nan)

    # ---------- user behaviour on categorical fields ----------
    for c in ["country", "device_id", "merchant_category", "channel"]:
        prev = gu(c).shift(1)
        pool[f"{c}_changed"] = (prev.notna() & (prev != pool[c])).astype(int)
        pair_n = pool.groupby(["user_id", c])[c].transform("size")
        pool[f"u_{c}_pair_n"] = pair_n
        pool[f"u_{c}_pair_ratio"] = pair_n / pool["u_n"]
        pool[f"u_{c}_nunique"] = gu(c).transform("nunique")
        pool[f"u_{c}_is_rare"] = (pool[f"u_{c}_pair_ratio"] <= 0.05).astype(int)

    hour_pair_n = pool.groupby(["user_id", "time_hour"])["time_hour"].transform("size")
    pool["u_hour_ratio"] = hour_pair_n / pool["u_n"]

    # ---------- device features ----------
    pool["d_n"] = pool.groupby("device_id")["amount"].transform("size")
    pool["d_user_nunique"] = pool.groupby("device_id")["user_id"].transform("nunique")
    pool["d_amount_mean"] = pool.groupby("device_id")["amount"].transform("mean")
    pool["d_amount_std"] = pool.groupby("device_id")["amount"].transform("std")
    pool["d_amount_ratio"] = pool["amount"] / pool["d_amount_mean"].replace(0, np.nan)
    pool["d_amount_z"] = (pool["amount"] - pool["d_amount_mean"]) / pool["d_amount_std"].replace(0, np.nan)

    dev = pool[["transaction_id", "device_id", "user_id", "timestamp"]].sort_values(
        ["device_id", "timestamp", "transaction_id"]
    )
    dev["d_prev_gap"] = dev.groupby("device_id")["timestamp"].diff()
    dev["d_next_gap"] = -dev.groupby("device_id")["timestamp"].diff(-1)
    prev_user = dev.groupby("device_id")["user_id"].shift(1)
    dev["d_user_switch"] = (prev_user.notna() & (prev_user != dev["user_id"])).astype(int)
    pool = pool.merge(
        dev[["transaction_id", "d_prev_gap", "d_next_gap", "d_user_switch"]],
        on="transaction_id", how="left"
    )

    # ---------- merchant / country / channel level features ----------
    for c in ["merchant_category", "country", "channel"]:
        pool[f"{c}_freq"] = pool.groupby(c)[c].transform("size") / len(pool)
        m = pool.groupby(c)["amount"].transform("mean")
        s = pool.groupby(c)["amount"].transform("std")
        pool[f"amount_z_{c}"] = (pool["amount"] - m) / s.replace(0, np.nan)

    # ---------- finalise ----------
    pool = pool.replace([np.inf, -np.inf], np.nan)
    for c in CAT_FEATURES:
        pool[c] = pool[c].astype(str)

    return pool


def score(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

    pool = build_features(df)

    missing_features = [c for c in FEATURES if c not in pool.columns]
    if missing_features:
        raise HTTPException(
            status_code=500,
            detail=f"Feature engineering did not produce expected columns: {missing_features}",
        )

    X = pool[FEATURES]

    probs = np.zeros(len(X))
    for m in MODELS:
        probs += m.predict_proba(X)[:, 1] / len(MODELS)
    pool["fraud_probability"] = probs

    # Fixed-threshold flag (stable across any batch size, uses the cutoff
    # learned at training time)
    pool["flag_fixed_cutoff"] = (pool["fraud_probability"] >= SCORE_CUTOFF).astype(int)

    # Rank-based flag within this batch (matches the notebook's top-k% approach;
    # most meaningful when the batch is reasonably large, e.g. hundreds+ rows)
    n_flag = max(1, int(round(len(pool) * FLAG_RATE)))
    order = np.argsort(-pool["fraud_probability"].values, kind="stable")
    rank_flag = np.zeros(len(pool), dtype=int)
    rank_flag[order[:n_flag]] = 1
    pool["flag_topk_batch"] = rank_flag

    return pool[["transaction_id", "fraud_probability", "flag_fixed_cutoff", "flag_topk_batch"]]


FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fraud Detection</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 640px;
         margin: 40px auto; padding: 0 20px; }
  h1 { font-size: 1.5rem; margin-bottom: 4px; }
  .sub { color: #888; font-size: 0.9rem; margin-bottom: 24px; }
  .card { border: 1px solid rgba(128,128,128,0.3); border-radius: 12px; padding: 22px;
          margin-bottom: 20px; }
  .field { margin: 12px 0; }
  .field label { display: block; font-size: 0.85rem; margin-bottom: 4px; color: #888; }
  .field input, .field select { width: 100%; padding: 8px 10px; border: 1px solid #ccc;
    border-radius: 6px; font-size: 0.95rem; background: transparent; color: inherit; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0 16px; }
  button { background: #2563eb; color: white; border: none; padding: 11px 20px;
           border-radius: 8px; cursor: pointer; font-size: 0.95rem; font-weight: 600;
           width: 100%; margin-top: 8px; }
  button:hover { background: #1e50c0; }
  #err { color: #c0392b; margin-top: 10px; font-size: 0.9rem; }

  #manualResult { display: none; }
  .verdict { padding: 22px; border-radius: 12px; text-align: center; margin-bottom: 18px; }
  .verdict.fraud { background: rgba(220,50,50,0.12); border: 1px solid #dc3232; }
  .verdict.ok { background: rgba(30,140,80,0.12); border: 1px solid #1e8c50; }
  .verdict h2 { margin: 0 0 4px 0; font-size: 1.3rem; }
  .verdict.fraud h2 { color: #dc3232; }
  .verdict.ok h2 { color: #1e8c50; }
  .verdict .icon { font-size: 2.2rem; margin-bottom: 6px; }

  .meter-row { display: flex; justify-content: space-between; font-size: 0.8rem;
    color: #888; margin-bottom: 4px; }
  .meter { position: relative; height: 10px; border-radius: 6px; background: rgba(128,128,128,0.2);
    overflow: visible; margin: 6px 0 2px 0; }
  .meter-fill { height: 100%; border-radius: 6px; }
  .meter-fill.fraud { background: #dc3232; }
  .meter-fill.ok { background: #1e8c50; }
  .meter-threshold { position: absolute; top: -3px; width: 2px; height: 16px; background: #888; }
  .meter-caption { font-size: 0.75rem; color: #888; margin-top: 4px; }

  .details-title { font-size: 0.85rem; color: #888; margin: 4px 0 10px 0; text-transform: uppercase;
    letter-spacing: 0.03em; }
  .details table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  .details td { padding: 5px 0; border-bottom: 1px solid rgba(128,128,128,0.15); }
  .details td:first-child { color: #888; }
  .details td:last-child { text-align: right; font-weight: 500; }

  .again { background: transparent; border: 1px solid #2563eb; color: #2563eb; margin-top: 14px; }
  .again:hover { background: rgba(37,99,235,0.08); }
</style>
</head>
<body>
  <h1>Fraud Detection</h1>
  <div class="sub">Enter a transaction to check whether it looks like fraud.</div>

  <div class="card" id="formCard">
    <div class="grid2">
      <div class="field"><label>User ID</label><input type="number" id="m_user_id" value="1"></div>
      <div class="field"><label>Device ID</label><input type="number" id="m_device_id" value="1"></div>
      <div class="field"><label>Amount ($)</label><input type="number" step="0.01" id="m_amount" value="100"></div>
      <div class="field"><label>Hours since previous transaction</label><input type="number" step="0.1" id="m_gap" value="5"></div>
      <div class="field"><label>Timestamp (unix seconds)</label><input type="number" id="m_timestamp" value=""></div>
      <div class="field"><label>Channel</label>
        <select id="m_channel"><option>online</option><option>pos</option><option>atm</option></select>
      </div>
      <div class="field"><label>Merchant category</label>
        <input type="text" id="m_merchant" value="grocery">
      </div>
      <div class="field"><label>Country</label>
        <input type="text" id="m_country" value="US">
      </div>
    </div>

    <div class="field">
      <label>Flagging threshold: <span id="thresholdLabel"></span>
        <span style="font-weight:400;">(lower = stricter, more gets flagged as fraud)</span></label>
      <input type="range" id="m_threshold" min="0.005" max="0.5" step="0.005" value="0.05">
    </div>

    <button id="manualSubmit">Check transaction</button>
    <div id="err"></div>
  </div>

  <div id="manualResult">
    <div class="card" id="verdictCard"></div>
    <div class="card details">
      <div class="details-title">Transaction details</div>
      <table id="detailsTable"></table>
      <button class="again" id="checkAnother">Check another transaction</button>

    </div>
  </div>

<script>
document.getElementById('m_timestamp').value = Math.floor(Date.now() / 1000);

const thresholdInput = document.getElementById('m_threshold');
const thresholdLabel = document.getElementById('thresholdLabel');
function updateThresholdLabel() {
  thresholdLabel.textContent = (parseFloat(thresholdInput.value) * 100).toFixed(1) + '%';
}
thresholdInput.addEventListener('input', updateThresholdLabel);
updateThresholdLabel();

function fieldLabel(id) {
  return {
    user_id: 'User ID', device_id: 'Device ID', amount: 'Amount',
    hours_since_prev_txn: 'Hours since previous transaction', timestamp: 'Timestamp',
    channel: 'Channel', merchant_category: 'Merchant category', country: 'Country',
  }[id];
}

document.getElementById('manualSubmit').addEventListener('click', async () => {
  const errEl = document.getElementById('err');
  errEl.textContent = '';

  const payload = {
    user_id: parseInt(document.getElementById('m_user_id').value),
    device_id: parseInt(document.getElementById('m_device_id').value),
    timestamp: parseInt(document.getElementById('m_timestamp').value),
    amount: parseFloat(document.getElementById('m_amount').value),
    hours_since_prev_txn: parseFloat(document.getElementById('m_gap').value),
    merchant_category: document.getElementById('m_merchant').value,
    country: document.getElementById('m_country').value,
    channel: document.getElementById('m_channel').value,
  };
  const userThreshold = parseFloat(thresholdInput.value);

  try {
    const res = await fetch('/predict_manual', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail ? JSON.stringify(detail.detail) : ('Request failed: ' + res.status));
    }
    const data = await res.json();
    render(data, userThreshold);
  } catch (e) {
    errEl.textContent = e.message;
  }
});

document.getElementById('checkAnother').addEventListener('click', () => {
  document.getElementById('manualResult').style.display = 'none';
  document.getElementById('formCard').style.display = 'block';
});

function render(data, threshold) {
  const prob = data.fraud_probability;
  const isFraud = prob >= threshold;

  // scale the meter so the threshold line sits at a fixed visible point,
  // and the fill goes up to 100% once probability reaches ~3x threshold
  const maxScale = threshold * 3;
  const fillPct = Math.min(100, (prob / maxScale) * 100);
  const thresholdPct = Math.min(100, (threshold / maxScale) * 100);

  const verdict = document.getElementById('verdictCard');
  verdict.innerHTML = `
    <div class="verdict ${isFraud ? 'fraud' : 'ok'}">
      <div class="icon">${isFraud ? '\u26A0\uFE0F' : '\u2705'}</div>
      <h2>${isFraud ? 'Flagged as fraud' : 'Looks legitimate'}</h2>
      <div>Fraud score: <b>${(prob * 100).toFixed(2)}%</b></div>
    </div>
    <div class="meter-row"><span>Risk level</span><span>Your threshold: ${(threshold * 100).toFixed(2)}%</span></div>
    <div class="meter">
      <div class="meter-fill ${isFraud ? 'fraud' : 'ok'}" style="width:${fillPct}%"></div>
      <div class="meter-threshold" style="left:${thresholdPct}%"></div>
    </div>
    <div class="meter-caption">${isFraud
      ? 'This transaction scored above your chosen threshold.'
      : 'This transaction scored below your chosen threshold.'}</div>
  `;

  const rows = Object.entries(data.input).map(([k, v]) =>
    `<tr><td>${fieldLabel(k)}</td><td>${v}</td></tr>`
  ).join('');
  document.getElementById('detailsTable').innerHTML = rows;

  document.getElementById('formCard').style.display = 'none';
  document.getElementById('manualResult').style.display = 'block';
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def frontend():
    return FRONTEND_HTML


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(MODELS), "n_features": len(FEATURES)}


@app.get("/config")
def config():
    return {
        "flag_rate": FLAG_RATE,
        "score_cutoff_on_test": SCORE_CUTOFF,
        "oof_roc_auc": CONFIG.get("oof_roc_auc"),
        "oof_roc_auc_ci": CONFIG.get("oof_roc_auc_ci"),
        "note": "oof_roc_auc ~0.52-0.53: signal is real but weak. This is an honest, "
                "not overstated, fraud model.",
    }


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="CSV has no rows")

    result = score(df)

    buf = io.StringIO()
    result.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=fraud_predictions.csv"},
    )


@app.post("/predict_json")
async def predict_json(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="CSV has no rows")

    result = score(df)
    return {
        "rows": result.to_dict(orient="records"),
        "oof_roc_auc": CONFIG.get("oof_roc_auc"),
    }


class ManualTransaction(BaseModel):
    user_id: int
    device_id: int
    timestamp: int
    amount: float
    hours_since_prev_txn: float
    merchant_category: str
    country: str
    channel: str


@app.post("/predict_manual")
def predict_manual(txn: ManualTransaction):
    df = pd.DataFrame([{
        "transaction_id": 1,
        "user_id": txn.user_id,
        "device_id": txn.device_id,
        "timestamp": txn.timestamp,
        "amount": txn.amount,
        "hours_since_prev_txn": txn.hours_since_prev_txn,
        "merchant_category": txn.merchant_category,
        "country": txn.country,
        "channel": txn.channel,
    }])
    result = score(df)
    row = result.iloc[0]
    return {
        "fraud_probability": float(row["fraud_probability"]),
        "flag_fixed_cutoff": int(row["flag_fixed_cutoff"]),
        "threshold": SCORE_CUTOFF,
        "input": txn.dict(),
    }
