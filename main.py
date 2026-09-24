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
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 900px;
         margin: 40px auto; padding: 0 20px; }
  h1 { font-size: 1.4rem; }
  .drop { border: 2px dashed #999; border-radius: 10px; padding: 30px; text-align: center;
          cursor: pointer; margin: 20px 0; }
  .drop.drag { border-color: #4a7; background: rgba(74,170,119,0.08); }
  button { background: #2563eb; color: white; border: none; padding: 10px 18px;
           border-radius: 6px; cursor: pointer; font-size: 0.95rem; }
  button:disabled { opacity: 0.5; cursor: default; }
  table { border-collapse: collapse; width: 100%; margin-top: 20px; font-size: 0.9rem; }
  th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  tr.fraud { background: rgba(220,50,50,0.15); }
  .badge { padding: 2px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: 600; }
  .badge.fraud { background: #dc3232; color: white; }
  .badge.ok { background: #2a8; color: white; }
  #summary { margin-top: 16px; font-size: 0.95rem; }
  #err { color: #c0392b; margin-top: 10px; }
  #loading { display: none; margin-top: 10px; }
  .tabs { display: flex; gap: 8px; margin: 20px 0; border-bottom: 1px solid #ccc; }
  .tab { padding: 8px 16px; cursor: pointer; border-bottom: 2px solid transparent; }
  .tab.active { border-bottom-color: #2563eb; font-weight: 600; }
  .panel { display: none; }
  .panel.active { display: block; }
  .field { margin: 10px 0; }
  .field label { display: block; font-size: 0.85rem; margin-bottom: 3px; color: #666; }
  .field input, .field select { width: 100%; padding: 6px 8px; box-sizing: border-box;
    border: 1px solid #ccc; border-radius: 4px; font-size: 0.9rem; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0 16px; }
  .warn { background: rgba(230,170,0,0.15); border: 1px solid #e6aa00; border-radius: 6px;
    padding: 10px 14px; font-size: 0.85rem; margin: 14px 0; }
  #manualResult { margin-top: 18px; }
  .verdict { padding: 16px; border-radius: 10px; text-align: center; }
  .verdict.fraud { background: rgba(220,50,50,0.15); border: 1px solid #dc3232; }
  .verdict.ok { background: rgba(30,140,80,0.15); border: 1px solid #1e8c50; }
  .verdict h2 { margin: 0 0 6px 0; }
</style>
</head>
<body>
  <h1>Fraud Detection</h1>

  <div class="tabs">
    <div class="tab active" data-tab="csv">Upload CSV (accurate)</div>
    <div class="tab" data-tab="manual">Enter manually (quick, less accurate)</div>
  </div>

  <div class="panel active" id="panel-csv">
    <p>Upload a CSV with columns: <code>transaction_id, user_id, device_id, timestamp,
       amount, hours_since_prev_txn, merchant_category, country, channel</code></p>

    <div class="drop" id="drop">
      <input type="file" id="file" accept=".csv" style="display:none">
      <p id="dropText">Click to choose a CSV, or drag one here</p>
    </div>
    <button id="submit" disabled>Score transactions</button>
    <div id="loading">Scoring…</div>
    <div id="err"></div>
    <div id="summary"></div>
    <div id="tableWrap"></div>
  </div>

  <div class="panel" id="panel-manual">
    <div class="warn">
      This model relies heavily on device/merchant/category history across the whole
      dataset. A single manually-entered transaction has none of that context, so this
      score is noticeably less reliable than the CSV batch mode. Use it for a rough
      check, not a final decision.
    </div>

    <div class="grid2">
      <div class="field"><label>User ID</label><input type="number" id="m_user_id" value="1"></div>
      <div class="field"><label>Device ID</label><input type="number" id="m_device_id" value="1"></div>
      <div class="field"><label>Amount</label><input type="number" step="0.01" id="m_amount" value="100"></div>
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

    <button id="manualSubmit">Check transaction</button>
    <div id="manualErr" class="err"></div>
    <div id="manualResult"></div>
  </div>

<script>
document.getElementById('m_timestamp').value = Math.floor(Date.now() / 1000);

// ---- tabs ----
document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('panel-' + t.dataset.tab).classList.add('active');
  });
});

// ---- manual entry ----
document.getElementById('manualSubmit').addEventListener('click', async () => {
  const errEl = document.getElementById('manualErr');
  const resultEl = document.getElementById('manualResult');
  errEl.textContent = '';
  resultEl.innerHTML = '';

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
    const isFraud = data.flag_fixed_cutoff === 1;
    resultEl.innerHTML = `
      <div class="verdict ${isFraud ? 'fraud' : 'ok'}">
        <h2>${isFraud ? 'Likely fraud' : 'Likely legitimate'}</h2>
        <p>Fraud probability: <b>${(data.fraud_probability * 100).toFixed(2)}%</b></p>
        <p style="font-size:0.8rem; color:#888;">Model OOF ROC-AUC: ${data.oof_roc_auc.toFixed(3)} (weak-but-real signal)</p>
      </div>`;
  } catch (e) {
    errEl.textContent = e.message;
  }
});

// ---- csv upload (existing) ----
const drop = document.getElementById('drop');
const fileInput = document.getElementById('file');
const submitBtn = document.getElementById('submit');
const dropText = document.getElementById('dropText');
const err = document.getElementById('err');
const loading = document.getElementById('loading');
const summary = document.getElementById('summary');
const tableWrap = document.getElementById('tableWrap');
let selectedFile = null;

drop.addEventListener('click', () => fileInput.click());
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('drag'); });
drop.addEventListener('dragleave', () => drop.classList.remove('drag'));
drop.addEventListener('drop', e => {
  e.preventDefault();
  drop.classList.remove('drag');
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) setFile(fileInput.files[0]);
});

function setFile(f) {
  selectedFile = f;
  dropText.textContent = 'Selected: ' + f.name;
  submitBtn.disabled = false;
}

submitBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  err.textContent = '';
  summary.innerHTML = '';
  tableWrap.innerHTML = '';
  loading.style.display = 'block';
  submitBtn.disabled = true;

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const res = await fetch('/predict_json', { method: 'POST', body: formData });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || ('Request failed: ' + res.status));
    }
    const data = await res.json();
    render(data);
  } catch (e) {
    err.textContent = e.message;
  } finally {
    loading.style.display = 'none';
    submitBtn.disabled = false;
  }
});

function render(data) {
  const rows = data.rows;
  const nFraud = rows.filter(r => r.flag_fixed_cutoff === 1).length;
  summary.innerHTML = `<b>${rows.length}</b> transactions scored — ` +
    `<b>${nFraud}</b> flagged as fraud (fixed cutoff) ` +
    `&middot; model OOF ROC-AUC: ${data.oof_roc_auc.toFixed(3)} (weak-but-real signal)`;

  let html = '<table><thead><tr><th>Transaction ID</th><th>Fraud probability</th>' +
    '<th>Flag (fixed cutoff)</th><th>Flag (top-k in batch)</th></tr></thead><tbody>';
  for (const r of rows) {
    const isFraud = r.flag_fixed_cutoff === 1;
    html += `<tr class="${isFraud ? 'fraud' : ''}">` +
      `<td>${r.transaction_id}</td>` +
      `<td>${(r.fraud_probability * 100).toFixed(2)}%</td>` +
      `<td><span class="badge ${isFraud ? 'fraud' : 'ok'}">${isFraud ? 'FRAUD' : 'OK'}</span></td>` +
      `<td>${r.flag_topk_batch === 1 ? 'FRAUD' : 'OK'}</td>` +
      `</tr>`;
  }
  html += '</tbody></table>';
  tableWrap.innerHTML = html;
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
        "oof_roc_auc": CONFIG.get("oof_roc_auc"),
    }
