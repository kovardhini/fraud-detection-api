# Fraud Detection API (batch scoring)

FastAPI app that wraps the 5-fold CatBoost fraud model from the training
notebook. Upload a CSV in the same shape as the original `test.csv` and get
back fraud scores.

## Files

```
fraud-detection-api/
├── main.py              # the API
├── requirements.txt
├── final_config.json    # feature list, flag rate, training metrics
└── models/
    ├── catboost_fold1.cbm
    ├── catboost_fold2.cbm
    ├── catboost_fold3.cbm
    ├── catboost_fold4.cbm
    └── catboost_fold5.cbm
```

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI, or:

```bash
curl -X POST "http://127.0.0.1:8000/predict_csv" \
  -F "file=@test.csv" \
  -o fraud_predictions.csv
```

## Required input CSV columns

Same 9 columns as the original `test.csv`:

```
transaction_id, user_id, device_id, timestamp, amount,
hours_since_prev_txn, merchant_category, country, channel
```

## Output columns

| column | meaning |
|---|---|
| `transaction_id` | passthrough id |
| `fraud_probability` | averaged probability across the 5 fold models |
| `flag_fixed_cutoff` | 1/0, using the fixed probability cutoff learned at training time (`score_cutoff_on_test` in `final_config.json`) |
| `flag_topk_batch` | 1/0, flags the top `flag_rate` fraction of **this batch** — matches the notebook's rank-based approach, most meaningful on larger batches (hundreds of rows+) |

## Deploying on Render (free tier)

1. Push this whole `fraud-detection-api/` folder to a new GitHub repo.
2. On [render.com](https://render.com): **New → Web Service** → connect the repo.
3. Settings:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance type**: Free
4. Deploy. Render gives you a URL like `https://fraud-detection-api.onrender.com`.
5. Test it:
   ```bash
   curl -X POST "https://fraud-detection-api.onrender.com/predict_csv" \
     -F "file=@test.csv" -o fraud_predictions.csv
   ```

**Free tier note:** the service sleeps after 15 minutes idle and takes
~30-50 seconds to wake up on the next request. Fine for demos/testing.

## Honesty check

Hit `GET /config` to see the model's actual out-of-fold performance
(`oof_roc_auc` ≈ 0.52-0.53 — real but weak signal, consistent with what
the training notebook reported). Don't expect this model to catch most
fraud; it's a modest lift over random ranking, not a high-confidence
detector.
