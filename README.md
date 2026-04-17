# FedSentinel

![Project Status: In Progress](https://img.shields.io/badge/Status-In%20Progress-yellow) ![Python](https://img.shields.io/badge/Made%20with-Python-blue)

Predict the Federal Reserve's next interest rate decision (CUT / HOLD / HIKE) from FOMC statement text using BERT embeddings and a supervised classifier.

## Problem Statement

The Federal Reserve's FOMC statements are among the most closely watched texts in finance — small shifts in language signal major policy pivots. FedSentinel embeds each statement with BERT and trains a classifier to predict whether the *next* meeting will cut, hold, or hike rates. A secondary drift-analysis pipeline tracks how Fed communication evolves over time.

## How It Works

1. **Scrape** — `FedScraper.py` pulls every FOMC press release from the Fed's website and saves them as dated `.txt` files.
2. **Embed** — `model.py` encodes each statement into a BERT `[CLS]` vector (768-dim, frozen weights).
3. **Label** — `labeler.py` extracts the rate decision from each statement via regex, then shifts labels by one meeting to create next-meeting prediction targets.
4. **Train** — `train.py` runs a chronological train/val/test split, tunes a logistic regression and SVM classifier, and evaluates on held-out data.
5. **Predict** — `predict.py` loads the saved model and returns a prediction + probability breakdown for any new statement.
6. **Drift** *(secondary)* — `main.py` computes cosine distance between consecutive statements and outputs a time-series drift plot.

## Usage

```bash
# 1. Scrape statements
python src/FedScraper.py

# 2. Train the rate-direction classifier
python src/train.py

# 3. Predict next decision from a statement file
python src/predict.py data/raw_html/20251210_Statement.txt

# 4. (Optional) Run semantic drift analysis
python src/main.py
```

## Outputs (`data/results/`)

| File | Description |
|------|-------------|
| `labeled_statements.csv` | All statements with `own_label` and `next_label` |
| `embeddings_cache.npy` | Cached BERT embeddings — (n, 768) float32 |
| `rate_classifier.joblib` | Trained sklearn Pipeline |
| `confusion_matrix.png` | Test-set confusion matrix |
| `predictions_timeline.png` | True vs predicted labels over time |
| `semantic_drift.csv` | Drift score per meeting date |
| `drift_plot.png` | Time-series drift visualization |

## Next Steps

- [ ] **Expand date range** — backfill pre-2021 statements; more data is the single biggest lever for classifier performance.
- [ ] **FinBERT embeddings** — swap `bert-base-uncased` for a finance-tuned model for higher-quality semantic representations.
- [ ] **Sentence-level features** — embed individual paragraphs separately to capture which section (inflation, labor, forward guidance) drives the prediction.
- [ ] **Rolling baseline drift** — compare each statement to a rolling N-meeting average rather than just t-1.
- [ ] **Interactive dashboard** — Streamlit app exposing predictions and drift scores for non-technical users.
- [ ] **Alerting** — schedule the scraper and flag when a new statement's drift or predicted class changes.
