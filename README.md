---
title: FedSentinel
emoji: 📄
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.0.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# FedSentinel

![Project Status: Complete](https://img.shields.io/badge/Status-Complete-brightgreen) ![Python](https://img.shields.io/badge/Made%20with-Python-blue)

Predict the Federal Reserve's next interest rate decision (CUT / HOLD / HIKE) from FOMC statement text using FinBERT embeddings, engineered features, and a supervised classifier with an optional Transformer sequence model.

## Problem Statement

The Federal Reserve's FOMC statements are among the most closely watched texts in finance — small shifts in language signal major policy pivots. FedSentinel embeds each statement with FinBERT, enriches it with structured features (Fed Funds Rate, sentiment drift, keyword counts), and trains a classifier to predict whether the *next* meeting will cut, hold, or hike rates. A secondary drift-analysis pipeline tracks how Fed communication evolves over time.

## How It Works

1. **Scrape** — `FedScraper.py` pulls every FOMC press release from 2003 to today across three different historical URL formats used by the Fed website.
2. **Embed** — `model.py` encodes each statement with `ProsusAI/finbert` using mean pooling over all token embeddings. The rate-decision sentence is masked before embedding so the model captures policy *reasoning*, not just the current decision.
3. **Label** — `train.py` extracts the rate decision from each statement via regex and shifts labels by one meeting to create next-meeting prediction targets.
4. **Feature engineering** — For each meeting, a 1541-dim feature vector is built:
   - Current FinBERT embedding (768-dim)
   - Previous meeting's embedding (768-dim, zero-padded for first sample)
   - Sentiment delta: cosine distance to the previous embedding (measures how much language shifted)
   - Fed Funds Rate going into the meeting (pre-decision, to avoid leakage)
   - Keyword counts: hawkish terms, dovish terms, net score
5. **Train** — `train.py` runs a chronological train/val/test split, tunes a Logistic Regression and SVM, and (if ≥150 labeled samples) also trains a 2-layer Transformer sequence model over the last 8 meetings. The best model by validation macro-F1 is used for test evaluation.
6. **Predict** — `predict.py` loads the saved model and returns a prediction + probability breakdown for any new statement.
7. **Drift** *(secondary)* — `main.py` computes cosine distance between consecutive statements and outputs a time-series drift plot.

## Architecture

```
Raw FOMC Statement Text
        │
        ▼
  [Decision phrase masked]
        │
        ▼
  ProsusAI/FinBERT
  (mean pooling over all tokens)
        │
        ▼
   768-dim embedding
        │
   + prev embedding (768)
   + sentiment delta  (1)    ← cosine distance to previous meeting
   + Fed Funds Rate   (1)    ← rate in effect before this meeting
   + keyword counts   (3)    ← hawkish / dovish / net
        │
        ▼
   1541-dim feature vector
        │
   ┌────┴────┐
   ▼         ▼
  SVM    Transformer     (Transformer trains if n ≥ 150)
   │    (seq of last 8   
   │     meetings)       
   └────┬────┘
        ▼
  Best by val macro-F1
        │
        ▼
  CUT / HOLD / HIKE
```

## Usage

```bash
# 1. Install dependencies (requires Python 3.10+)
pip install -r requirements.txt   # or: pip install torch transformers scikit-learn pandas numpy matplotlib joblib beautifulsoup4 requests

# 2. Scrape all FOMC statements (2003–present)
python src/FedScraper.py

# 3. Train
python src/train.py

# 4. Predict next decision from a statement file
python src/predict.py data/raw_html/20251210_Statement.txt

# 5. Predict with previous statement + meeting date for better accuracy
python src/predict.py data/raw_html/20251210_Statement.txt data/raw_html/20251029_Statement.txt 2025-12-10

# 6. (Optional) Run semantic drift analysis
python src/main.py
```

## Outputs (`data/results/`)

| File | Description |
|------|-------------|
| `labeled_statements.csv` | All statements with `own_label` and `next_label` |
| `embeddings_cache.npy` | Cached FinBERT embeddings — (n, 768) float32 |
| `rate_classifier.joblib` | Best sklearn pipeline (always saved; used by predict.py) |
| `transformer_model.pt` | Transformer weights (saved when n ≥ 150 and Transformer wins) |
| `confusion_matrix.png` | Test-set confusion matrix |
| `predictions_timeline.png` | True vs predicted labels over time |

## Current Results

Trained on 170 labeled samples (2003–2026), tested on the 2022–2026 period:

| Model | Val macro-F1 |
|-------|-------------|
| SVM (C=10, RBF kernel) | 0.447 |
| Logistic Regression (C=10) | 0.379 |
| Transformer (2-layer, seq=8) | 0.447 |

Test macro-F1: **0.41** — HOLD predictions are reasonable (F1=0.56), but the model struggles with HIKEs on the test set, as the 2022–2023 hiking cycle is stylistically different from the ZLB-era training data.

## Key Design Decisions

**Why mask the decision phrase?** The embedding shouldn't trivially encode "they just raised rates" — we want it to capture the surrounding policy language (inflation outlook, labor market assessment, forward guidance). The current decision is fed back in as a separate feature (via the previous embedding).

**Why mean pooling over [CLS]?** The `[CLS]` token compresses the whole sequence into one vector during pretraining. Mean pooling averages all token representations weighted by the attention mask, which gives a more stable sentence-level embedding.

**Why SVM over a neural head?** With ~170 training samples and a 1541-dim feature space, SVMs generalize better than neural classifiers — they only depend on the support vectors (boundary-adjacent points), not the full dataset.

**Why no leakage on FFR?** The rate feature uses the FFR set at the *previous* meeting, not the current one. At prediction time you always know last meeting's decision; you don't know this meeting's yet.

## Next Steps

- [ ] **Fix old label patterns** — 23 statements from 2003–2006 use `"raise its target for the federal funds rate"` phrasing not yet matched by the regex; adding them would push past 200 labeled samples.
- [ ] **Sentence-level features** — embed individual paragraphs separately to capture which section (inflation, labor, forward guidance) drives the prediction.
- [ ] **Rolling baseline drift** — compare each statement to a rolling N-meeting average rather than just t-1.
- [ ] **Interactive dashboard** — Streamlit app exposing predictions and drift scores for non-technical users.
- [ ] **Alerting** — schedule the scraper and flag when a new statement's drift or predicted class changes.
