# FedSentinel

![Project Status: Completed](https://img.shields.io/badge/Status-Completed-success) ![Python](https://img.shields.io/badge/Made%20with-Python-blue)

A tool for tracking how Federal Reserve communication changes over time using NLP-based semantic drift analysis.

## Problem Statement

The Federal Reserve's FOMC statements are among the most closely watched texts in finance — small shifts in language signal major policy pivots. FedSentinel quantifies those shifts by embedding each statement with BERT and measuring cosine distance between consecutive releases. The result is a time-series "drift score" that makes subtle language changes visible and comparable across decades of Fed communication.

## How It Works

1. **Scrape** — `FedScraper.py` pulls every FOMC press release from the Fed's website and saves them as dated `.txt` files.
2. **Embed** — `model.py` encodes each statement into a BERT [CLS] vector.
3. **Drift** — `main.py` computes cosine distance between consecutive statements and outputs a CSV + plot.

## Usage

```bash
# 1. Scrape statements
python src/FedScraper.py

# 2. Run drift analysis
python src/main.py
```

Results are saved to `data/results/`:
- `semantic_drift.csv` — drift score per meeting date
- `drift_plot.png` — time-series visualization

## Next Steps

- [ ] **Expand date range** — scraper currently covers 2000–2026; backfill pre-2000 statements from the Fed archive for a longer historical view.
- [ ] **Sentence-level drift** — instead of embedding the full statement, split by paragraph/sentence to pinpoint *which* sections shifted (e.g., inflation language vs. labor market language).
- [ ] **Keyword / topic tagging** — label high-drift periods with the macro event driving the change (rate hike cycle, QE, COVID response) to make the chart self-explanatory.
- [ ] **Rolling baseline** — compare each statement to a rolling N-meeting average rather than just t-1, to distinguish genuine pivots from one-meeting noise.
- [ ] **Upgrade embedder** — swap BERT for a finance-tuned model (e.g., FinBERT) or a more recent sentence-transformer for higher-quality embeddings.
- [ ] **Interactive dashboard** — expose the drift time-series via a simple Streamlit app so non-technical users can explore the data.
- [ ] **Alerting** — run the scraper on a schedule and flag when a new statement's drift score exceeds a threshold (potential policy signal).
