import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import FedModel

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
DATA_DIR     = os.path.join(project_root, "data", "raw_html")
RESULTS_DIR  = os.path.join(project_root, "data", "results")

LABEL_ORDER       = ["CUT", "HOLD", "HIKE"]
SEQ_LEN           = 8    # lookback window for the Transformer
TRANSFORMER_MIN_N = 150  # use Transformer only above this many labeled samples

# ---------------------------------------------------------------------------
# Fed Funds Rate lookup
# Each entry is (date_rate_took_effect, lower_bound_pct).
# To avoid leakage we use the rate in effect BEFORE each meeting (i.e. the
# rate set at the previous meeting).
# ---------------------------------------------------------------------------
_FFR_CHANGES = sorted([
    ("1998-09-29", 5.25), ("1998-10-15", 5.00), ("1998-11-17", 4.75),
    ("1999-06-30", 5.00), ("1999-08-24", 5.25), ("1999-11-16", 5.50),
    ("2000-02-02", 5.75), ("2000-03-21", 6.00), ("2000-05-16", 6.50),
    ("2001-01-03", 6.00), ("2001-01-31", 5.50), ("2001-03-20", 5.00),
    ("2001-04-18", 4.50), ("2001-05-15", 4.00), ("2001-06-27", 3.75),
    ("2001-08-21", 3.50), ("2001-09-17", 3.00), ("2001-10-02", 2.50),
    ("2001-11-06", 2.00), ("2001-12-11", 1.75), ("2002-11-06", 1.25),
    ("2003-06-25", 1.00), ("2004-06-30", 1.25), ("2004-08-10", 1.50),
    ("2004-09-21", 1.75), ("2004-11-10", 2.00), ("2004-12-14", 2.25),
    ("2005-02-02", 2.50), ("2005-03-22", 2.75), ("2005-05-03", 3.00),
    ("2005-06-30", 3.25), ("2005-08-09", 3.50), ("2005-09-20", 3.75),
    ("2005-11-01", 4.00), ("2005-12-13", 4.25), ("2006-01-31", 4.50),
    ("2006-03-28", 4.75), ("2006-05-10", 5.00), ("2006-06-29", 5.25),
    ("2007-09-18", 4.75), ("2007-10-31", 4.50), ("2007-12-11", 4.25),
    ("2008-01-22", 3.50), ("2008-01-30", 3.00), ("2008-03-18", 2.25),
    ("2008-04-30", 2.00), ("2008-10-08", 1.50), ("2008-10-29", 1.00),
    ("2008-12-16", 0.00),
    ("2015-12-16", 0.25), ("2016-12-14", 0.50),
    ("2017-03-15", 0.75), ("2017-06-14", 1.00), ("2017-12-13", 1.25),
    ("2018-03-21", 1.50), ("2018-06-13", 1.75), ("2018-09-26", 2.00),
    ("2018-12-19", 2.25),
    ("2019-07-31", 2.00), ("2019-09-18", 1.75), ("2019-10-30", 1.50),
    ("2020-03-03", 1.00), ("2020-03-15", 0.00),
    ("2022-03-16", 0.25), ("2022-05-04", 0.75), ("2022-06-15", 1.50),
    ("2022-07-27", 2.25), ("2022-09-21", 3.00), ("2022-11-02", 3.75),
    ("2022-12-14", 4.25), ("2023-02-01", 4.50), ("2023-03-22", 4.75),
    ("2023-05-03", 5.00), ("2023-07-26", 5.25),
    ("2024-09-18", 4.75), ("2024-11-07", 4.50), ("2024-12-18", 4.25),
])

def get_ffr_before_meeting(meeting_date):
    """FFR lower bound in effect at the START of the meeting (pre-decision)."""
    rate = 4.75  # approximate pre-1998 default
    for date_str, new_rate in _FFR_CHANGES:
        if pd.Timestamp(date_str) < meeting_date:
            rate = new_rate
        else:
            break
    return rate

# ---------------------------------------------------------------------------
# Keyword features
# ---------------------------------------------------------------------------
_HAWKISH = [
    "inflation", "inflationary", "price stability", "overheating",
    "tightening", "restrictive", "elevated", "overheat", "price pressures",
]
_DOVISH = [
    "employment", "labor market", "labour market", "accommodative",
    "gradual", "patient", "uncertainty", "downside", "support",
    "below.*target", "slack", "soft",
]

def keyword_features(text):
    lower = text.lower()
    hawk = sum(len(re.findall(pat, lower)) for pat in _HAWKISH)
    dove = sum(len(re.findall(pat, lower)) for pat in _DOVISH)
    return np.array([hawk, dove, hawk - dove], dtype=np.float32)

# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------
_LABEL_PATTERNS = [
    (r"decided\s+to\s+raise\s+the\s+target\s+range",       "HIKE"),
    (r"decided\s+to\s+lower\s+the\s+target\s+range",       "CUT"),
    (r"decided\s+to\s+maintain\s+the\s+target\s+range",    "HOLD"),
    (r"decided\s+to\s+keep\s+the\s+target\s+range",        "HOLD"),
    (r"voted\s+to\s+(?:raise|increase)\s+the\s+target",    "HIKE"),
    (r"voted\s+to\s+(?:lower|reduce|decrease)\s+the\s+target", "CUT"),
    (r"voted\s+to\s+(?:maintain|keep)\s+the\s+target",     "HOLD"),
    (r"increase.*federal\s+funds\s+rate",                  "HIKE"),
    (r"(?:decrease|lower|reduce).*federal\s+funds\s+rate", "CUT"),
    (r"(?:maintain|keep).*federal\s+funds\s+rate",         "HOLD"),
]

def extract_own_label(text):
    lower = text.lower()
    for pattern, label in _LABEL_PATTERNS:
        if re.search(pattern, lower):
            return label
    return "UNKNOWN"

def build_labeled_dataframe(df):
    df = df.copy()
    df["own_label"] = df["text"].apply(extract_own_label)
    unknown = df["own_label"] == "UNKNOWN"
    if unknown.any():
        print(f"Dropping {unknown.sum()} row(s) with no rate decision detected:")
        print(df[unknown][["date", "filename"]].to_string(index=False))
    df = df[~unknown].reset_index(drop=True)
    # next_label is the decision at the FOLLOWING meeting — never in the input features
    df["next_label"] = df["own_label"].shift(-1)
    df = df[:-1].reset_index(drop=True)
    print(f"\nLabeled {len(df)} samples. next_label distribution:")
    print(df["next_label"].value_counts().to_string())
    return df

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def compute_embeddings(df, model):
    embeddings = []
    for i, row in df.iterrows():
        embeddings.append(model.get_embedding(row["text"], mask_decision=True)[0])
        if (i + 1) % 10 == 0:
            print(f"  Embedded {i + 1}/{len(df)} statements...")
    return np.array(embeddings, dtype=np.float32)

def load_or_compute_embeddings(df, model, cache_path):
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached.shape[0] == len(df):
            print(f"Loaded embeddings from cache: {cache_path}")
            return cached
        print(f"Cache shape mismatch ({cached.shape[0]} vs {len(df)}), recomputing...")
    print(f"Computing embeddings for {len(df)} statements...")
    embeddings = compute_embeddings(df, model)
    np.save(cache_path, embeddings)
    print(f"Saved embeddings to: {cache_path}")
    return embeddings

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def sentiment_delta(embeddings):
    """Cosine distance to previous embedding — captures how much language changed."""
    deltas = [0.0]
    for i in range(1, len(embeddings)):
        sim = cosine_similarity(embeddings[i - 1 : i], embeddings[i : i + 1])[0][0]
        deltas.append(1.0 - float(sim))
    return np.array(deltas, dtype=np.float32).reshape(-1, 1)

def single_sample_features(emb, prev_emb, meeting_date, text):
    """Build feature vector for one sample (used in both training and prediction)."""
    delta = 1.0 - float(
        cosine_similarity(prev_emb.reshape(1, -1), emb.reshape(1, -1))[0][0]
    )
    ffr = get_ffr_before_meeting(meeting_date)
    kw  = keyword_features(text)
    return np.hstack([emb, prev_emb, [delta], [ffr], kw]).astype(np.float32)

def build_feature_matrix(df, embeddings):
    """
    Feature vector per sample:
      current embedding (768) | prev embedding (768) | sentiment delta (1)
      | FFR going in (1) | keyword counts: hawk, dove, net (3)
    Total: 1541 dims
    """
    prev_embs = np.vstack([
        np.zeros((1, embeddings.shape[1]), dtype=np.float32),
        embeddings[:-1],
    ])
    rows = [
        single_sample_features(embeddings[i], prev_embs[i], row["date"], row["text"])
        for i, (_, row) in enumerate(df.iterrows())
    ]
    return np.array(rows, dtype=np.float32)

# ---------------------------------------------------------------------------
# Sklearn classifiers
# ---------------------------------------------------------------------------
def build_pipeline(model_type, C):
    clf = (
        LogisticRegression(
            C=C, class_weight="balanced",
            max_iter=1000, random_state=42,
        )
        if model_type == "logistic"
        else SVC(kernel="rbf", C=C, class_weight="balanced", probability=True, random_state=42)
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

def tune_C(X_train, y_train, X_val, y_val, model_type):
    best_pipeline, best_C, best_score = None, None, -1
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        p = build_pipeline(model_type, C)
        p.fit(X_train, y_train)
        score = f1_score(y_val, p.predict(X_val), average="macro", zero_division=0)
        print(f"  {model_type} C={C:.3f} → val macro-F1: {score:.3f}")
        if score > best_score:
            best_score, best_C, best_pipeline = score, C, p
    print(f"  Best C={best_C} (macro-F1={best_score:.3f})")
    return best_pipeline, best_C

# ---------------------------------------------------------------------------
# Transformer sequence model
# ---------------------------------------------------------------------------
class FedTransformer(nn.Module):
    def __init__(self, emb_dim=768, proj_dim=128, nhead=4, num_layers=2,
                 num_classes=3, seq_len=SEQ_LEN, dropout=0.1):
        super().__init__()
        self.proj    = nn.Linear(emb_dim, proj_dim)
        self.pos_emb = nn.Embedding(seq_len, proj_dim)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=proj_dim, nhead=nhead, dim_feedforward=256,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.drop    = nn.Dropout(dropout)
        self.head    = nn.Linear(proj_dim, num_classes)

    def forward(self, x, src_key_padding_mask=None):
        # x: (batch, seq_len, emb_dim)
        x   = self.proj(x)
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x   = x + self.pos_emb(pos)
        x   = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.head(self.drop(x[:, -1, :]))

def build_sequences(embeddings, seq_len):
    """Left-pad short sequences so every sample has shape (seq_len, emb_dim)."""
    n, d = embeddings.shape
    seqs  = np.zeros((n, seq_len, d), dtype=np.float32)
    masks = np.ones((n, seq_len), dtype=bool)   # True = padding (ignored by Transformer)
    for i in range(n):
        start  = max(0, i - seq_len + 1)
        actual = embeddings[start : i + 1]
        seqs[i, -len(actual) :] = actual
        masks[i, -len(actual) :] = False
    return seqs, masks

def train_transformer(X_seqs, X_masks, y, X_val_seqs, X_val_masks, y_val,
                      emb_dim, device, epochs=200, lr=5e-4):
    device = torch.device("cpu")  # MPS lacks some Transformer ops; CPU is fine for this model size
    model     = FedTransformer(emb_dim=emb_dim, seq_len=X_seqs.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    Xt  = torch.tensor(X_seqs,      device=device)
    Mt  = torch.tensor(X_masks,     device=device)
    yt  = torch.tensor(y,           dtype=torch.long, device=device)
    Xvt = torch.tensor(X_val_seqs,  device=device)
    Mvt = torch.tensor(X_val_masks, device=device)
    yvt = torch.tensor(y_val,       dtype=torch.long, device=device)

    best_f1, best_state, no_improve, patience = 0.0, None, 0, 20
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(Xt, Mt), yt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(Xvt, Mvt).argmax(dim=1).cpu().numpy()
            val_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
            print(f"  Epoch {epoch + 1:3d} | loss={loss.item():.4f} | val macro-F1={val_f1:.3f}")
            if val_f1 > best_f1:
                best_f1      = val_f1
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve   = 0
            else:
                no_improve += 1
                if no_improve >= patience // 10:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_f1

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def print_results(y_true, y_pred, y_proba, dates):
    print("\n--- Classification Report (small test set — high variance) ---")
    print(classification_report(y_true, y_pred, labels=LABEL_ORDER, zero_division=0))
    print("--- Prediction Probabilities ---")
    header = f"{'Date':<14}" + "".join(f"{c:>8}" for c in LABEL_ORDER)
    print(header)
    for date, probs in zip(dates, y_proba):
        print(f"{str(date)[:10]:<14}" + "".join(f"{p:>8.3f}" for p in probs))

def save_plots(y_true, y_pred, df_test):
    cm   = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — Next Meeting Rate Decision")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()

    label_to_y = {"CUT": 0, "HOLD": 1, "HIKE": 2}
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(
        df_test["date"], [label_to_y[l] for l in df_test["next_label"]],
        marker="o", s=80, label="True", color="#1f77b4", zorder=3,
    )
    ax.scatter(
        df_test["date"], [label_to_y[l] for l in y_pred],
        marker="x", s=100, label="Predicted", color="#d62728", zorder=3, linewidths=2,
    )
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["CUT", "HOLD", "HIKE"])
    ax.set_xlabel("Meeting Date")
    ax.set_title("Next Meeting Prediction: True vs Predicted")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "predictions_timeline.png"))
    plt.close()
    print("Saved confusion_matrix.png and predictions_timeline.png")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_training():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("--- Loading FinBERT ---")
    model = FedModel()
    device = model.device

    print(f"\n--- Loading statements from {DATA_DIR} ---")
    raw_df = model.load_data_from_dir(DATA_DIR)
    print(f"Found {len(raw_df)} files.")

    print("\n--- Extracting Labels ---")
    df = build_labeled_dataframe(raw_df)
    df.to_csv(os.path.join(RESULTS_DIR, "labeled_statements.csv"), index=False)
    n = len(df)

    print("\n--- Computing Embeddings (decision phrases masked) ---")
    embeddings = load_or_compute_embeddings(
        df, model, os.path.join(RESULTS_DIR, "embeddings_cache.npy")
    )

    print("\n--- Building Feature Matrix ---")
    X = build_feature_matrix(df, embeddings)
    print(f"Feature dimension: {X.shape[1]}")

    print("\n--- Splitting Data (chronological) ---")
    train_end, val_end = int(n * 0.65), int(n * 0.80)
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    df_train = df.iloc[:train_end]
    df_val   = df.iloc[train_end:val_end]
    df_test  = df.iloc[val_end:]

    le = LabelEncoder()
    le.fit(LABEL_ORDER)
    y_train = le.transform(df_train["next_label"])
    y_val   = le.transform(df_val["next_label"])
    y_test  = le.transform(df_test["next_label"])
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # --- Sklearn ---
    print("\n--- Tuning Logistic Regression ---")
    lr_pipeline, lr_C   = tune_C(X_train, y_train, X_val, y_val, "logistic")
    print("\n--- Tuning SVM ---")
    svm_pipeline, svm_C = tune_C(X_train, y_train, X_val, y_val, "svm")

    lr_f1  = f1_score(y_val, lr_pipeline.predict(X_val),  average="macro", zero_division=0)
    svm_f1 = f1_score(y_val, svm_pipeline.predict(X_val), average="macro", zero_division=0)
    best_sklearn_type = "logistic" if lr_f1 >= svm_f1 else "svm"
    best_C_sklearn    = lr_C if best_sklearn_type == "logistic" else svm_C
    best_sklearn_f1   = max(lr_f1, svm_f1)
    print(f"\nBest sklearn: {best_sklearn_type} (LR={lr_f1:.3f}, SVM={svm_f1:.3f})")

    # Always retrain final sklearn on train+val and save it (predict.py uses this)
    final_sklearn = build_pipeline(best_sklearn_type, best_C_sklearn)
    final_sklearn.fit(
        np.vstack([X_train, X_val]),
        np.concatenate([y_train, y_val]),
    )
    joblib.dump(final_sklearn, os.path.join(RESULTS_DIR, "rate_classifier.joblib"))

    # --- Transformer (large-data path) ---
    use_transformer = n >= TRANSFORMER_MIN_N
    transformer_f1  = -1.0
    if use_transformer:
        print(f"\n--- Training Transformer (n={n} >= {TRANSFORMER_MIN_N}) ---")
        seqs, masks = build_sequences(embeddings, SEQ_LEN)
        S_tr, M_tr  = seqs[:train_end],       masks[:train_end]
        S_val, M_val = seqs[train_end:val_end], masks[train_end:val_end]
        S_te, M_te   = seqs[val_end:],          masks[val_end:]

        tfm, transformer_f1 = train_transformer(
            S_tr, M_tr, y_train,
            S_val, M_val, y_val,
            emb_dim=embeddings.shape[1], device=device,
        )
        torch.save(tfm.state_dict(), os.path.join(RESULTS_DIR, "transformer_model.pt"))
        print(f"Transformer val macro-F1={transformer_f1:.3f}")
    else:
        print(f"\n--- Skipping Transformer (n={n} < {TRANSFORMER_MIN_N} required) ---")

    # --- Pick best model for test evaluation ---
    if use_transformer and transformer_f1 > best_sklearn_f1:
        print(f"\nUsing Transformer for test evaluation (val F1 {transformer_f1:.3f} > {best_sklearn_f1:.3f})")
        tfm.eval()
        with torch.no_grad():
            logits = tfm(
                torch.tensor(S_te, device=device),
                torch.tensor(M_te, device=device),
            ).cpu().numpy()
        proba    = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        y_pred   = le.inverse_transform(proba.argmax(axis=1))
        col_idx  = [list(le.classes_).index(label) for label in LABEL_ORDER]
        y_proba  = proba[:, col_idx]
    else:
        print(f"\nUsing {best_sklearn_type} for test evaluation (val F1 {best_sklearn_f1:.3f})")
        y_pred      = le.inverse_transform(final_sklearn.predict(X_test))
        y_proba_raw = final_sklearn.predict_proba(X_test)
        col_idx     = [list(le.classes_).index(label) for label in LABEL_ORDER]
        y_proba     = y_proba_raw[:, col_idx]

    print_results(df_test["next_label"].tolist(), y_pred, y_proba, df_test["date"].tolist())
    save_plots(df_test["next_label"].tolist(), y_pred, df_test.reset_index(drop=True))
    print(f"\nModel saved to {RESULTS_DIR}. Done.")


if __name__ == "__main__":
    run_training()
