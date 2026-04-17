import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report, ConfusionMatrixDisplay, confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import FedModel

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
DATA_DIR = os.path.join(project_root, "data", "raw_html")
RESULTS_DIR = os.path.join(project_root, "data", "results")

LABEL_ORDER = ["CUT", "HOLD", "HIKE"]

_PATTERNS = [
    (r"decided\s+to\s+raise\s+the\s+target\s+range", "HIKE"),
    (r"decided\s+to\s+lower\s+the\s+target\s+range", "CUT"),
    (r"decided\s+to\s+maintain\s+the\s+target\s+range", "HOLD"),
    (r"decided\s+to\s+keep\s+the\s+target\s+range", "HOLD"),
]

# --- Labeling ---

def extract_own_label(text):
    lower = text.lower()
    for pattern, label in _PATTERNS:
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
    df["next_label"] = df["own_label"].shift(-1)
    df = df[:-1].reset_index(drop=True)
    print(f"\nLabeled {len(df)} samples. next_label distribution:")
    print(df["next_label"].value_counts().to_string())
    return df

# --- Embeddings ---

def compute_embeddings(df, model):
    embeddings = []
    for i, row in df.iterrows():
        embeddings.append(model.get_embedding(row["text"])[0])
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
    print(f"Saved embeddings to cache: {cache_path}")
    return embeddings

# --- Classifiers ---

def build_pipeline(model_type, C):
    clf = (
        LogisticRegression(C=C, class_weight="balanced", multi_class="multinomial", max_iter=1000, random_state=42)
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

# --- Evaluation ---

def print_results(y_true, y_pred, y_proba, dates):
    print("\n--- Classification Report (note: small test set, high variance) ---")
    print(classification_report(y_true, y_pred, labels=LABEL_ORDER, zero_division=0))

    print("--- Prediction Probabilities ---")
    header = f"{'Date':<14}" + "".join(f"{c:>8}" for c in LABEL_ORDER)
    print(header)
    for date, probs in zip(dates, y_proba):
        print(f"{str(date)[:10]:<14}" + "".join(f"{p:>8.3f}" for p in probs))

def save_plots(y_true, y_pred, df_test):
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — Next Meeting Rate Decision")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()

    label_to_y = {"CUT": 0, "HOLD": 1, "HIKE": 2}
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(df_test["date"], [label_to_y[l] for l in df_test["next_label"]],
               marker="o", s=80, label="True", color="#1f77b4", zorder=3)
    ax.scatter(df_test["date"], [label_to_y[l] for l in y_pred],
               marker="x", s=100, label="Predicted", color="#d62728", zorder=3, linewidths=2)
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

# --- Main ---

def run_training():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("--- Loading BERT Model ---")
    model = FedModel()

    print(f"\n--- Loading statements from {DATA_DIR} ---")
    raw_df = model.load_data_from_dir(DATA_DIR)
    print(f"Found {len(raw_df)} files.")

    print("\n--- Extracting Labels ---")
    df = build_labeled_dataframe(raw_df)
    df.to_csv(os.path.join(RESULTS_DIR, "labeled_statements.csv"), index=False)

    print("\n--- Computing Embeddings ---")
    X = load_or_compute_embeddings(df, model, os.path.join(RESULTS_DIR, "embeddings_cache.npy"))

    print("\n--- Splitting Data (chronological) ---")
    n = len(df)
    train_end, val_end = int(n * 0.65), int(n * 0.80)
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    df_train, df_val, df_test = df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

    le = LabelEncoder()
    le.fit(LABEL_ORDER)
    y_train = le.transform(df_train["next_label"])
    y_val   = le.transform(df_val["next_label"])
    y_test  = le.transform(df_test["next_label"])

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    print("\n--- Tuning Logistic Regression ---")
    lr_pipeline, lr_C = tune_C(X_train, y_train, X_val, y_val, "logistic")

    print("\n--- Tuning SVM ---")
    svm_pipeline, svm_C = tune_C(X_train, y_train, X_val, y_val, "svm")

    lr_f1  = f1_score(y_val, lr_pipeline.predict(X_val),  average="macro", zero_division=0)
    svm_f1 = f1_score(y_val, svm_pipeline.predict(X_val), average="macro", zero_division=0)
    best_type = "logistic" if lr_f1 >= svm_f1 else "svm"
    best_C    = lr_C if best_type == "logistic" else svm_C
    print(f"\nBest model: {best_type} (LR macro-F1={lr_f1:.3f}, SVM macro-F1={svm_f1:.3f})")

    print(f"Retraining {best_type} on train+val...")
    final = build_pipeline(best_type, best_C)
    final.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))

    y_pred        = le.inverse_transform(final.predict(X_test))
    y_proba       = final.predict_proba(X_test)
    proba_ordered = y_proba[:, [list(le.classes_).index(i) for i in range(len(LABEL_ORDER))]]

    print_results(df_test["next_label"].tolist(), y_pred, proba_ordered, df_test["date"].tolist())
    save_plots(df_test["next_label"].tolist(), y_pred, df_test.reset_index(drop=True))

    joblib.dump(final, os.path.join(RESULTS_DIR, "rate_classifier.joblib"))
    print(f"\nModel saved. Done.")

if __name__ == "__main__":
    run_training()
