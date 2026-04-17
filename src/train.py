import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import FedModel
from labeler import build_labeled_dataframe
from embedder import load_or_compute_embeddings
from classifier import build_logistic_pipeline, build_svm_pipeline, tune_C, save_model
from evaluate import (
    print_classification_report,
    plot_confusion_matrix,
    plot_predictions_timeline,
    print_probability_table,
    LABEL_ORDER,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
DATA_DIR = os.path.join(project_root, "data", "raw_html")
RESULTS_DIR = os.path.join(project_root, "data", "results")

def run_training(use_cache: bool = True):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("--- Loading BERT Model ---")
    model = FedModel()

    print(f"\n--- Loading statements from {DATA_DIR} ---")
    raw_df = model.load_data_from_dir(DATA_DIR)
    print(f"Found {len(raw_df)} files.")

    print("\n--- Extracting Labels ---")
    df = build_labeled_dataframe(raw_df)
    df.to_csv(os.path.join(RESULTS_DIR, "labeled_statements.csv"), index=False)
    print(f"Saved labeled_statements.csv ({len(df)} rows)")

    print("\n--- Computing Embeddings ---")
    cache_path = os.path.join(RESULTS_DIR, "embeddings_cache.npy")
    # embeddings_cache must align with labeled df, not raw_df
    # so we pass the filtered df and bust cache if row count changed
    X = load_or_compute_embeddings(df, model, cache_path)

    print("\n--- Splitting Data (chronological) ---")
    n = len(df)
    train_end = int(n * 0.65)
    val_end = int(n * 0.80)

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
    print(f"Train label dist: {dict(zip(*np.unique(df_train['next_label'], return_counts=True)))}")

    print("\n--- Tuning Logistic Regression ---")
    lr_pipeline, lr_C = tune_C(X_train, y_train, X_val, y_val, model_type="logistic")

    print("\n--- Tuning SVM ---")
    svm_pipeline, svm_C = tune_C(X_train, y_train, X_val, y_val, model_type="svm")

    # Pick best model by val macro-F1
    from sklearn.metrics import f1_score
    lr_val_f1  = f1_score(y_val, lr_pipeline.predict(X_val),  average="macro", zero_division=0)
    svm_val_f1 = f1_score(y_val, svm_pipeline.predict(X_val), average="macro", zero_division=0)

    best_type = "logistic" if lr_val_f1 >= svm_val_f1 else "svm"
    best_C    = lr_C if best_type == "logistic" else svm_C
    builder   = build_logistic_pipeline if best_type == "logistic" else build_svm_pipeline
    print(f"\nBest model: {best_type} (val macro-F1 LR={lr_val_f1:.3f}, SVM={svm_val_f1:.3f})")

    # Retrain on train+val combined
    print(f"Retraining {best_type} (C={best_C}) on train+val...")
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    final_pipeline = builder(best_C)
    final_pipeline.fit(X_trainval, y_trainval)

    # Evaluate on test set
    y_pred = final_pipeline.predict(X_test)
    y_pred_labels = le.inverse_transform(y_pred)
    y_proba = final_pipeline.predict_proba(X_test)
    # reorder proba columns to match LABEL_ORDER
    proba_order = [list(le.classes_).index(c) for c in LABEL_ORDER]
    y_proba_ordered = y_proba[:, proba_order]

    print_classification_report(df_test["next_label"].tolist(), y_pred_labels)
    print_probability_table(df_test["date"].tolist(), y_proba_ordered, LABEL_ORDER)

    plot_confusion_matrix(
        df_test["next_label"].tolist(), y_pred_labels,
        os.path.join(RESULTS_DIR, "confusion_matrix.png")
    )
    plot_predictions_timeline(
        df_test.reset_index(drop=True), y_pred_labels,
        os.path.join(RESULTS_DIR, "predictions_timeline.png")
    )

    save_model(final_pipeline, os.path.join(RESULTS_DIR, "rate_classifier.joblib"))
    print("\nDone. All outputs saved to data/results/")

if __name__ == "__main__":
    run_training()
