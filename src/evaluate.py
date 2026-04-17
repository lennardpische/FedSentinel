import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

LABEL_ORDER = ["CUT", "HOLD", "HIKE"]

def print_classification_report(y_true, y_pred) -> None:
    print("\n--- Classification Report ---")
    print(f"(Note: test set is small; metrics have high variance)\n")
    print(classification_report(y_true, y_pred, labels=LABEL_ORDER, zero_division=0))

def plot_confusion_matrix(y_true, y_pred, save_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — Next Meeting Rate Decision")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_predictions_timeline(df_test: pd.DataFrame, y_pred, save_path: str) -> None:
    label_to_y = {"CUT": 0, "HOLD": 1, "HIKE": 2}
    true_y = [label_to_y[l] for l in df_test["next_label"]]
    pred_y = [label_to_y[l] for l in y_pred]
    dates = df_test["date"].tolist()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(dates, true_y, marker="o", s=80, label="True", color="#1f77b4", zorder=3)
    ax.scatter(dates, pred_y, marker="x", s=100, label="Predicted", color="#d62728", zorder=3, linewidths=2)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["CUT", "HOLD", "HIKE"])
    ax.set_xlabel("Meeting Date")
    ax.set_title("Next Meeting Prediction: True vs Predicted")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Predictions timeline saved to {save_path}")

def print_probability_table(dates, y_proba: np.ndarray, class_names: list) -> None:
    print("\n--- Prediction Probabilities (test set) ---")
    header = f"{'Date':<14}" + "".join(f"{c:>8}" for c in class_names)
    print(header)
    print("-" * len(header))
    for date, probs in zip(dates, y_proba):
        row = f"{str(date)[:10]:<14}" + "".join(f"{p:>8.3f}" for p in probs)
        print(row)
