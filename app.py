import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

matplotlib.use("Agg")

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from model import FedModel
from train import single_sample_features, LABEL_ORDER

MODEL_PATH = "rate_classifier.joblib"

bert = FedModel()
pipeline = joblib.load(MODEL_PATH)

_le = LabelEncoder()
_le.fit(LABEL_ORDER)

EXAMPLE = (
    "Recent indicators suggest that economic activity has continued to expand at a solid pace. "
    "Job gains have moderated, and the unemployment rate has moved up but remains low. "
    "Inflation has eased over the past year but remains somewhat elevated. In recent months, "
    "there has been some further progress toward the Committee's 2 percent inflation objective. "
    "The Committee decided to maintain the target range for the federal funds rate at 5-1/4 to 5-1/2 percent."
)


def predict(statement: str):
    if not statement.strip():
        return "Paste a statement first.", None

    emb = bert.get_embedding(statement, mask_decision=True)[0]
    features = single_sample_features(
        emb, np.zeros_like(emb), pd.Timestamp.today(), statement
    ).reshape(1, -1)

    proba_raw = pipeline.predict_proba(features)[0]
    col_idx   = [list(_le.classes_).index(l) for l in LABEL_ORDER]
    proba     = proba_raw[col_idx]
    pred      = LABEL_ORDER[int(proba.argmax())]

    colors = {"CUT": "#e05252", "HOLD": "#f0a500", "HIKE": "#4caf7d"}
    fig, ax = plt.subplots(figsize=(4, 2))
    bars = ax.barh(LABEL_ORDER, proba, color=[colors[l] for l in LABEL_ORDER], height=0.4)
    for bar, p in zip(bars, proba):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{p:.0%}", va="center", fontsize=10)
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("Probability")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(left=False)
    plt.tight_layout()

    return f"{pred} — {float(proba.max()):.0%} confidence", fig


with gr.Blocks(title="FedSentinel", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# FedSentinel\nPaste an FOMC statement to predict the next rate decision.")

    statement_input = gr.Textbox(
        label="FOMC Statement",
        lines=7,
        placeholder="Paste the statement here...",
    )
    predict_btn = gr.Button("Predict", variant="primary")

    with gr.Row():
        label_out = gr.Textbox(label="Next Decision", interactive=False, scale=1)
        chart_out = gr.Plot(label="Probabilities", scale=2)

    predict_btn.click(fn=predict, inputs=statement_input, outputs=[label_out, chart_out])
    gr.Examples(examples=[[EXAMPLE]], inputs=statement_input)

demo.launch()
