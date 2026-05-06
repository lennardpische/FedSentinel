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

MODEL_PATH = os.path.join("data", "results", "rate_classifier.joblib")

print("Loading FinBERT...")
bert = FedModel()
print("Loading rate classifier...")
pipeline = joblib.load(MODEL_PATH)

_le = LabelEncoder()
_le.fit(LABEL_ORDER)

EXAMPLE_STATEMENT = (
    "Recent indicators suggest that economic activity has continued to expand at a solid pace. "
    "Job gains have moderated, and the unemployment rate has moved up but remains low. "
    "Inflation has eased over the past year but remains somewhat elevated. In recent months, "
    "there has been some further progress toward the Committee's 2 percent inflation objective. "
    "The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent "
    "over the longer run. The Committee judges that the risks to achieving its employment and "
    "inflation goals continue to move into better balance. The economic outlook is uncertain, "
    "and the Committee is attentive to the risks to both sides of its dual mandate. "
    "In support of its goals, the Committee decided to maintain the target range for the federal "
    "funds rate at 5-1/4 to 5-1/2 percent."
)


def predict(statement: str, prev_statement: str, meeting_date_str: str):
    if not statement.strip():
        return "Please paste an FOMC statement.", None

    emb = bert.get_embedding(statement, mask_decision=True)[0]
    prev_emb = (
        bert.get_embedding(prev_statement, mask_decision=True)[0]
        if prev_statement.strip()
        else np.zeros_like(emb)
    )

    try:
        meeting_date = pd.Timestamp(meeting_date_str)
    except Exception:
        meeting_date = pd.Timestamp.today()

    features = single_sample_features(emb, prev_emb, meeting_date, statement).reshape(1, -1)

    proba_raw = pipeline.predict_proba(features)[0]
    col_idx   = [list(_le.classes_).index(label) for label in LABEL_ORDER]
    proba     = proba_raw[col_idx]

    pred_label = LABEL_ORDER[int(proba.argmax())]
    confidence = float(proba.max())

    colors = {"CUT": "#e05252", "HOLD": "#f0a500", "HIKE": "#4caf7d"}
    fig, ax = plt.subplots(figsize=(5, 2.5))
    bars = ax.barh(LABEL_ORDER, proba, color=[colors[l] for l in LABEL_ORDER], height=0.45)
    for bar, prob in zip(bars, proba):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.1%}",
            va="center",
            fontsize=11,
        )
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Probability")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    return f"{pred_label}  ({confidence:.1%} confidence)", fig


with gr.Blocks(title="FedSentinel") as demo:
    gr.Markdown("## FedSentinel — FOMC Rate Decision Predictor")
    gr.Markdown(
        "Paste an FOMC statement to predict the **next** meeting's rate decision: "
        "**Cut**, **Hold**, or **Hike**."
    )

    statement_input = gr.Textbox(
        label="Current FOMC Statement",
        lines=8,
        placeholder="Paste the full statement here...",
    )
    with gr.Row():
        prev_input = gr.Textbox(
            label="Previous Statement (optional — improves accuracy)",
            lines=4,
            placeholder="Paste previous statement or leave blank...",
            scale=2,
        )
        date_input = gr.Textbox(
            label="Meeting Date (YYYY-MM-DD)",
            value=str(pd.Timestamp.today().date()),
            scale=1,
        )

    predict_btn = gr.Button("Predict Next Decision", variant="primary")

    with gr.Row():
        label_out = gr.Textbox(label="Predicted Next Decision", interactive=False, scale=1)
        chart_out = gr.Plot(label="Probabilities", scale=2)

    predict_btn.click(
        fn=predict,
        inputs=[statement_input, prev_input, date_input],
        outputs=[label_out, chart_out],
    )

    gr.Examples(
        examples=[[EXAMPLE_STATEMENT, "", "2024-07-31"]],
        inputs=[statement_input, prev_input, date_input],
    )

demo.launch()
