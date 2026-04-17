import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
MODEL_PATH = os.path.join(project_root, "data", "results", "rate_classifier.joblib")

from evaluate import LABEL_ORDER

def predict_next_decision(text: str, model_path: str = MODEL_PATH) -> dict:
    from model import FedModel
    from classifier import load_model
    import numpy as np

    bert = FedModel()
    pipeline = load_model(model_path)

    embedding = bert.get_embedding(text)  # shape (1, 768)
    prediction = pipeline.predict(embedding)[0]
    proba = pipeline.predict_proba(embedding)[0]

    # pipeline.classes_ is the integer-encoded label array; map back to strings
    le_classes = pipeline.classes_  # e.g. [0, 1, 2] corresponding to LABEL_ORDER
    prob_dict = {LABEL_ORDER[cls]: float(prob) for cls, prob in zip(le_classes, proba)}

    return {
        "prediction": LABEL_ORDER[int(prediction)],
        "probabilities": prob_dict,
        "confidence": float(proba[list(le_classes).index(prediction)]),
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_statement.txt>")
        print('       python src/predict.py "The Committee decided to..."')
        sys.exit(1)

    arg = sys.argv[1]
    if os.path.isfile(arg):
        with open(arg, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = arg

    result = predict_next_decision(text)
    print(f"\nPredicted next decision: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print("Probabilities:")
    for label, prob in result["probabilities"].items():
        print(f"  {label}: {prob:.3f}")
