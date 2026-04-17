import os
import sys
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import FedModel
from train import LABEL_ORDER

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
MODEL_PATH = os.path.join(project_root, "data", "results", "rate_classifier.joblib")

def predict_next_decision(text, model_path=MODEL_PATH):
    bert = FedModel()
    pipeline = joblib.load(model_path)

    embedding = bert.get_embedding(text)  # (1, 768)
    pred_int  = int(pipeline.predict(embedding)[0])
    proba     = pipeline.predict_proba(embedding)[0]

    prediction = LABEL_ORDER[pred_int]
    prob_dict  = {LABEL_ORDER[i]: float(proba[i]) for i in range(len(LABEL_ORDER))}

    return {
        "prediction": prediction,
        "probabilities": prob_dict,
        "confidence": prob_dict[prediction],
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_statement.txt>")
        print('       python src/predict.py "The Committee decided to..."')
        sys.exit(1)

    arg = sys.argv[1]
    text = open(arg, encoding="utf-8").read() if os.path.isfile(arg) else arg

    result = predict_next_decision(text)
    print(f"\nPredicted next decision: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print("Probabilities:")
    for label, prob in result["probabilities"].items():
        print(f"  {label}: {prob:.3f}")
