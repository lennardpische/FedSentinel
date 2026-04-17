import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

def build_logistic_pipeline(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            class_weight="balanced",
            multi_class="multinomial",
            max_iter=1000,
            random_state=42,
        )),
    ])

def build_svm_pipeline(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=C,
            class_weight="balanced",
            probability=True,
            random_state=42,
        )),
    ])

def tune_C(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    C_values: list = [0.001, 0.01, 0.1, 1.0, 10.0],
    model_type: str = "logistic",
) -> tuple:
    best_pipeline, best_C, best_score = None, None, -1
    builder = build_logistic_pipeline if model_type == "logistic" else build_svm_pipeline

    for C in C_values:
        pipeline = builder(C)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        score = f1_score(y_val, y_pred, average="macro", zero_division=0)
        print(f"  {model_type} C={C:.4f} → val macro-F1: {score:.3f}")
        if score > best_score:
            best_score, best_C, best_pipeline = score, C, pipeline

    print(f"  Best C={best_C} (macro-F1={best_score:.3f})")
    return best_pipeline, best_C

def save_model(pipeline: Pipeline, path: str) -> None:
    joblib.dump(pipeline, path)
    print(f"Model saved to {path}")

def load_model(path: str) -> Pipeline:
    return joblib.load(path)
