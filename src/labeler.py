import re
import pandas as pd

HIKE = "HIKE"
CUT = "CUT"
HOLD = "HOLD"
UNKNOWN = "UNKNOWN"

_PATTERNS = [
    (r"decided\s+to\s+raise\s+the\s+target\s+range", HIKE),
    (r"decided\s+to\s+lower\s+the\s+target\s+range", CUT),
    (r"decided\s+to\s+maintain\s+the\s+target\s+range", HOLD),
    (r"decided\s+to\s+keep\s+the\s+target\s+range", HOLD),
]

def extract_own_label(text: str) -> str:
    lower = text.lower()
    for pattern, label in _PATTERNS:
        if re.search(pattern, lower):
            return label
    return UNKNOWN

def build_labeled_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["own_label"] = df["text"].apply(extract_own_label)

    unknown_mask = df["own_label"] == UNKNOWN
    if unknown_mask.any():
        print(f"Dropping {unknown_mask.sum()} row(s) with no rate decision detected:")
        print(df[unknown_mask][["date", "filename"]].to_string(index=False))
    df = df[~unknown_mask].reset_index(drop=True)

    df["next_label"] = df["own_label"].shift(-1)
    df = df[:-1].reset_index(drop=True)  # drop last row (no future label)

    print(f"\nLabeled {len(df)} samples. next_label distribution:")
    print(df["next_label"].value_counts().to_string())
    return df
