import os
import numpy as np
import pandas as pd

def extract_embeddings(df: pd.DataFrame, model) -> np.ndarray:
    embeddings = []
    for i, row in df.iterrows():
        emb = model.get_embedding(row["text"])  # shape (1, 768)
        embeddings.append(emb[0])
        if (i + 1) % 10 == 0:
            print(f"  Embedded {i + 1}/{len(df)} statements...")
    return np.array(embeddings, dtype=np.float32)

def load_or_compute_embeddings(df: pd.DataFrame, model, cache_path: str) -> np.ndarray:
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached.shape[0] == len(df):
            print(f"Loaded embeddings from cache: {cache_path}")
            return cached
        print(f"Cache shape mismatch ({cached.shape[0]} vs {len(df)}), recomputing...")

    print(f"Computing embeddings for {len(df)} statements...")
    embeddings = extract_embeddings(df, model)
    np.save(cache_path, embeddings)
    print(f"Saved embeddings to cache: {cache_path}")
    return embeddings
