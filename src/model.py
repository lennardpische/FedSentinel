import os
import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


class FedModel:
    # Sentences that state the current rate decision — masked before embedding
    # so the representation captures policy *reasoning*, not the decision itself.
    _DECISION_RE = re.compile(
        r"(?:decided|voted)\s+to\s+(?:raise|increase|lower|reduce|decrease|maintain|keep)"
        r"\s+the\s+target\s+range[^.]*\.",
        re.IGNORECASE,
    )

    def __init__(self, model_name="ProsusAI/finbert"):
        print(f"Loading model: {model_name}...")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def preprocess_text(self, text):
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\[.*?\]", "", text)
        return text.strip()

    def mask_decision_phrases(self, text):
        return self._DECISION_RE.sub("", text)

    def get_embedding(self, text, mask_decision=True):
        """
        Returns a (1, hidden_size) numpy array.

        Mean pooling: sum token embeddings weighted by attention mask, then
        divide by token count — this gives a stable sentence representation
        vs the single [CLS] vector.
        """
        if mask_decision:
            text = self.mask_decision_phrases(text)
        clean = self.preprocess_text(text)

        inputs = self.tokenizer(
            clean,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        token_embs   = outputs.last_hidden_state          # (1, seq_len, H)
        mask         = inputs["attention_mask"]            # (1, seq_len)
        mask_exp     = mask.unsqueeze(-1).float()          # (1, seq_len, 1)
        sum_embs     = (token_embs * mask_exp).sum(dim=1)  # (1, H)
        count        = mask_exp.sum(dim=1).clamp(min=1e-9) # (1, 1)
        return (sum_embs / count).cpu().numpy()            # (1, H)

    def compare_statements(self, text1, text2):
        return cosine_similarity(self.get_embedding(text1), self.get_embedding(text2))[0][0]

    def load_data_from_dir(self, directory):
        data = []
        for filename in sorted(os.listdir(directory)):
            if not filename.endswith(".txt"):
                continue
            path = os.path.join(directory, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            date_match = re.search(r"(\d{8})", filename)
            date = date_match.group(1) if date_match else "Unknown"
            data.append({"date": date, "text": text, "filename": filename})

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
        return df.sort_values("date").reset_index(drop=True)
