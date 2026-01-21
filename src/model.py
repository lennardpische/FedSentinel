import os
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import re

class FedModel:
    def __init__(self, model_name = 'bert-base-uncased'):
        print(f"Loading model: {model_name}...")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available:
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}.")

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def preprocess_text(self, text):
        if not text: return ""
        text = re.sub(r'\s+','', text)
        text = re.sub(r'\[.*?\]','',text)
        return text.strip()
    
    def get_embedding(self, text):
        clean_text = self.preprocess_text(text)
        inputs = self.tokenizer(
            clean_text,
            return_tensor = 'pt',
            padding=True,
            truncation=True,
            max_length = 512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    def compare_statements(self, text1, text2):
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        return cosine_similarity(emb1, emb2)[0][0]
    
    def load_data_from_dir(self, directory):
        data = []
        files = sorted(os.listdir(directory))
        for filename in files:
            if filename.endswith('.txt'):
                path = os.path.join(directory, filename)
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()

                date_match = re.search(r'(\d{8})', filename)
                date = date_match.group(1) if date_match else "Unknown"
                data.append({'date': date, 'text': text, 'filename': filename})

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'], format = '%Y%m%d', errors='coerce')
        return df.sort_values('date').reset_index(drop=True)

