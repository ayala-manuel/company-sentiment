# app/services/prediction_service.py
import nltk
import numpy as np
import torch

from model.embeddings import EmbeddingModel
from model.classifier import ClassifierWrapper
from app.config import EMBEDDING_MODEL, MODEL_PATH, DEVICE, ID2LABEL
from utils.text_cleaner import clean_text


class PredictionService:
    def __init__(self):
        self.embedder = EmbeddingModel(model_name=EMBEDDING_MODEL, device=DEVICE)
        self.model = ClassifierWrapper(model_path=MODEL_PATH, device=DEVICE)
        self.model.model.eval()

    def predict(self, text: str):
        # 1. Tokenizar en frases
        sentences = nltk.sent_tokenize(text)

        # 2. Limpiar y codificar
        cleaned = [clean_text(s) for s in sentences]
        embeddings = self.embedder.encode(cleaned)
        embeddings = torch.tensor(embeddings).to(DEVICE)

        # 3. Predicción
        with torch.no_grad():
            logits = self.model.predict_logits(embeddings)
            preds = logits.argmax(dim=1).cpu().numpy()

        # 4. Sentimiento promedio
        avg = np.mean(preds)
        label = ID2LABEL[int(round(avg))]

        # 5. Resultado por oración
        sentence_results = [
            {"sentence": s, "label": ID2LABEL[p], "score": int(p)}
            for s, p in zip(sentences, preds)
        ]

        return {
            "overall_sentiment": {"label": label, "mean_score": float(avg)},
            "sentence_sentiments": sentence_results
        }