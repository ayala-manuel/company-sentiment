# model/predictor.py

from model.embeddings import EmbeddingModel
from model.classifier import ClassifierWrapper
from utils.text_cleaner import clean_text
import torch
from typing import List, Union

class SentimentPredictor:
    def __init__(self,
                 model_path: str = "./model.pt",
                 embedding_model: str = "philschmid/bge-base-financial-matryoshka",
                 device: Union[str, torch.device] = None):
        self.embedder = EmbeddingModel(model_name=embedding_model, device=device)
        self.classifier = ClassifierWrapper(model_path=model_path, device=device)

    def predict(self, text: str) -> int:
        clean = clean_text(text)
        emb = self.embedder.encode(clean)
        return self.classifier.predict_class(emb)

    def predict_proba(self, text: str) -> dict:
        clean = clean_text(text)
        emb = self.embedder.encode(clean)
        return self.classifier.predict_proba(emb)

    def predict_all(self, text: str) -> dict:
        """
        Devuelve clase + probabilidades.
        """
        pred_class = self.predict(text)
        pred_proba = self.predict_proba(text)
        return {
            "predicted_class": pred_class,
            "probabilities": pred_proba
        }

    def batch_predict(self, sentences: List[str]) -> List[dict]:
        """
        Recibe lista de frases, devuelve predicciones y probabilidades.
        """
        clean_sentences = [clean_text(s) for s in sentences]
        embeddings = self.embedder.encode(clean_sentences)
        results = []

        for i in range(len(sentences)):
            emb = embeddings[i].unsqueeze(0)
            pred = self.classifier.predict_class(emb)
            proba = self.classifier.predict_proba(emb)
            results.append({
                "sentence": sentences[i],
                "predicted_class": pred,
                "probabilities": proba
            })
        return results
