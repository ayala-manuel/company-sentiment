# api.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
import nltk
import numpy as np

from model.embeddings import EmbeddingModel
from model.classifier import ClassifierWrapper
from utils.text_cleaner import clean_text

# Descargar el tokenizer si es la primera vez
nltk.download('punkt')

# Config
MODEL_PATH = "./model.pt"
EMBEDDING_MODEL = "philschmid/bge-base-financial-matryoshka"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicialización
app = FastAPI()
embedder = EmbeddingModel(model_name=EMBEDDING_MODEL, device=DEVICE)
model = ClassifierWrapper(model_path=MODEL_PATH, device=DEVICE)
model.model.eval()

# Mapping inverso
id2label = {0: "negative", 1: "neutral", 2: "positive"}

# Entrada esperada
class TextInput(BaseModel):
    text: str

# Endpoint
@app.post("/predict")
def predict_sentiment(input: TextInput):
    # 1. Tokenizar en frases
    sentences = nltk.sent_tokenize(input.text)

    # 2. Limpiar y codificar
    cleaned = [clean_text(s) for s in sentences]
    embeddings = embedder.encode(cleaned)
    embeddings = torch.tensor(embeddings).to(DEVICE)

    # 3. Predicción
    with torch.no_grad():
        logits = model.predict_logits(embeddings)
        preds = logits.argmax(dim=1).cpu().numpy()

    # 4. Sentimiento promedio
    avg_sentiment = np.mean(preds)
    avg_rounded = int(round(avg_sentiment))
    overall = id2label[avg_rounded]

    # 5. Preparar respuesta
    result = {
        "overall_sentiment": {
            "label": overall,
            "mean_score": float(avg_sentiment)
        },
        "sentence_sentiments": [
            {"sentence": s, "label": id2label[p], "score": int(p)}
            for s, p in zip(sentences, preds)
        ]
    }

    return result
