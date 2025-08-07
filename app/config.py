# app/config.py

import nltk
import torch

# Descargar el tokenizer si es la primera vez
nltk.download('punkt')

MODEL_PATH = "./model.pt"
EMBEDDING_MODEL = "philschmid/bge-base-financial-matryoshka"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}