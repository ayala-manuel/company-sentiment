# model/classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class Classifier(nn.Module):
    def __init__(self, emb_dim=768, hidden_dim=256, num_classes=3, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

class ClassifierWrapper:
    """
    Clase auxiliar para cargar un modelo entrenado y hacer predicciones.
    """
    def __init__(self, model_path: str, device: Union[str, torch.device] = None):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cargar checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        args = checkpoint['model_args']
        self.model = Classifier(**args).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.label_map = {0: -1, 1: 0, 2: 1}
        self.label_names = {-1: "negative", 0: "neutral", 1: "positive"}

    def predict_logits(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Predice los logits del modelo dado un embedding.
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)  # convertir a batch de 1
        return self.model(embedding.to(self.device))

    def predict_class(self, embedding: torch.Tensor) -> int:
        logits = self.predict_logits(embedding)
        pred_idx = logits.argmax(dim=1).item()
        return self.label_map[pred_idx]

    def predict_proba(self, embedding: torch.Tensor) -> dict:
        logits = self.predict_logits(embedding)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        return {self.label_names[self.label_map[i]]: float(probs[i]) for i in range(len(probs))}
