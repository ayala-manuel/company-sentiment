# model/embeddings.py

from sentence_transformers import SentenceTransformer
import torch
from typing import List, Union

class EmbeddingModel:
    def __init__(self, model_name: str = "philschmid/bge-base-financial-matryoshka", device: Union[str, torch.device] = None):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name).to(self.device)

    def encode(self, texts: Union[str, List[str]], convert_to_tensor=True) -> Union[torch.Tensor, List[float]]:
        """
        Embebe uno o varios textos.

        Parámetros:
            texts (str o List[str]): texto(s) a convertir en embedding.
            convert_to_tensor (bool): si True, devuelve tensores PyTorch; si False, listas.

        Retorna:
            embeddings (torch.Tensor o List[float]): representaciones vectoriales.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            device=self.device,
            convert_to_tensor=convert_to_tensor
        )
        return embeddings
