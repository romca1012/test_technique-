from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return emb