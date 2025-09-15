from typing import List, Sequence
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _minmax(x: np.ndarray) -> np.ndarray:
    # compatible NumPy 2.0 (utilise np.ptp)
    rng = np.ptp(x)  # == x.max() - x.min()
    if rng == 0:
        return np.zeros_like(x)
    return (x - x.min()) / (rng + 1e-8)

class HybridReranker:
    def __init__(self, docs: Sequence[str]):
        self.vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,2))
        self.X = self.vectorizer.fit_transform(docs)

    def score(self, q_text: str, cand_indices: List[int], sem_sims: np.ndarray, alpha: float = 0.75):
        q = self.vectorizer.transform([q_text])
        tfidf_sims = cosine_similarity(q, self.X[cand_indices]).ravel()
        # normalisation
        sem_sims = _minmax(sem_sims)
        tfidf_sims = _minmax(tfidf_sims)
        final = alpha * sem_sims + (1 - alpha) * tfidf_sims
        order = np.argsort(-final)
        return order, final[order]
