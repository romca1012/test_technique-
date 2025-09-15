import numpy as np
from sklearn.neighbors import NearestNeighbors

class KnnIndex:
    def __init__(self, n_neighbors: int = 10):
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
        self._fitted = False
        self._ids = None
        self._movie_ids = None
        self._emb = None

    def fit(self, emb: np.ndarray, ids, movie_ids):
        self.nn.fit(emb)
        self._fitted = True
        self._emb = emb
        self._ids = np.array(list(ids))
        self._movie_ids = np.array(list(movie_ids))

    def query_same_movie(self, emb_query: np.ndarray, movie_id: str, k: int):
        if not self._fitted:
            raise RuntimeError("Index not fitted.")
        pool_k = min(max(50, 5*k), len(self._ids))
        distances, indices = self.nn.kneighbors(emb_query, n_neighbors=pool_k)
        idxs = indices[0]
        dists = distances[0]
        mask = (self._movie_ids[idxs] == movie_id)
        idxs = idxs[mask]
        dists = dists[mask]
        sims = 1 - dists
        order = np.argsort(-sims)
        idxs = idxs[order][:k]
        sims = sims[order][:k]
        return self._ids[idxs], sims