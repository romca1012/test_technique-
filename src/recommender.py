from typing import List, Dict, Any
import numpy as np

from .config import settings
from .preprocessing import normalize_text, make_corpus_row
from .embedder import TextEmbedder
from .indexer import KnnIndex
from .hybrid_ranker import HybridReranker
from .explainer import best_overlapping_sentences, keywords_hint
from .repository import ReviewRepo


def to_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.strip().replace(',', '.')
            if x == '':
                return None
        return float(x)
    except Exception:
        return None


class ReviewRecommender:
    def __init__(self, data_path: str = None):
        # Charge un fichier unique si data_path est fourni, sinon les sources (2 CSV)
        if data_path:
            self.repo = ReviewRepo(path=data_path, sources=None)
        else:
            self.repo = ReviewRepo(path=None, sources=settings.sources)

        df = self.repo.all().copy()

        #si aucun enregistrement n'a été chargé, on arrête proprement
        if df.empty:
            raise RuntimeError(
                "Le corpus est vide après chargement. Vérifie src/config.py (chemins des CSV) "
                "et que les CSV contiennent des colonnes 'title'/'body' (ou leurs alias)."
            )

        # Construit le texte normalisé (titre. corps) + garde le body seul
        import pandas as pd  # import local OK
        df["text_norm"], df["body_only"] = zip(*df.apply(
            lambda r: make_corpus_row(
                normalize_text(r.get("title", "")),
                normalize_text(r.get("body", ""))
            ), axis=1
        ))

        # Filtre: garde les lignes avec un minimum de contenu textuel
        df = df[df["text_norm"].str.len() >= 5].reset_index(drop=True)

        # Si plus rien à indexer, expliciter l'erreur (mieux qu'un 500 obscur)
        if df.empty:
            raise RuntimeError(
                "Aucune critique exploitable après normalisation (vérifie que 'title'/'body' ne sont pas vides/'None')."
            )

        self.df = df

        # Embeddings + index
        self.embedder = TextEmbedder(settings.model_name)
        self.emb = self.embedder.encode(self.df["text_norm"].tolist())

        self.index = KnnIndex(n_neighbors=50)
        self.index.fit(
            self.emb,
            self.df["review_id"].astype(str).tolist(),
            self.df["movie_id"].astype(str).tolist()
        )

        self.reranker = HybridReranker(self.df["text_norm"].tolist())

    def similar(self, review_id: str, k: int = 5, min_sim: float = None) -> List[Dict[str, Any]]:
        min_sim = settings.min_sim if min_sim is None else min_sim
        q = self.repo.by_id(review_id)
        movie_id = str(q["movie_id"])

        q_text, q_body = make_corpus_row(
            normalize_text(q.get("title", "")),
            normalize_text(q.get("body", ""))
        )
        q_emb = self.embedder.encode([q_text])

        cand_ids, sem_sims = self.index.query_same_movie(q_emb, movie_id=movie_id, k=50)

        # retire la critique source
        mask = cand_ids != str(review_id)
        cand_ids = cand_ids[mask]
        sem_sims = sem_sims[mask]

        if len(cand_ids) == 0:
            return []

        # map review_id -> index ligne pour le rerank
        idx_map = {rid: i for i, rid in enumerate(self.df["review_id"].astype(str).tolist())}
        cand_indices = [idx_map[rid] for rid in cand_ids]

        # rerank hybride (sem + tfidf)
        order, scores = self.reranker.score(q_text, cand_indices, sem_sims)
        cand_indices = [cand_indices[i] for i in order]
        sem_sims = sem_sims[order]

        out = []
        for i, idx in enumerate(cand_indices[:k]):
            row = self.df.iloc[idx]
            sim = float(sem_sims[i])
            if sim < min_sim:
                continue

            # champs sûrs (jamais "None" chaîne)
            title = (row.get("title") or "")
            body = (row.get("body") or "")
            snippet = (body[:220] + "…") if len(body) > 220 else body

            out.append({
                "review_id": str(row["review_id"]),
                "movie_id": str(row["movie_id"]),
                "movie_title": row.get("movie_title"),
                "user_id": row.get("user_id"),
                "rating": to_float(row.get("rating")),
                "title": title,
                "snippet": snippet,
                "similarity": round(sim, 4),
                "explanations": {
                    "matching_sentences": best_overlapping_sentences(q.get("body", ""), body, top_n=2),
                    "keywords": keywords_hint(body)
                }
            })

        return out
