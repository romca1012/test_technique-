# Similar Reviews (same movie) — SensCritique (Demo)

# TEST

Implémentation **Python** d'un système de recommandations de critiques **intra-film**, avec
**embeddings sémantiques** (sentence-transformers) + **rerank lexical** (TF‑IDF).

## Données
La donnée doit etre ajouté dans un dossier data a la racine du repository`data/` :
```
review_id,movie_id,movie_title,user_id,rating,title,body,created_at,lang
```

## Installation & exécution
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn src.api:app --reload --port 8000
# → http://127.0.0.1:8000/similar?review_id=<ID>&k=3
```

## Utiliser 2 fichiers CSV séparés (sans fusionner)
Par défaut, la configuration lit **deux fichiers** placés dans `data/` :
- `data/fightclub_critiques.csv` → `movie_id=FC`, `movie_title=Fight Club`
- `data/interstellar_critiques.csv` → `movie_id=INT`, `movie_title=Interstellar`

Vous pouvez modifier ces chemins/identifiants dans `src/config.py` (section `sources`).  
Le système charge et **normalise** ces fichiers au démarrage, en mémoire, sans créer de fichier combiné.



## 1) Objectif & contraintes

* **Objectif** : proposer, pour une critique lue, d’autres **critiques du même film** au contenu **sémantiquement proche**.
* **Contraintes** :

  * Temps de réponse court (P95 < 300 ms sur \~quelques milliers de critiques / film).
  * Données en CSV hétérogènes (noms de colonnes variés) → **normalisation**.
  * **Explicabilité** : phrases proches & mots-clés.

---

## 2) Vue d’ensemble (composants)

```mermaid
flowchart LR
  subgraph Ingestion_Preparation
    A[CSV fightclub_critiques.csv]:::file
    B[CSV interstellar_critiques.csv]:::file
    C[repository.py read and normalize]:::comp
    D[preprocessing.py normalize_text and make_corpus_row]:::comp
  end

  subgraph Similarity_Engine
    E[embedder.py SentenceTransformers]:::comp
    F[indexer.py kNN cosine]:::comp
    G[hybrid_ranker.py TFIDF unigrams bigrams cosine]:::comp
    H[explainer.py RapidFuzz keywords]:::comp
  end

  I[FastAPI endpoint similar]:::api

  A --> C
  B --> C
  C --> D --> E --> F
  D --> G
  I -->|review_id and k| F
  F -->|same movie pool| G
  G --> H --> I

  classDef comp fill:#eef6ff,stroke:#5b8def,color:#0b3b8c;
  classDef api fill:#e8fff3,stroke:#21a47a,color:#0d5c43;
  classDef file fill:#fff7e6,stroke:#d5a54a,color:#7a4f08;
  ```


## 3) Choix techniques (et pourquoi)

* **Python + FastAPI (Uvicorn)** : écosystème ML riche, FastAPI performant, typing & validation, docs auto.
* **Sentence-Transformers – paraphrase-multilingual-MiniLM-L12-v2** :

  * Multilingue FR/EN, **rapide** (dim. ≈ 384), bon compromis qualité/latence.
  * `normalize_embeddings=True` → cosine = dot product.
* **scikit-learn NearestNeighbors(metric='cosine', algorithm='brute')** :

  * Simple & fiable pour **quelques milliers** d’items.
  * Filtrage métier **même film** appliqué après la recherche (pool > k) pour assurer du rappel.
* **TF-IDF (1,2-gram)** :

  * Capte la proximité **lexicale** (mots/bigrammes), complémentaire à la sémantique.
* **RapidFuzz** : explications rapides et robustes (token\_set\_ratio).
* **pandas** : ingestion & nettoyage tabulaire.

**Trade-offs**

* *Brute force kNN* : OK au départ ; pour des millions d’items, passer à **FAISS/HNSW**.
* *Embeddings à chaud* : calculés au démarrage. À l’échelle → **pré-calcul** & persistance.

---

## 4) Flux de requête (/similar)

```mermaid
sequenceDiagram
  participant Client
  participant API as FastAPI
  participant Rec as ReviewRecommender
  participant Idx as kNN Index
  participant Rank as HybridRanker
  participant Exp as Explainer

  Client->>API: GET /similar?review_id=R&k=K
  API->>Rec: similar(R,K)
  Rec->>Rec: by_id(R) + q_text + q_emb
  Rec->>Idx: query_same_movie(q_emb, movie_id, pool=50)
  Idx-->>Rec: cand_ids + sem_sims (même film)
  Rec->>Rank: score(q_text, cand_indices, sem_sims, α)
  Rank-->>Rec: order + scores
  Rec->>Exp: matching_sentences + keywords
  Exp-->>Rec: explanations
  Rec-->>API: top-K (≥ min_sim)
  API-->>Client: JSON results
```

## 5) Contrat d’API (extrait)

`GET /similar?review_id=<str>&k=<int(1..20)>`

**200**

```json
{
  "review_id": "FC_20761",
  "top_k": 5,
  "results": [
    {
      "review_id": "FC_...",
      "movie_id": "FC",
      "movie_title": "Fight Club",
      "user_id": "...",
      "rating": 4.5,
      "title": "...",
      "snippet": "...",
      "similarity": 0.8123,
      "explanations": {
        "matching_sentences": ["...","..."],
        "keywords": ["violence","bagarre", "..."]
      }
    }
  ]
}
```

**404** : `review_id not found` • **400** : corpus vide/erreur de chargement.

---

## 6) Pourquoi cette pile ? (en une ligne chacun)

* **FastAPI** : perfs + typing + DX excellente.
* **Sentence‑Transformers (MiniLM)** : sémantique multilingue rapide.
* **kNN cosine** : simplicité & robustesse au volume initial.
* **TF‑IDF** : précision lexicale complémentaire.
* **RapidFuzz** : explications légères et utiles au produit.

---

## 7) Roadmap courte

1. Artefacts persistés (embeddings, index) pour démarrages rapides.
2. FAISS/HNSW quand le volume augmente.
3. Rerank enrichi (signaux produit) + A/B tests.
4. Feature flags (seuils, alpha) et dashboard de monitoring.




L’IA m’a aidé à structurer l’explication d’architecture et à relire la documentation. Le code Python et les choix finaux ont été réalisés et validés par moi à l'exception de l'hybrid ranker qui allie proximité sémantique et lexicale et de l organisation de l appel des fonctions clés dans le fichier recommender.py .
