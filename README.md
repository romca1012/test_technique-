# Similar Reviews (same movie) — SensCritique (Demo)

Implémentation **Python** d'un système de recommandations de critiques **intra-film**, avec
**embeddings sémantiques** (sentence-transformers) + **rerank lexical** (TF‑IDF).

## Données
Le corpus combiné est dans `data/sample_reviews.csv` avec les colonnes :
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

## Tests
```bash
pytest -q
```

## Utiliser 2 fichiers CSV séparés (sans fusionner)
Par défaut, la configuration lit **deux fichiers** placés dans `data/` :
- `data/fightclub_critiques.csv` → `movie_id=FC`, `movie_title=Fight Club`
- `data/interstellar_critiques.csv` → `movie_id=INT`, `movie_title=Interstellar`

Vous pouvez modifier ces chemins/identifiants dans `src/config.py` (section `sources`).  
Le système charge et **normalise** ces fichiers au démarrage, en mémoire, sans créer de fichier combiné.