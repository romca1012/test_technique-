from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class SourceSpec:
    path: str
    movie_id: str
    movie_title: str

@dataclass
class Settings:
    data_path: Optional[str] = None
    sources: List[SourceSpec] = field(default_factory=lambda: [
        SourceSpec(path="data/fightclub_critiques.csv", movie_id="FC",  movie_title="Fight Club"),
        SourceSpec(path="data/interstellar_critiques.csv", movie_id="INT", movie_title="Interstellar"),
    ])
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    top_k: int = 5
    min_sim: float = 0.25

settings = Settings()