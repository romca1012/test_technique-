from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from .config import settings, SourceSpec


def _read_csv_smart(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8", on_bad_lines="skip")
    except TypeError:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding="utf-8", error_bad_lines=False)
        except Exception:
            pass
    except Exception:
        pass

    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="latin-1", on_bad_lines="skip")
    except TypeError:
        return pd.read_csv(path, sep=None, engine="python", encoding="latin-1", error_bad_lines=False)


def _pick_column(cols_map: dict, *cands: str, default=None) -> pd.Series:
    for c in cands:
        if c in cols_map:
            return cols_map[c]
    n = len(next(iter(cols_map.values()))) if cols_map else 0
    return pd.Series([default] * n)


def _normalize(df: pd.DataFrame, movie_id: str, movie_title: str) -> Tuple[pd.DataFrame, dict]:
    # map noms en minuscule
    cols_map = {c.lower(): df[c] for c in df.columns}
    # IDs
    id_alias = ("review_id", "id", "reviewid")
    # Utilisateur
    user_alias = ("user_id", "username", "user", "author", "auteur", "uid")
    # Note
    rating_alias = ("rating", "note", "score")
    # Titre : inclut 'review_title'
    title_alias = ("title", "titre", "headline", "subject", "review_title")
    # Corps : inclut 'review_content'
    body_alias = ("body", "critique", "texte", "content", "review", "text", "comment", "review_content")
    # Dates
    created_alias = ("created_at", "date", "datetime", "timestamp", "created", "review_date_creation")
    # Langue
    lang_alias = ("lang", "language", "langue")

    out = pd.DataFrame({
        "review_id":   _pick_column(cols_map, *id_alias),
        "movie_id":    movie_id,
        "movie_title": movie_title,
        "user_id":     _pick_column(cols_map, *user_alias),
        "rating":      _pick_column(cols_map, *rating_alias),
        "title":       _pick_column(cols_map, *title_alias),
        "body":        _pick_column(cols_map, *body_alias),
        "created_at":  _pick_column(cols_map, *created_alias),
        "lang":        _pick_column(cols_map, *lang_alias, default="fr"),
    })

    for col in ["review_id", "user_id", "title", "body"]:
        out[col] = out[col].astype(str).fillna("").str.strip()
        mask_placeholder = out[col].str.lower().isin(["none", "null", "nan"])
        out.loc[mask_placeholder, col] = ""

    before = len(out)
    mask = (out["title"].str.len() + out["body"].str.len()) >= 3
    out = out[mask].reset_index(drop=True)
    after = len(out)

    diags = {
        "rows_in": before,
        "rows_kept": after,
        "had_title_col": any(a in cols_map for a in title_alias),
        "had_body_col": any(a in cols_map for a in body_alias),
        "columns": list(df.columns),
    }

    return out, diags


class ReviewRepo:
    def __init__(self, path: Optional[str] = None, sources: Optional[List[SourceSpec]] = None):
        if path:
            raw = _read_csv_smart(path)
            df_norm, diags = _normalize(raw, movie_id="UNK", movie_title="Unknown")
            if diags["rows_kept"] == 0:
                raise RuntimeError(
                    f"Aucune ligne gardée pour {path}. Colonnes détectées: {diags['columns']}. "
                    f"Présence title? {diags['had_title_col']}, body? {diags['had_body_col']}."
                )
            df_all = df_norm
        else:
            srcs = sources if sources is not None else settings.sources
            parts = []
            problems = []
            for spec in srcs:
                raw = _read_csv_smart(spec.path)
                df_norm, diags = _normalize(raw, movie_id=spec.movie_id, movie_title=spec.movie_title)
                if diags["rows_kept"] == 0:
                    problems.append(
                        f"- {spec.path}: 0/ {diags['rows_in']} lignes gardées. "
                        f"Colonnes: {diags['columns']}. title? {diags['had_title_col']}, body? {diags['had_body_col']}"
                    )
                else:
                    parts.append(df_norm)

            if not parts:
                details = "\n".join(problems) if problems else "(aucune source lisible)"
                raise RuntimeError(
                    "Aucune critique exploitable après normalisation de toutes les sources.\n" + details
                )

            df_all = pd.concat(parts, ignore_index=True)

        if not df_all.empty and df_all["review_id"].duplicated().any():
            df_all["review_id"] = df_all.apply(
                lambda r: f"{r['movie_id']}_{r['review_id']}", axis=1
            )

        self.df = df_all

    def all(self) -> pd.DataFrame:
        return self.df

    def by_id(self, review_id: str) -> Dict[str, Any]:
        row = self.df[self.df["review_id"].astype(str) == str(review_id)]
        if row.empty:
            raise KeyError(f"review_id {review_id} not found")
        return row.iloc[0].to_dict()
