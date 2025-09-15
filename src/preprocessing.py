import re
from typing import Tuple

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if t.lower() in {"none", "null", "nan", ""}:
        return ""
    t = t.lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[«»“”]", '"', t)

    if not re.search(r"[a-z0-9àâçéèêëîïôûùüÿñæœ]", t):
        return ""
    return t.strip()

def make_corpus_row(title: str, body: str) -> Tuple[str, str]:
    title = title or ""
    body = body or ""
    return f"{title}. {body}".strip(), body
