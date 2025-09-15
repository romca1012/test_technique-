from typing import List
import re
from rapidfuzz import fuzz

def split_sentences(text: str) -> List[str]:
    sents = re.split(r'(?<=[\.!\?])\s+', str(text).strip())
    return [s for s in sents if s]

def best_overlapping_sentences(q_body: str, cand_body: str, top_n: int = 2) -> List[str]:
    q_sents = split_sentences(q_body)
    c_sents = split_sentences(cand_body)
    scored = []
    for cs in c_sents:
        score = max((fuzz.token_set_ratio(cs, qs) for qs in q_sents), default=0)
        scored.append((score, cs))
    scored.sort(reverse=True)
    return [s for _, s in scored[:top_n]]

def keywords_hint(text: str, top_n: int = 5) -> List[str]:
    import collections
    t = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ0-9 ]", " ", str(text).lower())
    toks = [w for w in t.split() if len(w) > 2]
    c = collections.Counter(toks)
    return [w for w, _ in c.most_common(top_n)]