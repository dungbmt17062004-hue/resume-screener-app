"""Hybrid matcher: 60% semantic cosine + 40% technical skill overlap."""
import difflib, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from core.taxonomy import ALL_SKILLS

W_SEM, W_SKILL = 0.6, 0.4
ALIASES = {"nodejs": "node.js", "node": "node.js", "k8s": "kubernetes", "sklearn": "scikit-learn", "js": "javascript",
           "ts": "typescript", "postgres": "postgresql", "ml": "machine learning", "cicd": "ci/cd", "tf": "tensorflow"}
SINGLE = [s for s in ALL_SKILLS if " " not in s]

class Matcher:
    def __init__(self):
        self._model, self.backend = None, "tfidf-fallback"
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self.backend = "all-MiniLM-L6-v2"
        except Exception:
            pass

    def semantic(self, a: str, b: str) -> float:
        if self._model is not None:
            ea, eb = self._model.encode([a, b], normalize_embeddings=True)
            return float(np.clip(np.dot(ea, eb), 0, 1))
        m = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), stop_words="english").fit_transform([a, b])
        return float(np.clip((m[0].multiply(m[1])).sum(), 0, 1))

    @staticmethod
    def extract_skills(text: str) -> set:
        low = text.lower()
        found = {s for s in ALL_SKILLS if re.search(rf"(?<![\w.]){re.escape(s)}(?![\w])", low)}
        for t in set(re.findall(r"[a-z][a-z0-9+#.\-/]+", low)):
            if t in ALIASES:
                found.add(ALIASES[t])
            elif len(t) >= 6 and t not in found:  # fuzzy cho lỗi chính tả, vd "kubernetis"
                hit = difflib.get_close_matches(t, SINGLE, n=1, cutoff=0.88)
                if hit:
                    found.add(hit[0])
        return found

    def match(self, resume: str, jd: str, probs: dict | None = None, jd_domain: str | None = None) -> dict:
        sem = self.semantic(resume, jd)
        r, j = self.extract_skills(resume), self.extract_skills(jd)
        overlap = len(r & j) / len(j) if j else sem
        hybrid = W_SEM * sem + W_SKILL * overlap
        fit = (probs or {}).get(jd_domain, 0.0) if jd_domain else 0.0
        return {"hybrid_score": round(hybrid * 100, 1), "semantic": round(sem * 100, 1),
                "skill_overlap": round(overlap * 100, 1), "track_fit": round(fit * 100, 1),
                "matched": sorted(r & j), "missing": sorted(j - r), "extra": sorted(r - j)[:15],
                "backend": self.backend}
