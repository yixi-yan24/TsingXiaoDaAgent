import os, json, pickle
import numpy as np
from typing import Optional

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

MODEL_NAME = "shibing624/text2vec-base-chinese"
EMBEDDING_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EMBEDDING_CACHE_FILE = os.path.join(EMBEDDING_CACHE_DIR, "embeddings.pkl")
EMBEDDING_META_FILE = os.path.join(EMBEDDING_CACHE_DIR, "embeddings_meta.json")

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(MODEL_NAME)
    return _model


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def compute_embeddings(texts: list[str]) -> np.ndarray:
    """Compute embeddings for a list of texts."""
    model = _load_model()
    return model.encode(texts, show_progress_bar=True, normalize_embeddings=True)


def build_index(minors: list) -> tuple[np.ndarray, list[str]]:
    """Build embedding index from minor programs.

    Returns:
        embeddings: (N, D) numpy array, normalized
        texts: list of N search texts
    """
    texts = []
    for m in minors:
        combined = (
            f"{m.department} {m.name} "
            f"{m.major_restrictions[:200]} "
            f"{m.prerequisites[:200]} "
        )
        texts.append(combined)
    embeddings = compute_embeddings(texts)
    return embeddings, texts


def save_index(embeddings: np.ndarray, texts: list[str], minor_ids: list[str]):
    """Save embedding index to disk."""
    os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
    with open(EMBEDDING_CACHE_FILE, "wb") as f:
        pickle.dump({"embeddings": embeddings, "texts": texts, "minor_ids": minor_ids}, f)
    with open(EMBEDDING_META_FILE, "w", encoding="utf-8") as f:
        json.dump({"texts": texts, "minor_ids": minor_ids}, f, ensure_ascii=False, indent=2)


def load_index() -> Optional[tuple[np.ndarray, list[str], list[str]]]:
    """Load saved embedding index."""
    if not os.path.exists(EMBEDDING_CACHE_FILE):
        return None
    try:
        with open(EMBEDDING_CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        return data["embeddings"], data["texts"], data["minor_ids"]
    except Exception:
        return None


def get_or_build_index(minors: list, force_rebuild: bool = False
                       ) -> tuple[np.ndarray, list[str], list[str]]:
    """Get cached index or build a new one."""
    if not force_rebuild:
        cached = load_index()
        if cached is not None and len(cached[2]) == len(minors):
            return cached

    minor_ids = [m.name for m in minors]
    embeddings, texts = build_index(minors)
    try:
        save_index(embeddings, texts, minor_ids)
    except Exception:
        pass
    return embeddings, texts, minor_ids


def semantic_search(query: str, minors: list, top_k: int = 5) -> list[tuple]:
    """Search minors by semantic similarity to query.

    Returns list of (minor, score) tuples, sorted by relevance descending.
    """
    model = _load_model()
    embeddings, texts, minor_ids = get_or_build_index(minors)
    query_emb = model.encode([query], normalize_embeddings=True)[0]

    scores = [float(np.dot(query_emb, emb)) for emb in embeddings]
    ranked = sorted(zip(minors, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
