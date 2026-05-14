from sentence_transformers import SentenceTransformer

_EMB = None

def get_emb() -> SentenceTransformer:
    global _EMB
    if _EMB is None:
        _EMB = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _EMB
