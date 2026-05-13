# backend/indexes.py
import os, json, faiss, numpy as np, logging
from collections import OrderedDict
from threading import Lock
from typing import List

logger = logging.getLogger(__name__)

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "data")

INDEX_PATH   = lambda vid: os.path.join(DATA, "indexes", f"{vid}.faiss")
META_PATH    = lambda vid: os.path.join(DATA, "indexes", f"{vid}.json")
VINDEX_PATH  = lambda vid: os.path.join(DATA, "indexes", f"{vid}.vfaiss")
ACLIP_PATH   = lambda vid: os.path.join(DATA, "indexes", f"{vid}.aclip.faiss")
SVINDEX_PATH = lambda vid: os.path.join(DATA, "indexes", f"{vid}.svfaiss")
SACLIP_PATH  = lambda vid: os.path.join(DATA, "indexes", f"{vid}.saclip.faiss")

# Thread-safe LRU cache for FAISS index objects
_cache: OrderedDict = OrderedDict()
_CACHE_MAX = 50
_lock = Lock()


def _cache_get(key):
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
        return None


def _cache_put(key, val):
    with _lock:
        _cache[key] = val
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)


def evict_video(video_id: str) -> None:
    with _lock:
        for k in [k for k in _cache if k[0] == video_id]:
            del _cache[k]


def _save_ip_index(path: str, embeddings: np.ndarray):
    d = embeddings.shape[1]
    if os.path.exists(path):
        try:
            if faiss.read_index(path).d != d:
                os.remove(path)
        except Exception:
            os.remove(path)
    index = faiss.IndexFlatIP(d)
    vecs = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    index.add(vecs.astype("float32"))
    faiss.write_index(index, path)


# --- Text index ---
def save_index(video_id: str, embeddings: np.ndarray, chunks: list):
    from db import db
    _save_ip_index(INDEX_PATH(video_id), embeddings)
    with open(META_PATH(video_id), "w") as f:
        json.dump({"video_id": video_id, "n": len(chunks)}, f)
    with db() as conn:
        conn.executemany(
            "INSERT INTO chunks(video_id, idx, start, end, text) VALUES(?,?,?,?,?)",
            [(video_id, i, c["start"], c["end"], c["text"]) for i, c in enumerate(chunks)],
        )
        conn.commit()
    _cache_put((video_id, "text"), faiss.read_index(INDEX_PATH(video_id)))


def load_index(video_id: str):
    from db import db
    key = (video_id, "text")
    index = _cache_get(key)
    if index is None:
        index = faiss.read_index(INDEX_PATH(video_id))
        _cache_put(key, index)
    with db() as conn:
        rows = conn.execute(
            "SELECT idx, start, end, text FROM chunks WHERE video_id=? ORDER BY idx",
            (video_id,),
        ).fetchall()
    return index, rows


# --- Visual metadata (no caption FAISS) ---
def save_visual_metadata(video_id: str, chunks: list):
    from db import db
    with db() as conn:
        conn.executemany(
            "INSERT INTO visual_chunks(video_id,idx,start,end,frame,objects,caption) VALUES(?,?,?,?,?,?,?)",
            [(video_id, i, c["start"], c["end"], c["frame"],
              json.dumps(c.get("objects", [])), c.get("caption", ""))
             for i, c in enumerate(chunks)],
        )
        conn.commit()


# --- Action clip metadata ---
def save_action_clips_metadata(video_id: str, rows: list):
    from db import db
    with db() as conn:
        conn.executemany(
            "INSERT INTO visual_clips(video_id,idx,start,end,objects,caption) VALUES(?,?,?,?,?,?)",
            [(video_id, i, r["start"], r["end"],
              json.dumps(r.get("objects", [])), r.get("caption", ""))
             for i, r in enumerate(rows)],
        )
        conn.commit()


# --- SigLIP visual index ---
def save_siglip_visual_index(video_id: str, embeddings: np.ndarray):
    _save_ip_index(SVINDEX_PATH(video_id), embeddings)
    _cache_put((video_id, "svfaiss"), faiss.read_index(SVINDEX_PATH(video_id)))


def load_siglip_visual_index(video_id: str):
    from db import db
    key = (video_id, "svfaiss")
    index = _cache_get(key)
    if index is None:
        index = faiss.read_index(SVINDEX_PATH(video_id))
        _cache_put(key, index)
    with db() as conn:
        rows = conn.execute(
            "SELECT idx,start,end,frame,objects,caption FROM visual_chunks WHERE video_id=? ORDER BY idx",
            (video_id,),
        ).fetchall()
    return index, rows


# --- SigLIP action clips index ---
def save_siglip_action_clips_index(video_id: str, embeddings: np.ndarray):
    _save_ip_index(SACLIP_PATH(video_id), embeddings)
    _cache_put((video_id, "saclip"), faiss.read_index(SACLIP_PATH(video_id)))


def load_siglip_action_clips_index(video_id: str):
    from db import db
    key = (video_id, "saclip")
    index = _cache_get(key)
    if index is None:
        index = faiss.read_index(SACLIP_PATH(video_id))
        _cache_put(key, index)
    with db() as conn:
        rows = conn.execute(
            "SELECT idx,start,end,objects,caption FROM visual_clips WHERE video_id=? ORDER BY idx",
            (video_id,),
        ).fetchall()
    return index, rows


# --- Legacy visual index (caption-based, still referenced by store.py re-exports) ---
def save_visual_index(video_id: str, embeddings: np.ndarray, chunks: list):
    from db import db
    _save_ip_index(VINDEX_PATH(video_id), embeddings)
    with db() as conn:
        conn.executemany(
            "INSERT INTO visual_chunks(video_id,idx,start,end,frame,objects,caption) VALUES(?,?,?,?,?,?,?)",
            [(video_id, i, c["start"], c["end"], c["frame"],
              json.dumps(c.get("objects", [])), c.get("caption", ""))
             for i, c in enumerate(chunks)],
        )
        conn.commit()


def load_visual_index(video_id: str):
    from db import db
    key = (video_id, "vfaiss")
    index = _cache_get(key)
    if index is None:
        index = faiss.read_index(VINDEX_PATH(video_id))
        _cache_put(key, index)
    with db() as conn:
        rows = conn.execute(
            "SELECT idx,start,end,frame,objects,caption FROM visual_chunks WHERE video_id=? ORDER BY idx",
            (video_id,),
        ).fetchall()
    return index, rows


def save_action_clips_index(video_id: str, embeddings: np.ndarray, rows: list):
    from db import db
    _save_ip_index(ACLIP_PATH(video_id), embeddings)
    with db() as conn:
        conn.executemany(
            "INSERT INTO visual_clips(video_id,idx,start,end,objects,caption) VALUES(?,?,?,?,?,?)",
            [(video_id, i, r["start"], r["end"],
              json.dumps(r.get("objects", [])), r.get("caption", ""))
             for i, r in enumerate(rows)],
        )
        conn.commit()


def load_action_clips_index(video_id: str):
    from db import db
    key = (video_id, "aclip")
    index = _cache_get(key)
    if index is None:
        index = faiss.read_index(ACLIP_PATH(video_id))
        _cache_put(key, index)
    with db() as conn:
        rows = conn.execute(
            "SELECT idx,start,end,objects,caption FROM visual_clips WHERE video_id=? ORDER BY idx",
            (video_id,),
        ).fetchall()
    return index, rows
