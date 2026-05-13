# Video Semantic — Backend & Frontend Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor monolithic backend into focused modules, add async ingestion with progress polling, surface video titles, add search history and object filter suggestions, and normalize score display.

**Architecture:** Backend splits into `db.py`, `indexes.py`, `context.py`, `embeddings.py`, and three router files; `store.py` becomes thin re-exports so `ingest.py`/`visual_ingest.py` need no changes. Async ingestion uses `ThreadPoolExecutor(max_workers=1)` with an in-process `_jobs` dict polled by the frontend every 2 s. All new API fields are additive — no breaking changes.

**Tech Stack:** Python 3.10+ / FastAPI / FAISS / SQLite / SentenceTransformers / yt-dlp Python API · React 18 / TypeScript / Vite / TailwindCSS

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `backend/embeddings.py` | Lazy BGE-small singleton |
| Create | `backend/db.py` | `init_db`, `db()`, `clear_video`, `store_video_meta`, caption cache helpers |
| Create | `backend/indexes.py` | FAISS LRU cache + all save/load functions; exports `DATA` |
| Create | `backend/context.py` | `build_video_context`, `filter_videos_by_context`, topic derivation |
| Modify | `backend/store.py` | Thin re-exports only (no logic) |
| Modify | `backend/models.py` | Add `ScoreRange`, `IngestJobResponse`, `JobStatusResponse`; add `score_range` to `UnifiedSearchResponse` |
| Create | `backend/routers/__init__.py` | Empty package marker |
| Create | `backend/routers/videos.py` | `GET /videos`, `DELETE /videos/{id}`, `POST /ov_verify` |
| Create | `backend/routers/ingest.py` | `POST /ingest` (async), `GET /ingest/status/{id}`, `POST /build_contexts` |
| Create | `backend/routers/search.py` | `POST /query` (+ `score_range`), legacy GET endpoints |
| Modify | `backend/app.py` | Startup event → `init_db()`, include three routers, remove inline logic |
| Modify | `frontend/src/lib/api.ts` | New types + `ingestStatus()` method |
| Create | `frontend/src/hooks/useSearchHistory.ts` | localStorage search history hook |
| Modify | `frontend/src/components/IngestModal.tsx` | Polling progress with stage checklist |
| Modify | `frontend/src/components/ResultCard.tsx` | `title` prop, relative score % |
| Modify | `frontend/src/components/FilterPanel.tsx` | `<datalist>` for object suggestions |
| Modify | `frontend/src/pages/SearchPage.tsx` | `videoTitles` map, `score_range`, history chips |
| Modify | `frontend/src/components/VideoLibrary.tsx` | Title column, source URL link, Re-index button |

---

## Task 1: Create `backend/embeddings.py`

**Files:** Create `backend/embeddings.py`

- [ ] **Create the file**

```python
# backend/embeddings.py
from sentence_transformers import SentenceTransformer

_EMB = None

def get_emb() -> SentenceTransformer:
    global _EMB
    if _EMB is None:
        _EMB = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _EMB
```

- [ ] **Verify import works**

```bash
cd backend && python -c "from embeddings import get_emb; print('ok')"
```
Expected: `ok`

- [ ] **Commit**

```bash
git add backend/embeddings.py
git commit -m "refactor: add lazy BGE embedding singleton"
```

---

## Task 2: Create `backend/db.py`

**Files:** Create `backend/db.py`

- [ ] **Create the file**

```python
# backend/db.py
import os, sqlite3, json, shutil, logging
from contextlib import contextmanager
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "data")
DB_PATH = os.path.join(DATA, "indexes", "meta.sqlite")


def init_db() -> None:
    """Run once at app startup — creates tables, migrations, and column indexes."""
    os.makedirs(os.path.join(DATA, "indexes"), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks(
            video_id TEXT, idx INTEGER, start REAL, end REAL, text TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS visual_chunks(
            video_id TEXT, idx INTEGER, start REAL, end REAL,
            frame TEXT, objects TEXT, caption TEXT,
            UNIQUE(video_id, idx) ON CONFLICT REPLACE
        )
    """)
    try:
        conn.execute("ALTER TABLE visual_chunks ADD COLUMN caption TEXT")
    except Exception:
        pass

    conn.execute("""
        CREATE TABLE IF NOT EXISTS visual_clips(
            video_id TEXT, idx INTEGER, start REAL, end REAL,
            objects TEXT, caption TEXT,
            UNIQUE(video_id, idx) ON CONFLICT REPLACE
        )
    """)
    try:
        conn.execute("ALTER TABLE visual_clips ADD COLUMN caption TEXT")
    except Exception:
        pass

    conn.execute("""
        CREATE TABLE IF NOT EXISTS caption_cache(
            video_id TEXT, frame TEXT, caption TEXT, objects TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(video_id, frame)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS video_context(
            video_id TEXT PRIMARY KEY, title TEXT, source_url TEXT,
            summary TEXT, topics TEXT, objects_topk TEXT, actions_topk TEXT,
            lang TEXT, emb BLOB, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Performance indexes (idempotent)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_video ON chunks(video_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_visual_chunks_video ON visual_chunks(video_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_visual_clips_video ON visual_clips(video_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_caption_cache_video ON caption_cache(video_id)")
    conn.commit()
    conn.close()


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
    finally:
        conn.close()


def clear_video(video_id: str) -> dict:
    with db() as conn:
        for table in ("chunks", "visual_chunks", "visual_clips", "video_context", "caption_cache"):
            conn.execute(f"DELETE FROM {table} WHERE video_id=?", (video_id,))
        conn.commit()

    indexes_dir = os.path.join(DATA, "indexes")
    media_dir   = os.path.join(DATA, "media")
    frames_dir  = os.path.join(DATA, "frames", video_id)
    exts = [".faiss", ".json", ".vfaiss", ".aclip.faiss", ".svfaiss", ".saclip.faiss"]
    files = [os.path.join(indexes_dir, f"{video_id}{e}") for e in exts]
    files += [os.path.join(media_dir, f"{video_id}.mp4"),
              os.path.join(media_dir, f"{video_id}.wav")]

    deleted, failed = [], []
    for p in files:
        try:
            os.remove(p); deleted.append(p)
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("Failed to remove %s", p, exc_info=True); failed.append(p)
    if os.path.isdir(frames_dir):
        try:
            shutil.rmtree(frames_dir); deleted.append(frames_dir)
        except OSError:
            logger.warning("Failed to remove frames dir %s", frames_dir, exc_info=True)
            failed.append(frames_dir)
    return {"deleted": deleted, "failed": failed}


def store_video_meta(video_id: str, title: Optional[str], source_url: Optional[str]) -> None:
    with db() as conn:
        conn.execute(
            "UPDATE video_context SET title=?, source_url=?, updated_at=CURRENT_TIMESTAMP WHERE video_id=?",
            (title, source_url, video_id),
        )
        conn.commit()


def get_cached_captions(video_id: str, frames: List[str]) -> Dict:
    if not frames:
        return {}
    qmarks = ",".join(["?"] * len(frames))
    with db() as conn:
        rows = conn.execute(
            f"SELECT frame, caption, objects FROM caption_cache WHERE video_id=? AND frame IN ({qmarks})",
            (video_id, *frames),
        ).fetchall()
    out = {}
    for frame, caption, objects in rows:
        try:
            objs = json.loads(objects) if objects else []
        except (json.JSONDecodeError, TypeError):
            objs = []
        out[frame] = {"caption": caption or "", "objects": objs}
    return out


def put_cached_captions(video_id: str, entries: Dict) -> None:
    if not entries:
        return
    with db() as conn:
        conn.executemany(
            """INSERT INTO caption_cache(video_id, frame, caption, objects, updated_at)
               VALUES(?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(video_id, frame) DO UPDATE SET
                 caption=excluded.caption, objects=excluded.objects,
                 updated_at=CURRENT_TIMESTAMP""",
            [(video_id, frame, e.get("caption", ""), json.dumps(e.get("objects", [])))
             for frame, e in entries.items()],
        )
        conn.commit()
```

- [ ] **Verify**

```bash
cd backend && python -c "from db import init_db, db; init_db(); print('init_db ok')"
```
Expected: `init_db ok`

- [ ] **Commit**

```bash
git add backend/db.py
git commit -m "refactor: extract db.py with init_db, clear_video, caption cache helpers"
```

---

## Task 3: Create `backend/indexes.py`

**Files:** Create `backend/indexes.py`

- [ ] **Create the file**

```python
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
```

- [ ] **Verify**

```bash
cd backend && python -c "from indexes import evict_video, DATA; print('indexes ok', DATA)"
```
Expected: `indexes ok /path/to/backend/data`

- [ ] **Commit**

```bash
git add backend/indexes.py
git commit -m "refactor: extract indexes.py with FAISS LRU cache"
```

---

## Task 4: Create `backend/context.py`

**Files:** Create `backend/context.py`

- [ ] **Create the file** — this is the content moved from `store.py` with updated imports:

```python
# backend/context.py
import json, re, logging
import numpy as np
from collections import Counter
from typing import Optional, List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

_nlp = None
def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["ner"])
    return _nlp


def derive_text_summary_for(video_id: str, conn) -> str:
    rows = conn.execute(
        "SELECT text FROM chunks WHERE video_id=? ORDER BY start LIMIT 5", (video_id,)
    ).fetchall()
    if not rows:
        return ""
    texts = [r[0] for r in rows if r[0] and r[0].strip()]
    combined = " ".join(texts)
    return combined[:200] + "..." if len(combined) > 200 else combined


def top_objects_for(video_id: str, conn, k: int = 20) -> Dict[str, int]:
    rows = conn.execute(
        "SELECT objects FROM visual_clips WHERE video_id=?", (video_id,)
    ).fetchall()
    counts: Counter = Counter()
    for (obj_json,) in rows:
        if obj_json:
            try:
                for o in json.loads(obj_json):
                    counts[o] += 1
            except (json.JSONDecodeError, TypeError):
                pass
    return dict(counts.most_common(k))


def top_actions_for(video_id: str, conn, k: int = 20) -> Dict[str, int]:
    rows = conn.execute(
        "SELECT text FROM chunks WHERE video_id=?", (video_id,)
    ).fetchall()
    nlp = _get_nlp()
    verb_counts: Counter = Counter()
    vo_counts: Counter = Counter()
    for (txt,) in rows:
        if not txt:
            continue
        doc = nlp(txt)
        for tok in doc:
            if tok.pos_ == "VERB" and tok.lemma_ not in ("be", "have", "do"):
                v = tok.lemma_.lower()
                verb_counts[v] += 1
                for child in tok.children:
                    if child.dep_ in ("dobj", "obj") and child.pos_ in ("PROPN", "NOUN", "PRON"):
                        vo_counts[f"{v} {child.lemma_.lower()}"] += 1
                        break
    out: Dict[str, int] = {}
    for key, count in vo_counts.most_common(k):
        out[key] = count
    if len(out) < k:
        for verb, count in verb_counts.most_common(k - len(out)):
            out[verb] = count
    return out


def _fetch_texts_for_video(video_id: str, conn, max_chunks: int = 300) -> List[str]:
    rows = conn.execute(
        "SELECT text FROM chunks WHERE video_id=? ORDER BY start LIMIT ?",
        (video_id, max_chunks),
    ).fetchall()
    texts = [t for (t,) in rows if t and t.strip()]
    return [re.sub(r"\s+", " ", t).strip() for t in texts]


def derive_topics(summary: str, objects_topk: dict, actions_topk: dict,
                  texts: list = None, topn: int = 10) -> List[str]:
    texts = texts or []
    corpus = [t for t in texts if t] + ([summary] if summary else [])
    topics: set = set(list(objects_topk.keys())[:5]) | set(list(actions_topk.keys())[:5])
    if corpus:
        vec = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=1000,
            stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        )
        X = vec.fit_transform(corpus)
        scores = X.sum(axis=0).A1
        vocab = np.array(vec.get_feature_names_out())
        for term in vocab[scores.argsort()[::-1]]:
            if term in topics or re.match(r"^\d+$", term):
                continue
            topics.add(term)
            if len(topics) >= topn + 10:
                break
    return list(topics)


def derive_topics_bertopic(texts: List[str], topn: int = 10,
                           min_topic_size: int = 5) -> List[str]:
    from bertopic import BERTopic
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return []
    topic_model = BERTopic(
        min_topic_size=min_topic_size, calculate_probabilities=False,
        verbose=False, embedding_model="BAAI/bge-small-en-v1.5",
    )
    topics, _ = topic_model.fit_transform(texts)
    info = topic_model.get_topic_info()
    out: List[str] = []
    for _, row in info.sort_values("Count", ascending=False).head(5).iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue
        for term, _ in (topic_model.get_topic(tid) or [])[:3]:
            out.append(term)
    final, seen = [], set()
    for t in out:
        t = t.strip().lower()
        if t and t not in seen:
            final.append(t); seen.add(t)
        if len(final) >= topn:
            break
    return final


def build_video_context(video_id: str) -> None:
    from db import db
    from embeddings import get_emb
    try:
        with db() as conn:
            summary = derive_text_summary_for(video_id, conn)
            objects_topk = top_objects_for(video_id, conn, k=20)
            actions_topk = top_actions_for(video_id, conn, k=20)
            texts = _fetch_texts_for_video(video_id, conn, max_chunks=300)
            topics = (derive_topics_bertopic(texts, topn=10, min_topic_size=5)
                      if len(texts) >= 10
                      else derive_topics(summary, objects_topk, actions_topk, texts, topn=10))
            fused: List[str] = []
            seen: set = set()
            for t in (list(objects_topk.keys())[:5] + list(actions_topk.keys())[:5] + topics):
                t0 = t.strip().lower()
                if t0 and t0 not in seen:
                    fused.append(t0); seen.add(t0)
            text_for_emb = summary or " ".join(fused)
            emb_blob = (get_emb().encode([text_for_emb], normalize_embeddings=True)
                        .astype("float32")[0].tobytes()
                        if text_for_emb.strip() else None)
            conn.execute("""
                INSERT INTO video_context(video_id,title,source_url,summary,topics,
                                          objects_topk,actions_topk,lang,emb,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
                ON CONFLICT(video_id) DO UPDATE SET
                  summary=excluded.summary, topics=excluded.topics,
                  objects_topk=excluded.objects_topk, actions_topk=excluded.actions_topk,
                  emb=excluded.emb, updated_at=CURRENT_TIMESTAMP
            """, (video_id, None, None, summary, json.dumps(fused),
                  json.dumps(objects_topk), json.dumps(actions_topk), "en", emb_blob))
            conn.commit()
        logger.info("Built context for %s: %d topics", video_id, len(fused))
    except Exception:
        logger.exception("Error building video context for %s", video_id)


def _fetch_contexts(video_ids: Optional[List[str]] = None) -> List[Dict]:
    from db import db
    with db() as conn:
        if video_ids:
            qmarks = ",".join(["?"] * len(video_ids))
            rows = conn.execute(
                f"SELECT video_id,topics,objects_topk,actions_topk,emb FROM video_context WHERE video_id IN ({qmarks})",
                video_ids,
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT video_id,topics,objects_topk,actions_topk,emb FROM video_context"
            ).fetchall()
    out = []
    for vid, topics, objs, acts, emb in rows:
        out.append({
            "video_id": vid,
            "topics": json.loads(topics or "[]"),
            "objects_topk": json.loads(objs or "{}"),
            "actions_topk": json.loads(acts or "{}"),
            "emb": np.frombuffer(emb, dtype=np.float32) if emb else None,
        })
    return out


def filter_videos_by_context(query: str, restrict_videos: Optional[List[str]] = None,
                              topn: int = 50, min_cos: float = 0.18) -> List[str]:
    from embeddings import get_emb
    qv = get_emb().encode([query], normalize_embeddings=True).astype("float32")[0]
    contexts = _fetch_contexts(restrict_videos)
    scored: List[Tuple[str, float]] = []
    for c in contexts:
        if c["emb"] is None:
            scored.append((c["video_id"], 0.1)); continue
        cos = float(np.dot(qv, c["emb"]))
        q_lower = query.lower()
        if any(t.lower() in q_lower for t in c["topics"]):
            cos += 0.05
        if any(o.lower() in q_lower for o in c["objects_topk"]):
            cos += 0.03
        if any(a.lower() in q_lower for a in c["actions_topk"]):
            cos += 0.03
        scored.append((c["video_id"], cos))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [vid for vid, s in scored if s >= min_cos][:topn]


def passes_hard_context(video_id: str, require_any: Optional[List[str]] = None,
                         require_all: Optional[List[str]] = None) -> bool:
    contexts = _fetch_contexts([video_id])
    if not contexts:
        return True
    c = contexts[0]
    bag = {w.lower() for w in [*c["topics"], *c["objects_topk"], *c["actions_topk"]]}
    if require_all and not all(w.lower() in bag for w in require_all):
        return False
    if require_any and not any(w.lower() in bag for w in require_any):
        return False
    return True
```

- [ ] **Verify**

```bash
cd backend && python -c "from context import filter_videos_by_context; print('context ok')"
```
Expected: `context ok`

- [ ] **Commit**

```bash
git add backend/context.py
git commit -m "refactor: extract context.py with build_video_context and filter helpers"
```

---

## Task 5: Slim `backend/store.py` to re-exports

**Files:** Modify `backend/store.py`

`ingest.py` imports `DATA, save_index` from `store`. `visual_ingest.py` imports `DATA, save_visual_metadata, save_action_clips_metadata, save_siglip_visual_index, save_siglip_action_clips_index`. `app.py` imports several more. Replace the entire file with re-exports so neither file needs touching.

- [ ] **Replace `store.py` entirely**

```python
# backend/store.py — re-exports for backward compatibility
# ingest.py and visual_ingest.py import from here; do not remove.
from indexes import (
    DATA,
    save_index, load_index,
    save_visual_index, load_visual_index,
    save_action_clips_index, load_action_clips_index,
    save_visual_metadata, save_action_clips_metadata,
    save_siglip_visual_index, load_siglip_visual_index,
    save_siglip_action_clips_index, load_siglip_action_clips_index,
    evict_video,
)
from db import (
    db,
    clear_video,
    get_cached_captions,
    put_cached_captions,
    store_video_meta,
)
from context import (
    build_video_context,
    filter_videos_by_context,
    passes_hard_context,
)
```

- [ ] **Verify both ingest modules still import cleanly**

```bash
cd backend && python -c "from store import DATA, save_index, save_visual_metadata; print('store ok')"
cd backend && python -c "import ingest; print('ingest ok')"
cd backend && python -c "import visual_ingest; print('visual_ingest ok')"
```
Expected: three `ok` lines (visual_ingest will load torch, allow it).

- [ ] **Commit**

```bash
git add backend/store.py
git commit -m "refactor: slim store.py to re-exports from db/indexes/context"
```

---

## Task 6: Update `backend/models.py`

**Files:** Modify `backend/models.py`

- [ ] **Add new models** — append after the existing `OVVerifyRequest` class:

```python
class ScoreRange(BaseModel):
    min: float
    max: float
```

- [ ] **Add `score_range` to `UnifiedSearchResponse`** — change the class to:

```python
class UnifiedSearchResponse(BaseModel):
    video_id: Optional[str] = None
    mode: MODE
    hits: List[UnifiedSearchHit] = Field(default_factory=list)
    info: dict = Field(default_factory=dict)
    score_range: Optional[ScoreRange] = None
```

- [ ] **Add ingest job models** — append after `ScoreRange`:

```python
class IngestJobResponse(BaseModel):
    job_id: Optional[str]
    video_id: str
    status: str   # "queued" | "already_exists"
    message: Optional[str] = None

class JobStatusResponse(BaseModel):
    job_id: str
    video_id: str
    status: str   # "queued" | "running" | "done" | "error"
    stage: str
    error: Optional[str] = None
```

- [ ] **Verify**

```bash
cd backend && python -c "from models import ScoreRange, IngestJobResponse, JobStatusResponse, UnifiedSearchResponse; print('models ok')"
```
Expected: `models ok`

- [ ] **Commit**

```bash
git add backend/models.py
git commit -m "refactor: add ScoreRange, IngestJobResponse, JobStatusResponse to models"
```

---

## Task 7: Create `backend/routers/__init__.py`

**Files:** Create `backend/routers/__init__.py`

- [ ] **Create empty package marker**

```bash
mkdir -p backend/routers && touch backend/routers/__init__.py
```

- [ ] **Commit**

```bash
git add backend/routers/__init__.py
git commit -m "refactor: add routers package"
```

---

## Task 8: Create `backend/routers/videos.py`

**Files:** Create `backend/routers/videos.py`

- [ ] **Create the file**

```python
# backend/routers/videos.py
import os, json, logging
from fastapi import APIRouter, HTTPException, Path
from typing import Optional, List
from models import OVVerifyRequest
from db import clear_video
from indexes import evict_video

router = APIRouter()
logger = logging.getLogger(__name__)

BASE = os.path.dirname(os.path.dirname(__file__))
_data_dir    = os.path.join(BASE, "data")
_frames_dir  = os.path.join(_data_dir, "frames")
_media_dir   = os.path.join(_data_dir, "media")
_indexes_dir = os.path.join(_data_dir, "indexes")

_VIDEO_ID_RE = r"^[A-Za-z0-9_-]{1,64}$"


def _thumbnail_url_for(video_id: str) -> Optional[str]:
    frames_subdir = os.path.join(_frames_dir, video_id)
    if not os.path.isdir(frames_subdir):
        return None
    try:
        jpgs = sorted(f for f in os.listdir(frames_subdir)
                      if f.startswith("frame-") and f.endswith(".jpg"))
    except OSError:
        return None
    return f"/frames/{video_id}/{jpgs[0]}" if jpgs else None


@router.get("/videos")
async def list_videos():
    try:
        if not os.path.exists(_media_dir):
            return {"videos": []}
        from db import db
        videos = []
        for filename in os.listdir(_media_dir):
            if not filename.endswith(".mp4"):
                continue
            vid = filename[:-4]
            has_text   = os.path.exists(os.path.join(_indexes_dir, f"{vid}.faiss"))
            has_visual = os.path.exists(os.path.join(_indexes_dir, f"{vid}.svfaiss"))
            has_action = os.path.exists(os.path.join(_indexes_dir, f"{vid}.saclip.faiss"))
            # Fetch title, source_url, top_objects from video_context
            title, source_url, top_objects = None, None, []
            with db() as conn:
                row = conn.execute(
                    "SELECT title, source_url, objects_topk FROM video_context WHERE video_id=?",
                    (vid,),
                ).fetchone()
            if row:
                title, source_url, obj_json = row
                try:
                    obj_dict = json.loads(obj_json or "{}")
                    top_objects = list(obj_dict.keys())[:10]
                except (json.JSONDecodeError, TypeError):
                    top_objects = []
            videos.append({
                "video_id": vid,
                "title": title,
                "source_url": source_url,
                "has_text_search": has_text,
                "has_visual_search": has_visual,
                "has_action_search": has_action,
                "thumbnail_url": _thumbnail_url_for(vid),
                "top_objects": top_objects,
            })
        return {"videos": sorted(videos, key=lambda x: x["video_id"])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing videos: {e}")


@router.delete("/videos/{video_id}")
def delete_video(video_id: str = Path(..., pattern=_VIDEO_ID_RE)):
    try:
        result = clear_video(video_id)
        evict_video(video_id)
    except Exception as e:
        logger.exception("delete_video failed for %s", video_id)
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")
    return {"success": True, "video_id": video_id, **result}


@router.post("/ov_verify")
def ov_verify(req: OVVerifyRequest):
    from gdino import detect_on_image
    out = {}
    for f in req.frames:
        try:
            res = detect_on_image(
                image_path=os.path.abspath(f),
                prompts=req.prompts,
                box_threshold=req.box_threshold,
                text_threshold=req.text_threshold,
            )
            out[f] = res
        except Exception as e:
            out[f] = {"detections": [], "debug": {"error": str(e)}}
    return {"results": out}
```

- [ ] **Commit**

```bash
git add backend/routers/videos.py
git commit -m "refactor: add routers/videos.py with GET /videos (title+top_objects), DELETE, ov_verify"
```

---

## Task 9: Create `backend/routers/ingest.py`

**Files:** Create `backend/routers/ingest.py`

- [ ] **Create the file**

```python
# backend/routers/ingest.py
import os, re, uuid, time, logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, List
from urllib.parse import urlparse, parse_qs
from fastapi import APIRouter, HTTPException
from models import VideoIngestRequest

router = APIRouter()
logger = logging.getLogger(__name__)

BASE = os.path.dirname(os.path.dirname(__file__))
_media_dir = os.path.join(BASE, "data", "media")

_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, "_JobState"] = {}
_JOB_TTL = 600  # seconds to keep terminal jobs


@dataclass
class _JobState:
    job_id: str
    video_id: str
    status: str        # queued | running | done | error
    stage: str
    error: Optional[str] = None
    created_at: float = field(default_factory=time.monotonic)


def _set(job_id: str, stage: str, status: str = "running"):
    if job_id in _jobs:
        _jobs[job_id].status = status
        _jobs[job_id].stage = stage


def _run_ingest(job_id: str, url: str, video_id: str):
    try:
        _set(job_id, "Downloading & extracting frames…")
        # Fetch title before downloading (no-download metadata call)
        title, source_url = None, None
        try:
            import yt_dlp
            with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get("title")
                source_url = info.get("webpage_url") or url
        except Exception:
            logger.warning("Could not fetch yt-dlp metadata for %s", url)

        from visual_ingest import ingest_visual as do_visual
        do_visual(url)

        _set(job_id, "Transcribing audio…")
        from ingest import ingest as do_ingest
        do_ingest(url)

        _set(job_id, "Building context…")
        from context import build_video_context
        build_video_context(video_id)

        if title or source_url:
            from db import store_video_meta
            store_video_meta(video_id, title, source_url)

        _jobs[job_id].status = "done"
        _jobs[job_id].stage = "Done"
    except Exception as e:
        logger.exception("Ingest failed for %s", video_id)
        _jobs[job_id].status = "error"
        _jobs[job_id].error = str(e)


def _extract_video_id(video_url: str, override: Optional[str]) -> str:
    if override:
        return override
    if "youtube.com" in video_url or "youtu.be" in video_url:
        if "youtu.be/" in video_url:
            vid = video_url.split("youtu.be/")[-1].split("?")[0]
        else:
            parsed = urlparse(video_url)
            vid = parse_qs(parsed.query).get("v", [None])[0]
        if not vid:
            raise ValueError("Could not extract video ID from YouTube URL")
        return vid
    return re.sub(r"[^a-zA-Z0-9_-]", "", video_url.split("/")[-1])[:11]


@router.post("/ingest")
async def ingest_video(request: VideoIngestRequest):
    try:
        video_id = _extract_video_id(request.video_url, request.video_id)
    except ValueError as e:
        raise HTTPException(400, str(e))

    media_path = os.path.join(_media_dir, f"{video_id}.mp4")
    if os.path.exists(media_path):
        return {"job_id": None, "video_id": video_id,
                "status": "already_exists", "message": f"Video {video_id} already exists"}

    job_id = str(uuid.uuid4())
    _jobs[job_id] = _JobState(job_id=job_id, video_id=video_id,
                               status="queued", stage="Queued…")
    _executor.submit(_run_ingest, job_id, request.video_url, video_id)
    return {"job_id": job_id, "video_id": video_id, "status": "queued"}


@router.get("/ingest/status/{job_id}")
async def ingest_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found or expired")
    if job.status in ("done", "error") and (time.monotonic() - job.created_at) > _JOB_TTL:
        del _jobs[job_id]
        raise HTTPException(404, "Job expired")
    return {
        "job_id": job.job_id,
        "video_id": job.video_id,
        "status": job.status,
        "stage": job.stage,
        "error": job.error,
    }


@router.post("/build_contexts")
async def rebuild_video_contexts(video_ids: Optional[List[str]] = None):
    from db import db
    from context import build_video_context
    if video_ids is None:
        with db() as conn:
            rows = conn.execute("""
                SELECT DISTINCT video_id FROM chunks
                UNION SELECT DISTINCT video_id FROM visual_chunks
                UNION SELECT DISTINCT video_id FROM visual_clips
            """).fetchall()
        video_ids = [r[0] for r in rows]
    results = []
    for vid in video_ids:
        try:
            build_video_context(vid)
            results.append({"video_id": vid, "status": "success"})
        except Exception as e:
            logger.exception("build_video_context failed for %s", vid)
            results.append({"video_id": vid, "status": "error", "error": str(e)})
    ok = sum(1 for r in results if r["status"] == "success")
    return {"success": True, "message": f"Built contexts for {ok}/{len(results)} videos",
            "results": results}
```

- [ ] **Commit**

```bash
git add backend/routers/ingest.py
git commit -m "feat: async ingest with job status polling in routers/ingest.py"
```

---

## Task 10: Create `backend/routers/search.py`

**Files:** Create `backend/routers/search.py`

- [ ] **Create the file** — this contains all search logic extracted from `app.py`, plus `score_range` added to every `/query` response branch. The helper functions (`dedupe_hits`, `nms_time`, `_caption_hits_lazy`, etc.) move here verbatim. Only imports and the `score_range` lines are new.

```python
# backend/routers/search.py
import os, json, logging
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
from models import (
    SearchResponse, SearchHit, UnifiedSearchRequest,
    UnifiedSearchHit, UnifiedSearchResponse, ScoreRange, MAX_K,
)
from indexes import (
    load_index, load_siglip_visual_index, load_siglip_action_clips_index,
)
from db import get_cached_captions, put_cached_captions
from context import filter_videos_by_context
from embeddings import get_emb

router = APIRouter()
logger = logging.getLogger(__name__)

BASE = os.path.dirname(os.path.dirname(__file__))
_indexes_dir = os.path.join(BASE, "data", "indexes")
_frames_dir  = os.path.join(BASE, "data", "frames")

# SigLIP text tower — lazy
_SIGLIP = None
def _get_siglip():
    global _SIGLIP
    if _SIGLIP is None:
        from visual_ingest import SigLIPEncoder
        _SIGLIP = SigLIPEncoder("google/siglip-base-patch16-224")
    return _SIGLIP

# Florence-2 captioner — lazy
_CAPTIONER = None
def _get_captioner():
    global _CAPTIONER
    if _CAPTIONER is None:
        from visual_ingest import Florence2Captioner
        _CAPTIONER = Florence2Captioner("microsoft/Florence-2-base")
    return _CAPTIONER


def _val(h, name, default=None):
    return h.get(name, default) if isinstance(h, dict) else getattr(h, name, default)


def dedupe_hits(hits, prefer="max", key_mode="auto"):
    def make_key(h):
        vid = _val(h, "video_id")
        frame = _val(h, "frame")
        if key_mode == "frame" or (key_mode == "auto" and frame):
            return ("frame", vid, frame)
        start = float(_val(h, "start", 0.0))
        end   = float(_val(h, "end", start))
        return ("time", vid, round(start, 3), round(end, 3))
    best = {}
    for h in hits:
        k = make_key(h)
        cur = best.get(k)
        s = float(_val(h, "score", 0.0))
        if cur is None or (prefer == "max" and s > float(_val(cur, "score", -1e9))):
            best[k] = h
    out = list(best.values())
    out.sort(key=lambda x: (-float(_val(x, "score", 0.0)), float(_val(x, "start", 0.0))))
    return out


def nms_time(hits, tol=0.5):
    sorted_hits = sorted(hits, key=lambda x: -float(_val(x, "score", 0.0)))
    kept = []
    def mid(h):
        return (float(_val(h, "start", 0.0)) + float(_val(h, "end", 0.0))) * 0.5
    for h in sorted_hits:
        m, vid = mid(h), _val(h, "video_id")
        if all(not (vid == _val(k, "video_id") and abs(m - mid(k)) <= tol) for k in kept):
            kept.append(h)
    return kept


def _caption_hits_lazy(video_id: Optional[str], hits: list) -> None:
    if not hits:
        return
    by_vid: dict = {}
    for h in hits:
        vid = video_id or h.get("video_id")
        if not vid or not h.get("frame"):
            continue
        by_vid.setdefault(vid, []).append(h)
    captioner = None
    for vid, vhits in by_vid.items():
        frames = [h["frame"] for h in vhits]
        cached = get_cached_captions(vid, frames)
        missing = [h for h in vhits if h["frame"] not in cached]
        new_entries: dict = {}
        if missing:
            if captioner is None:
                captioner = _get_captioner()
            from PIL import Image as _PilImage
            for h in missing:
                frame = h["frame"]
                abs_path = frame if os.path.isabs(frame) else os.path.join(BASE, frame)
                try:
                    img = _PilImage.open(abs_path).convert("RGB")
                    result = captioner.process_image(img)
                except Exception:
                    logger.exception("lazy caption failed for %s", frame)
                    result = {"caption": "", "objects": []}
                new_entries[frame] = result
            put_cached_captions(vid, new_entries)
        for h in vhits:
            entry = cached.get(h["frame"]) or new_entries.get(h["frame"]) or {"caption": "", "objects": []}
            h["caption"] = entry["caption"]
            h["objects"] = entry["objects"]


def _parse_objects(raw: Optional[str]) -> list:
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


def _apply_filter_objects(objs: list, filter_objects: Optional[str]) -> bool:
    if not filter_objects or not objs:
        return True
    needle = filter_objects.strip().lower()
    return not needle or any(needle == str(o).strip().lower() for o in objs)


def _hit_matches_filter(hit: dict, filter_objects: Optional[str]) -> bool:
    if not filter_objects:
        return True
    needle = filter_objects.strip().lower()
    if not needle:
        return True
    caption = (hit.get("caption") or "").lower()
    objs = hit.get("objects") or []
    if caption and needle in caption:
        return True
    if any(needle in str(o).strip().lower() for o in objs):
        return True
    return not caption and not objs


def _maybe_caption_rerank(hits, *, verify_on, prompts, require_all,
                           w_base=0.7, w_caption=0.3):
    if not (verify_on and prompts and hits):
        return hits
    terms = [p.strip().lower() for p in prompts if p.strip()]
    req   = [r.strip().lower() for r in (require_all or []) if r.strip()]
    for h in hits:
        cap = (h.get("caption") or "").lower()
        if not cap:
            h["verify_score"] = 0.0
            h["score_fused"] = w_base * float(h.get("score", 0.0))
            continue
        matched = sum(1 for t in terms if t in cap)
        verify = matched / len(terms) if terms else 0.0
        if req and not all(r in cap for r in req):
            verify *= 0.3
        h["verify_score"] = verify
        h["score_fused"] = w_base * float(h.get("score", 0.0)) + w_caption * verify
    return hits


def _postproc_hits(hits, *, key_mode, k):
    hits = dedupe_hits(hits, key_mode=key_mode)
    hits = nms_time(hits, tol=0.5)
    return hits[:k] if k else hits


def _as_unified(h: dict) -> UnifiedSearchHit:
    return UnifiedSearchHit(
        start=float(h.get("start", 0.0)),
        end=float(h.get("end", h.get("start", 0.0))),
        score=float(h.get("score_fused", h.get("score", 0.0))),
        frame=h.get("frame"),
        objects=h.get("objects"),
        caption=h.get("caption"),
        text=h.get("text"),
        video_id=h.get("video_id"),
    )


def _score_range(hits: list) -> ScoreRange:
    scores = [float(h.get("score_fused", h.get("score", 0.0))) for h in hits]
    if not scores:
        return ScoreRange(min=0.0, max=0.0)
    return ScoreRange(min=min(scores), max=max(scores))


def have_indexes(video_id: str, need_text=False, need_visual=False, need_action=False) -> bool:
    ok = True
    if need_text:
        ok &= os.path.exists(os.path.join(_indexes_dir, f"{video_id}.faiss"))
    if need_visual:
        ok &= os.path.exists(os.path.join(_indexes_dir, f"{video_id}.svfaiss"))
    if need_action:
        ok &= os.path.exists(os.path.join(_indexes_dir, f"{video_id}.saclip.faiss"))
    return ok


def list_video_ids_with(ext: str) -> list:
    if not os.path.isdir(_indexes_dir):
        return []
    return [fn[:-len(ext)] for fn in os.listdir(_indexes_dir) if fn.endswith(ext)]


def _globally(needle_ext: str, restrict: Optional[list]) -> list:
    vids = list_video_ids_with(needle_ext)
    return [v for v in vids if (not restrict or v in restrict)]


def representative_frame_for_segment(video_id: str, seg_start: float, seg_end: float) -> Optional[str]:
    from db import db
    mid = 0.5 * (float(seg_start) + float(seg_end))
    with db() as conn:
        row = conn.execute("""
            SELECT frame FROM visual_chunks
            WHERE video_id=? AND start <= ? AND end >= ?
            ORDER BY ABS((start+end)/2.0 - ?) ASC LIMIT 1
        """, (video_id, mid, mid, mid)).fetchone()
        if not row:
            row = conn.execute("""
                SELECT frame FROM visual_chunks
                WHERE video_id=? AND end >= ? AND start <= ?
                ORDER BY CASE WHEN ? BETWEEN start AND end THEN 0
                              ELSE MIN(ABS(start-?), ABS(end-?)) END ASC LIMIT 1
            """, (video_id, seg_start, seg_end, mid, mid, mid)).fetchone()
    return row[0] if row else None


def search_text_single(video_id: str, q: str, k: int) -> list:
    index, rows = load_index(video_id)
    qv = get_emb().encode([q], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, k)
    return [{"video_id": video_id, "start": float(rows[idx][1]), "end": float(rows[idx][2]),
             "score": float(s), "text": rows[idx][3]}
            for s, idx in zip(D[0].tolist(), I[0].tolist()) if idx != -1]


def search_visual_single(video_id: str, q: str, k: int, filter_objects: Optional[str]) -> list:
    index, rows = load_siglip_visual_index(video_id)
    qv = _get_siglip().encode_text([q])
    D, I = index.search(qv, k)
    out = []
    for s, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        _, start, end, frame, objects, caption = rows[idx]
        objs = _parse_objects(objects)
        if not _apply_filter_objects(objs, filter_objects):
            continue
        out.append({"video_id": video_id, "start": float(start), "end": float(end),
                    "score": float(s), "frame": frame, "objects": objs, "caption": caption or ""})
    return out


def search_action_single(video_id: str, q: str, k: int, filter_objects: Optional[str]) -> list:
    index, rows = load_siglip_action_clips_index(video_id)
    qv = _get_siglip().encode_text([q])
    D, I = index.search(qv, k)
    out = []
    for s, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        _, start, end, objects_json, caption = rows[idx]
        objs = _parse_objects(objects_json)
        if not _apply_filter_objects(objs, filter_objects):
            continue
        out.append({"video_id": video_id, "start": float(start), "end": float(end),
                    "score": float(s), "objects": objs, "caption": caption or ""})
    return out


def search_text_global(q: str, k: int, restrict_videos=None) -> list:
    candidates = filter_videos_by_context(q, restrict_videos, topn=100, min_cos=0.18)
    vids = candidates if candidates else _globally(".faiss", restrict_videos)
    all_hits = []
    for vid in vids:
        try:
            all_hits.extend(search_text_single(vid, q, k))
        except Exception:
            logger.warning("text global skipped %s", vid, exc_info=True)
    all_hits.sort(key=lambda h: h["score"], reverse=True)
    return all_hits


def search_visual_global(q: str, k: int, filter_objects=None, restrict_videos=None) -> list:
    candidates = filter_videos_by_context(q, restrict_videos, topn=100, min_cos=0.18)
    vids = candidates if candidates else _globally(".svfaiss", restrict_videos)
    all_hits = []
    for vid in vids:
        try:
            all_hits.extend(search_visual_single(vid, q, k, filter_objects))
        except Exception:
            logger.warning("visual global skipped %s", vid, exc_info=True)
    all_hits.sort(key=lambda h: h["score"], reverse=True)
    return all_hits


def search_action_global(q: str, k: int, filter_objects=None, restrict_videos=None) -> list:
    candidates = filter_videos_by_context(q, restrict_videos, topn=100, min_cos=0.18)
    vids = candidates if candidates else _globally(".saclip.faiss", restrict_videos)
    all_hits = []
    for vid in vids:
        try:
            all_hits.extend(search_action_single(vid, q, k, filter_objects))
        except Exception:
            logger.warning("action global skipped %s", vid, exc_info=True)
    all_hits.sort(key=lambda h: h["score"], reverse=True)
    return all_hits


def chain_actions(video_id: str, steps: list, k_per_step=40, max_gap=8.0,
                  filter_objects=None):
    cand = [search_action_single(video_id, q, k_per_step, filter_objects) for q in steps]
    paths = [[h] for h in cand[0]] if cand and cand[0] else []
    for t in range(1, len(steps)):
        new_paths = []
        for h in cand[t]:
            best, best_score = None, -1e9
            for p in paths:
                prev = p[-1]
                if h["start"] >= prev["end"] and (h["start"] - prev["end"] <= max_gap):
                    score = sum(x["score"] for x in p) + h["score"]
                    if score > best_score:
                        best, best_score = p, score
            if best:
                new_paths.append(best + [h])
        if not new_paths and cand[t]:
            new_paths = [[h] for h in cand[t][:3]]
        paths = new_paths if new_paths else paths
    paths.sort(key=lambda p: sum(x["score"] for x in p), reverse=True)
    full = [p for p in paths if len(p) == len(steps)]
    chosen = full[0] if full else (paths[0] if paths else [])
    return chosen, cand


# ── Legacy GET endpoints (deprecated, kept for backward compat) ──

@router.get("/search", response_model=SearchResponse)
async def search(video_id: str = Query(...), q: str = Query(...),
                 k: int = Query(5, ge=1, le=MAX_K)):
    """Deprecated: use POST /query instead."""
    index, rows = load_index(video_id)
    qv = get_emb().encode([q], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, k)
    hits = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        _, start, end, text = rows[idx]
        hits.append(SearchHit(start=start, end=end, text=text, score=score))
    return SearchResponse(video_id=video_id, hits=hits)


@router.get("/vsearch")
async def vsearch(video_id: str = Query(...), q: str = Query(...),
                  k: int = Query(6, ge=1, le=MAX_K), filter_objects: str = None):
    """Deprecated: use POST /query with mode=visual instead."""
    try:
        hits = search_visual_single(video_id, q, k, filter_objects)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    hits = _postproc_hits(hits, key_mode="auto", k=k)
    return {"video_id": video_id, "hits": hits}


@router.get("/asearch")
async def asearch(video_id: str = Query(...), q: str = Query(...),
                  k: int = Query(40, ge=1, le=MAX_K), filter_objects: str = None):
    """Deprecated: use POST /query with mode=action instead."""
    try:
        raw = search_action_single(video_id, q, k, filter_objects)
        hits = _postproc_hits(raw, key_mode="time", k=k)
        return {"video_id": video_id, "hits": hits}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/asearch_chain")
async def asearch_chain(video_id: str = Query(...),
                        steps: List[str] = Query(...),
                        k_per_step: int = Query(40, ge=1, le=MAX_K),
                        max_gap: float = Query(8.0, ge=0.0, le=60.0),
                        filter_objects: str = None):
    """Deprecated: use POST /query with mode=action_chain instead."""
    try:
        path, cand = chain_actions(video_id, steps, k_per_step=k_per_step,
                                   max_gap=max_gap, filter_objects=filter_objects)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Action clip index missing: {e}")
    hits = _postproc_hits(path, key_mode="time", k=None)
    return {"video_id": video_id, "steps": steps, "best_path": hits,
            "candidates_preview": [c[:5] for c in cand]}


@router.get("/asearch_all")
def asearch_all(q: str = Query(...), k: int = Query(50, ge=1, le=MAX_K),
                filter_objects: str = None,
                videos: list = Query(None)):
    try:
        hits = search_action_global(q, k, filter_objects=filter_objects, restrict_videos=videos)
        return {"query": q, "hits": hits[:k]}
    except Exception as e:
        logger.exception("/asearch_all failed")
        raise HTTPException(status_code=500, detail=f"Search error: {e}")


# ── Unified /query ──

@router.post("/query", response_model=UnifiedSearchResponse)
def unified_query(body: UnifiedSearchRequest):
    from utils_unified import extract_video_id
    scope = (body.scope or "video").lower()
    restrict = body.videos

    if scope == "video":
        vid = body.video_id or extract_video_id(body.video_url or "")
        if not vid:
            raise HTTPException(400, "video_id or video_url required for scope='video'")
        if body.ingest_if_needed and not have_indexes(
            vid,
            need_text=body.mode == "text",
            need_visual=body.mode == "visual",
            need_action=body.mode in ("action", "action_chain"),
        ):
            from ingest import ingest as do_ingest
            from visual_ingest import ingest_visual as do_visual
            if body.mode == "text":
                do_ingest(body.video_url or "")
            else:
                do_visual(body.video_url or "")

        if body.mode == "text":
            try:
                raw = search_text_single(vid, body.query or "", body.k)
            except FileNotFoundError:
                raise HTTPException(404, f"No text index for video {vid} — ingest first")
            hits = [UnifiedSearchHit(start=h["start"], end=h["end"], score=h["score"],
                                     text=h["text"], video_id=h["video_id"]) for h in raw]
            return UnifiedSearchResponse(video_id=vid, mode="text", hits=hits,
                                         score_range=_score_range(raw))

        if body.mode == "visual":
            try:
                raw = search_visual_single(vid, body.query or "", body.k * 3, body.filter_objects)
            except FileNotFoundError:
                raise HTTPException(404, f"No visual index for video {vid} — ingest first")
            _caption_hits_lazy(vid, raw)
            if body.filter_objects:
                raw = [h for h in raw if _hit_matches_filter(h, body.filter_objects)]
            raw = _maybe_caption_rerank(raw, verify_on=body.verify_with_gdino,
                                        prompts=body.verify_prompts or [],
                                        require_all=body.verify_require_all or [])
            raw.sort(key=lambda h: float(h.get("score_fused", h.get("score", 0.0))), reverse=True)
            sr = _score_range(raw)
            raw = _postproc_hits(raw, key_mode="auto", k=body.k)
            return UnifiedSearchResponse(video_id=vid, mode="visual",
                                         hits=[_as_unified(h) for h in raw], score_range=sr)

        if body.mode == "action":
            try:
                raw = search_action_single(vid, body.query or "", body.k * 3, body.filter_objects)
            except FileNotFoundError:
                raise HTTPException(404, f"No action index for video {vid} — ingest first")
            for h in raw:
                if not h.get("frame"):
                    h["frame"] = representative_frame_for_segment(vid, h["start"], h["end"])
            _caption_hits_lazy(vid, raw)
            if body.filter_objects:
                raw = [h for h in raw if _hit_matches_filter(h, body.filter_objects)]
            raw = _maybe_caption_rerank(raw, verify_on=body.verify_with_gdino,
                                        prompts=body.verify_prompts or [],
                                        require_all=body.verify_require_all or [])
            raw.sort(key=lambda h: float(h.get("score_fused", h.get("score", 0.0))), reverse=True)
            sr = _score_range(raw)
            raw = _postproc_hits(raw, key_mode="time", k=body.k)
            return UnifiedSearchResponse(video_id=vid, mode="action",
                                         hits=[_as_unified(h) for h in raw], score_range=sr)

        if body.mode == "action_chain":
            if not body.steps:
                raise HTTPException(400, "steps required for mode=action_chain")
            path_hits, cand = chain_actions(vid, body.steps, k_per_step=body.k,
                                            max_gap=body.max_gap,
                                            filter_objects=body.filter_objects)
            path_hits = _maybe_caption_rerank(path_hits, verify_on=body.verify_with_gdino,
                                              prompts=body.verify_prompts or [],
                                              require_all=body.verify_require_all or [])
            path_hits.sort(key=lambda h: float(h.get("score_fused", h.get("score", 0.0))), reverse=True)
            sr = _score_range(path_hits)
            path_hits = _postproc_hits(path_hits, key_mode="time", k=body.k)
            return UnifiedSearchResponse(video_id=vid, mode="action_chain",
                                         hits=[_as_unified(h) for h in path_hits],
                                         score_range=sr,
                                         info={"steps": body.steps,
                                               "preview_per_step": [c[:5] for c in cand]})
        raise HTTPException(400, f"Unknown mode {body.mode}")

    if scope == "global":
        if body.mode == "text":
            raw = search_text_global(body.query or "", body.k, restrict_videos=restrict)
            sr = _score_range(raw)
            hits = [UnifiedSearchHit(start=h["start"], end=h["end"], score=h["score"],
                                     text=h["text"], video_id=h["video_id"]) for h in raw]
            return UnifiedSearchResponse(video_id=None, mode="text", hits=hits, score_range=sr)

        if body.mode == "visual":
            raw = search_visual_global(body.query or "", body.k * 3,
                                       filter_objects=body.filter_objects,
                                       restrict_videos=restrict)
            _caption_hits_lazy(None, raw)
            if body.filter_objects:
                raw = [h for h in raw if _hit_matches_filter(h, body.filter_objects)]
            raw = _maybe_caption_rerank(raw, verify_on=body.verify_with_gdino,
                                        prompts=body.verify_prompts or [],
                                        require_all=body.verify_require_all or [])
            raw.sort(key=lambda h: float(h.get("score_fused", h.get("score", 0.0))), reverse=True)
            sr = _score_range(raw)
            raw = _postproc_hits(raw, key_mode="auto", k=body.k)
            return UnifiedSearchResponse(video_id=None, mode="visual",
                                         hits=[_as_unified(h) for h in raw], score_range=sr)

        if body.mode == "action":
            raw = search_action_global(body.query or "", body.k * 3,
                                       filter_objects=body.filter_objects,
                                       restrict_videos=restrict)
            for h in raw:
                if not h.get("frame"):
                    h["frame"] = representative_frame_for_segment(h["video_id"], h["start"], h["end"])
            _caption_hits_lazy(None, raw)
            if body.filter_objects:
                raw = [h for h in raw if _hit_matches_filter(h, body.filter_objects)]
            raw = _maybe_caption_rerank(raw, verify_on=body.verify_with_gdino,
                                        prompts=body.verify_prompts or [],
                                        require_all=body.verify_require_all or [])
            raw.sort(key=lambda h: float(h.get("score_fused", h.get("score", 0.0))), reverse=True)
            sr = _score_range(raw)
            raw = _postproc_hits(raw, key_mode="time", k=body.k)
            return UnifiedSearchResponse(video_id=None, mode="action",
                                         hits=[_as_unified(h) for h in raw], score_range=sr)

        if body.mode == "action_chain":
            if not body.steps:
                raise HTTPException(400, "steps required for mode=action_chain")
            vids = _globally(".saclip.faiss", restrict)
            per_video = []
            for vid in vids:
                try:
                    path_hits, cand = chain_actions(vid, body.steps, k_per_step=body.k,
                                                    max_gap=body.max_gap,
                                                    filter_objects=body.filter_objects)
                    path_hits = _maybe_caption_rerank(path_hits, verify_on=body.verify_with_gdino,
                                                      prompts=body.verify_prompts or [],
                                                      require_all=body.verify_require_all or [])
                    total = sum(float(h.get("score_fused", h.get("score", 0.0))) for h in path_hits)
                    per_video.append((vid, path_hits, cand, total))
                except Exception:
                    continue
            if not per_video:
                return UnifiedSearchResponse(video_id=None, mode="action_chain", hits=[],
                                             info={"steps": body.steps})
            per_video.sort(key=lambda x: x[3], reverse=True)
            best_vid, best_path, best_cands, _ = per_video[0]
            best_path.sort(key=lambda h: float(h.get("score_fused", h.get("score", 0.0))), reverse=True)
            sr = _score_range(best_path)
            best_path = _postproc_hits(best_path, key_mode="time", k=body.k)
            return UnifiedSearchResponse(video_id=best_vid, mode="action_chain",
                                         hits=[_as_unified(h) for h in best_path],
                                         score_range=sr,
                                         info={"steps": body.steps, "selected_video": best_vid,
                                               "preview_per_step": [c[:5] for c in best_cands]})
        raise HTTPException(400, "Unknown or unsupported mode for scope='global'")

    raise HTTPException(400, f"Unknown scope {scope}")
```

- [ ] **Commit**

```bash
git add backend/routers/search.py
git commit -m "refactor: extract routers/search.py with score_range in /query responses"
```

---

## Task 11: Slim `backend/app.py`

**Files:** Modify `backend/app.py`

Replace the entire file with the slim version below. All logic has moved to routers; `app.py` only wires things together.

- [ ] **Replace `app.py`**

```python
# backend/app.py
import os, logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from db import init_db
from routers import search, ingest, videos

logging.basicConfig(level=logging.INFO)
app = FastAPI()

_cors_raw = os.environ.get("CORS_ORIGINS", "").strip()
if _cors_raw:
    _cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
else:
    _cors_origins = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_BASE = os.path.dirname(__file__)
_data_dir   = os.path.join(_BASE, "data")
_frames_dir = os.path.join(_data_dir, "frames")
_media_dir  = os.path.join(_data_dir, "media")
os.makedirs(_frames_dir, exist_ok=True)
os.makedirs(_media_dir, exist_ok=True)
app.mount("/frames", StaticFiles(directory=_frames_dir), name="frames")
app.mount("/media",  StaticFiles(directory=_media_dir),  name="media")

app.include_router(search.router)
app.include_router(ingest.router)
app.include_router(videos.router)


@app.on_event("startup")
async def startup():
    init_db()
```

- [ ] **Start the server and confirm it boots**

```bash
cd backend && uvicorn app:app --reload --port 8000
```
Expected: `Application startup complete.` with no import errors.

- [ ] **Verify key endpoints respond**

```bash
curl -s http://localhost:8000/videos | python3 -m json.tool | head -5
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"mode":"text","query":"test","scope":"global","k":3}' | python3 -m json.tool | head -10
```
Expected: both return valid JSON (videos list may be empty; query returns `{"video_id":null,"mode":"text","hits":[],...}`).

- [ ] **Commit**

```bash
git add backend/app.py
git commit -m "refactor: slim app.py to router includes + init_db startup"
```

---

## Task 12: Backend smoke test — ingest status endpoint

With the server still running from Task 11:

- [ ] **Test the new /ingest/status 404 path**

```bash
curl -s http://localhost:8000/ingest/status/nonexistent-id
```
Expected: `{"detail":"Job not found or expired"}`

- [ ] **Test /ingest returns queued immediately** (use a fake URL to verify the shape, it will fail in the background)

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"video_url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}' | python3 -m json.tool
```
Expected: `{"job_id": "<uuid>", "video_id": "dQw4w9WgXcQ", "status": "queued"}`

- [ ] **Confirm score_range appears in /query response**

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"mode":"visual","query":"a person","scope":"global","k":5}' | python3 -m json.tool | grep -A3 score_range
```
Expected: `"score_range": {"min": ..., "max": ...}` (or `null` if no indexes exist yet).

- [ ] **Commit** (nothing to commit — this is a verification task only)

---

## Task 13: Update `frontend/src/lib/api.ts`

**Files:** Modify `frontend/src/lib/api.ts`

- [ ] **Replace the entire file**

```typescript
// frontend/src/lib/api.ts
const BASE_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000'

export interface VideoMeta {
  video_id: string
  title: string | null
  source_url: string | null
  has_text_search: boolean
  has_visual_search: boolean
  has_action_search: boolean
  thumbnail_url?: string | null
  top_objects: string[]
}

export interface ScoreRange {
  min: number
  max: number
}

export interface UnifiedSearchHit {
  start: number
  end: number
  score: number
  text?: string
  frame?: string
  objects?: string[]
  caption?: string
  video_id: string
}

export interface UnifiedSearchResponse {
  video_id: string | null
  mode: 'text' | 'visual' | 'action' | 'action_chain'
  hits: UnifiedSearchHit[]
  info: Record<string, unknown>
  score_range: ScoreRange | null
}

export interface IngestJobResponse {
  job_id: string | null
  video_id: string
  status: 'queued' | 'already_exists'
  message?: string
}

export interface JobStatusResponse {
  job_id: string
  video_id: string
  status: 'queued' | 'running' | 'done' | 'error'
  stage: string
  error: string | null
}

export interface IngestResponse {
  success: boolean
  message: string
  video_id: string
  status: 'completed' | 'already_exists'
}

export type SearchMode = 'text' | 'visual' | 'action' | 'action_chain'
export type SearchScope = 'video' | 'global'

export interface UnifiedSearchRequest {
  video_url?: string
  video_id?: string
  query?: string
  mode: SearchMode
  k?: number
  filter_objects?: string
  steps?: string[]
  max_gap?: number
  ingest_if_needed?: boolean
  scope?: SearchScope
  videos?: string[]
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, init)
  if (!res.ok) {
    const body = await res.text()
    throw new Error(body || `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  baseUrl: BASE_URL,

  getVideos(): Promise<{ videos: VideoMeta[] }> {
    return request('/videos')
  },

  ingest(video_url: string, video_id?: string): Promise<IngestJobResponse> {
    return request('/ingest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_url, video_id }),
    })
  },

  ingestStatus(jobId: string): Promise<JobStatusResponse> {
    return request(`/ingest/status/${encodeURIComponent(jobId)}`)
  },

  buildContexts(videoIds?: string[]): Promise<{ success: boolean; message: string }> {
    return request('/build_contexts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(videoIds ?? null),
    })
  },

  query(req: UnifiedSearchRequest): Promise<UnifiedSearchResponse> {
    return request('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
  },

  deleteVideo(videoId: string): Promise<{ success: boolean; video_id: string }> {
    return request(`/videos/${encodeURIComponent(videoId)}`, { method: 'DELETE' })
  },

  assetUrl(path: string): string {
    return path.startsWith('http') ? path : `${BASE_URL}${path}`
  },

  frameUrl(framePath: string): string {
    const match = framePath.match(/(?:data\/)?frames\/(.+)/)
    const rel = match ? match[1] : framePath
    return `${BASE_URL}/frames/${rel}`
  },

  mediaUrl(videoId: string): string {
    return `${BASE_URL}/media/${videoId}.mp4`
  },
}
```

- [ ] **Commit**

```bash
git add frontend/src/lib/api.ts
git commit -m "feat: update api.ts with new types and ingestStatus/buildContexts methods"
```

---

## Task 14: Create `frontend/src/hooks/useSearchHistory.ts`

**Files:** Create `frontend/src/hooks/useSearchHistory.ts`

- [ ] **Create the file**

```typescript
// frontend/src/hooks/useSearchHistory.ts
import { useState, useCallback } from 'react'
import type { SearchMode } from '@/lib/api'

const STORAGE_KEY = 'vsearch:history'
const MAX_HISTORY = 10

export interface HistoryEntry {
  query: string
  mode: SearchMode
  timestamp: number
}

function load(): HistoryEntry[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]')
  } catch {
    return []
  }
}

function save(entries: HistoryEntry[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(entries))
}

export function useSearchHistory() {
  const [history, setHistory] = useState<HistoryEntry[]>(load)

  const push = useCallback((query: string, mode: SearchMode) => {
    if (!query.trim()) return
    setHistory((prev) => {
      const filtered = prev.filter((e) => !(e.query === query && e.mode === mode))
      const next = [{ query, mode, timestamp: Date.now() }, ...filtered].slice(0, MAX_HISTORY)
      save(next)
      return next
    })
  }, [])

  const remove = useCallback((query: string, mode: SearchMode) => {
    setHistory((prev) => {
      const next = prev.filter((e) => !(e.query === query && e.mode === mode))
      save(next)
      return next
    })
  }, [])

  return { history, push, remove }
}
```

- [ ] **Commit**

```bash
git add frontend/src/hooks/useSearchHistory.ts
git commit -m "feat: add useSearchHistory hook with localStorage persistence"
```

---

## Task 15: Update `frontend/src/components/IngestModal.tsx`

**Files:** Modify `frontend/src/components/IngestModal.tsx`

Replace the entire file with the polling version:

- [ ] **Replace the file**

```typescript
// frontend/src/components/IngestModal.tsx
import { useState, useEffect, useRef } from 'react'
import { CheckCircle, AlertCircle, Loader2 } from 'lucide-react'
import { Dialog } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { api } from '@/lib/api'
import type { JobStatusResponse } from '@/lib/api'

interface IngestModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onSuccess: () => void
}

const STAGES = [
  'Downloading & extracting frames…',
  'Transcribing audio…',
  'Building context…',
]

function stageIndex(stage: string): number {
  return STAGES.findIndex((s) => s === stage)
}

export function IngestModal({ open, onOpenChange, onSuccess }: IngestModalProps) {
  const [url, setUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [jobStatus, setJobStatus] = useState<JobStatusResponse | null>(null)
  const [alreadyExists, setAlreadyExists] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  useEffect(() => {
    return () => stopPolling()
  }, [])

  const handleClose = (val: boolean) => {
    if (loading) return
    stopPolling()
    onOpenChange(val)
    if (!val) {
      setUrl('')
      setJobStatus(null)
      setAlreadyExists(false)
      setError(null)
    }
  }

  const handleIngest = async () => {
    if (!url.trim()) return
    setLoading(true)
    setError(null)
    setJobStatus(null)
    setAlreadyExists(false)
    try {
      const res = await api.ingest(url.trim())
      if (res.status === 'already_exists') {
        setAlreadyExists(true)
        setLoading(false)
        onSuccess()
        return
      }
      if (!res.job_id) {
        setError('No job ID returned')
        setLoading(false)
        return
      }
      const jobId = res.job_id
      pollRef.current = setInterval(async () => {
        try {
          const status = await api.ingestStatus(jobId)
          setJobStatus(status)
          if (status.status === 'done') {
            stopPolling()
            setLoading(false)
            onSuccess()
          } else if (status.status === 'error') {
            stopPolling()
            setError(status.error ?? 'Ingestion failed')
            setLoading(false)
          }
        } catch (e) {
          stopPolling()
          setError(e instanceof Error ? e.message : 'Polling failed')
          setLoading(false)
        }
      }, 2000)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Ingestion failed')
      setLoading(false)
    }
  }

  const done = jobStatus?.status === 'done'
  const currentStageIdx = jobStatus ? stageIndex(jobStatus.stage) : -1

  return (
    <Dialog open={open} onOpenChange={handleClose} title="Add video"
            description="Paste a YouTube URL to ingest and index.">
      <div className="space-y-4">
        <div className="space-y-1.5">
          <label className="text-xxs font-medium uppercase tracking-wide text-subtle">
            Video URL
          </label>
          <Input
            placeholder="https://www.youtube.com/watch?v=…"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !loading && handleIngest()}
            disabled={loading}
            autoFocus
          />
        </div>

        {loading && jobStatus && (
          <div className="space-y-2 rounded-md border border-accent/30 bg-accent-soft px-3 py-2.5">
            {STAGES.map((stage, i) => {
              const completed = currentStageIdx > i || done
              const active = currentStageIdx === i && !done
              return (
                <div key={stage} className="flex items-center gap-2 text-xs">
                  {completed ? (
                    <CheckCircle className="h-3.5 w-3.5 text-emerald-400 flex-shrink-0" />
                  ) : active ? (
                    <Loader2 className="h-3.5 w-3.5 text-accent animate-spin flex-shrink-0" />
                  ) : (
                    <span className="h-3.5 w-3.5 rounded-full border border-border flex-shrink-0" />
                  )}
                  <span className={active ? 'text-accent font-medium' : completed ? 'text-muted line-through' : 'text-dim'}>
                    {stage}
                  </span>
                </div>
              )
            })}
          </div>
        )}

        {loading && !jobStatus && (
          <div className="flex items-center gap-3 rounded-md border border-accent/30 bg-accent-soft px-3 py-2.5">
            <Loader2 className="h-4 w-4 animate-spin text-accent" />
            <p className="text-xs font-medium text-accent">Starting…</p>
          </div>
        )}

        {alreadyExists && (
          <div className="flex items-start gap-2.5 rounded-md border border-emerald-500/20 bg-emerald-500/10 px-3 py-2.5">
            <CheckCircle className="h-4 w-4 mt-0.5 flex-shrink-0 text-emerald-400" />
            <p className="text-xs font-medium text-emerald-400">Already indexed</p>
          </div>
        )}

        {done && (
          <div className="flex items-start gap-2.5 rounded-md border border-emerald-500/20 bg-emerald-500/10 px-3 py-2.5">
            <CheckCircle className="h-4 w-4 mt-0.5 flex-shrink-0 text-emerald-400" />
            <div className="text-xs">
              <p className="font-medium text-emerald-400">Done</p>
              <p className="text-muted">
                Video ID <span className="font-mono text-fg">{jobStatus?.video_id}</span>
              </p>
            </div>
          </div>
        )}

        {error && (
          <div className="flex items-start gap-2.5 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2.5">
            <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0 text-red-400" />
            <div className="text-xs">
              <p className="font-medium text-red-400">Failed</p>
              <p className="text-muted break-all">{error}</p>
            </div>
          </div>
        )}

        <div className="flex gap-2 pt-1">
          {done || alreadyExists ? (
            <Button variant="primary" size="md" className="flex-1" onClick={() => handleClose(false)}>
              Done
            </Button>
          ) : (
            <Button variant="primary" size="md" className="flex-1"
                    onClick={handleIngest} disabled={loading || !url.trim()}>
              {loading ? 'Ingesting…' : 'Ingest'}
            </Button>
          )}
          <Button variant="secondary" size="md" onClick={() => handleClose(false)} disabled={loading}>
            Cancel
          </Button>
        </div>
      </div>
    </Dialog>
  )
}
```

- [ ] **Commit**

```bash
git add frontend/src/components/IngestModal.tsx
git commit -m "feat: IngestModal now polls /ingest/status with stage checklist"
```

---

## Task 16: Update `frontend/src/components/ResultCard.tsx`

**Files:** Modify `frontend/src/components/ResultCard.tsx`

Two changes: accept an optional `title` prop, and accept `scoreRange` to compute normalized percent.

- [ ] **Replace the file**

```typescript
// frontend/src/components/ResultCard.tsx
import { useState } from 'react'
import { ExternalLink, Play } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Dialog } from '@/components/ui/dialog'
import { VideoPlayer } from '@/components/VideoPlayer'
import { api } from '@/lib/api'
import { formatTimeRange, isYouTubeId, youtubeUrl, truncate, cn } from '@/lib/utils'
import type { UnifiedSearchHit, ScoreRange } from '@/lib/api'

interface ResultCardProps {
  hit: UnifiedSearchHit
  index: number
  title?: string
  scoreRange: ScoreRange | null
}

function FrameThumb({ framePath }: { framePath: string }) {
  const [failed, setFailed] = useState(false)
  if (failed) {
    return (
      <div className="w-full aspect-video bg-surface2 flex items-center justify-center">
        <Play className="h-6 w-6 text-dim" />
      </div>
    )
  }
  return (
    <img src={api.frameUrl(framePath)} alt="" onError={() => setFailed(true)}
         loading="lazy" className="w-full aspect-video object-cover" />
  )
}

function normalizeScore(score: number, range: ScoreRange | null): number {
  if (!range) return Math.round(score * 100)
  const { min, max } = range
  if (max === min) return Math.round(score * 100)
  return Math.round(((score - min) / (max - min)) * 100)
}

function ScorePill({ pct }: { pct: number }) {
  const tone =
    pct >= 60 ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
    : pct >= 40 ? 'bg-accent-soft text-accent border-accent/20'
    : 'bg-surface2 text-muted border-border'
  return (
    <span className={cn('inline-flex items-center gap-1 rounded-full border px-1.5 py-0.5 font-mono text-[10px] leading-none', tone)}>
      {pct}%
    </span>
  )
}

export function ResultCard({ hit, index, title, scoreRange }: ResultCardProps) {
  const [open, setOpen] = useState(false)
  const pct = normalizeScore(hit.score, scoreRange)
  const isYT = isYouTubeId(hit.video_id)
  const hasFrame = Boolean(hit.frame)
  const hasCaption = Boolean(hit.caption?.trim())
  const hasText = Boolean(hit.text?.trim())
  const primaryText = hasText ? hit.text : hasCaption ? hit.caption : null
  const displayName = title ?? hit.video_id

  return (
    <>
      <article
        className="group rounded-lg border border-border bg-panel overflow-hidden transition-colors duration-150 hover:border-border-strong animate-slide-up cursor-pointer"
        style={{ animationDelay: `${Math.min(index, 20) * 20}ms` }}
        onClick={() => setOpen(true)}
      >
        <div className="relative">
          {hasFrame ? <FrameThumb framePath={hit.frame!} /> : (
            <div className="w-full aspect-video bg-surface2 flex items-center justify-center">
              <span className="font-mono text-xs text-dim">{formatTimeRange(hit.start, hit.end)}</span>
            </div>
          )}
          <div className="absolute inset-0 flex items-end justify-between p-2 bg-gradient-to-t from-black/60 via-transparent to-transparent">
            <span className="inline-flex items-center gap-1 rounded-md bg-black/50 px-1.5 py-0.5 font-mono text-[10px] text-white backdrop-blur-sm">
              {formatTimeRange(hit.start, hit.end)}
            </span>
            <ScorePill pct={pct} />
          </div>
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center opacity-0 transition-opacity duration-150 group-hover:opacity-100">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-white/95 shadow-lg">
              <Play className="h-4 w-4 text-black translate-x-[1px]" fill="currentColor" />
            </div>
          </div>
        </div>

        <div className="p-3 space-y-1.5">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-medium text-fg truncate">{displayName}</span>
            {isYT && (
              <button onClick={(e) => { e.stopPropagation(); window.open(youtubeUrl(hit.video_id, hit.start), '_blank') }}
                      title="Open in YouTube"
                      className="flex-shrink-0 p-1 -m-1 text-subtle hover:text-fg transition-colors">
                <ExternalLink className="h-3 w-3" />
              </button>
            )}
          </div>
          {title && (
            <span className="font-mono text-[10px] text-dim truncate block">{hit.video_id}</span>
          )}
          {primaryText && (
            <p className={cn('text-sm leading-snug line-clamp-3', hasText ? 'text-fg' : 'text-fg/90')}>
              {hasText ? `"${truncate(primaryText, 180)}"` : truncate(primaryText, 180)}
            </p>
          )}
          {hit.objects && hit.objects.length > 0 && (
            <div className="flex flex-wrap gap-1 pt-0.5">
              {hit.objects.slice(0, 4).map((obj) => (
                <Badge key={obj} variant="neutral" className="text-[10px] px-1.5 py-0">{obj}</Badge>
              ))}
              {hit.objects.length > 4 && (
                <Badge variant="outline" className="text-[10px] px-1.5 py-0">+{hit.objects.length - 4}</Badge>
              )}
            </div>
          )}
        </div>
      </article>

      <Dialog open={open} onOpenChange={setOpen} title={displayName}
              description={`${formatTimeRange(hit.start, hit.end)} · ${pct}% match`}
              className="w-[min(94vw,820px)]">
        <div className="space-y-4">
          {open && <VideoPlayer videoId={hit.video_id} startSeconds={hit.start} endSeconds={hit.end} />}
          {hasText && (
            <section>
              <h3 className="text-xxs font-medium uppercase tracking-wide text-subtle mb-1.5">Transcript</h3>
              <p className="rounded-md border border-border bg-surface/50 px-3 py-2 text-sm text-fg leading-relaxed">"{hit.text}"</p>
            </section>
          )}
          {hasCaption && (
            <section>
              <h3 className="text-xxs font-medium uppercase tracking-wide text-subtle mb-1.5">Caption</h3>
              <p className="rounded-md border border-border bg-surface/50 px-3 py-2 text-sm text-fg leading-relaxed">{hit.caption}</p>
            </section>
          )}
          {hit.objects && hit.objects.length > 0 && (
            <section>
              <h3 className="text-xxs font-medium uppercase tracking-wide text-subtle mb-1.5">Keywords</h3>
              <div className="flex flex-wrap gap-1.5">
                {hit.objects.map((obj) => <Badge key={obj} variant="neutral">{obj}</Badge>)}
              </div>
            </section>
          )}
          <div className="flex gap-2 pt-1">
            {isYT && (
              <Button variant="secondary" size="md" className="flex-1"
                      onClick={() => window.open(youtubeUrl(hit.video_id, hit.start), '_blank')}>
                <ExternalLink className="h-4 w-4" />
                Open on YouTube at {formatTimeRange(hit.start, hit.end)}
              </Button>
            )}
            <Button variant="primary" size="md" onClick={() => setOpen(false)}>Close</Button>
          </div>
        </div>
      </Dialog>
    </>
  )
}
```

- [ ] **Commit**

```bash
git add frontend/src/components/ResultCard.tsx
git commit -m "feat: ResultCard shows title and normalized score relative to result set"
```

---

## Task 17: Update `frontend/src/components/FilterPanel.tsx`

**Files:** Modify `frontend/src/components/FilterPanel.tsx`

Add `objectSuggestions: string[]` prop and wire a `<datalist>` to the filter input.

- [ ] **Replace the `FilterPanelProps` interface and `FilterPanel` function** — change the interface and add `objectSuggestions`:

Change the interface to:
```typescript
interface FilterPanelProps {
  videos: VideoMeta[]
  selectedVideo: string
  onVideoChange: (id: string) => void
  scope: SearchScope
  onScopeChange: (scope: SearchScope) => void
  filterObjects: string
  onFilterObjectsChange: (val: string) => void
  objectSuggestions: string[]
}
```

- [ ] **Update the function signature** to accept `objectSuggestions`:

```typescript
export function FilterPanel({
  videos, selectedVideo, onVideoChange,
  scope, onScopeChange,
  filterObjects, onFilterObjectsChange,
  objectSuggestions,
}: FilterPanelProps) {
```

- [ ] **Replace the filter input section** — find the block that renders the `Input` for filter and replace it:

Old block (inside the collapsible div):
```tsx
          <div>
            <label className="block text-xxs font-medium uppercase tracking-wide text-subtle mb-1.5">
              Filter by keyword
            </label>
            <Input
              placeholder="person, knife, cutting board"
              value={filterObjects}
              onChange={(e) => onFilterObjectsChange(e.target.value)}
              className="h-8 text-xs"
            />
            <p className="mt-1 text-xxs text-subtle">
              Matches words in the lazy-generated caption for each frame.
            </p>
          </div>
```

New block:
```tsx
          <div>
            <label className="block text-xxs font-medium uppercase tracking-wide text-subtle mb-1.5">
              Filter by keyword
            </label>
            <input
              list="filter-objects-list"
              placeholder="person, knife, cutting board"
              value={filterObjects}
              onChange={(e) => onFilterObjectsChange(e.target.value)}
              className="h-8 w-full rounded-md border border-border bg-surface px-3 text-xs text-fg placeholder:text-dim hover:border-border-strong focus:outline-none focus:border-accent/50 focus:ring-2 focus:ring-accent-ring transition-colors"
            />
            <datalist id="filter-objects-list">
              {objectSuggestions.map((obj) => (
                <option key={obj} value={obj} />
              ))}
            </datalist>
            <p className="mt-1 text-xxs text-subtle">
              Matches words in the lazy-generated caption for each frame.
            </p>
          </div>
```

- [ ] **Also update the video select** to show title:

Find `{v.video_id}` inside the `<option>` tag and replace it:
```tsx
                {videos.map((v) => (
                  <option key={v.video_id} value={v.video_id}>
                    {v.title ?? v.video_id}
                  </option>
                ))}
```

- [ ] **Commit**

```bash
git add frontend/src/components/FilterPanel.tsx
git commit -m "feat: FilterPanel adds datalist suggestions and title in video select"
```

---

## Task 18: Update `frontend/src/pages/SearchPage.tsx`

**Files:** Modify `frontend/src/pages/SearchPage.tsx`

Three additions: `videoTitles` map passed to `ResultCard`, `score_range` from query response, and search history chips.

- [ ] **Add imports at the top**:

Add after the existing imports:
```typescript
import { useSearchHistory } from '@/hooks/useSearchHistory'
import type { ScoreRange } from '@/lib/api'
```

- [ ] **Add state and hook** inside `SearchPage()`, after the existing `useState` declarations:

```typescript
  const [scoreRange, setScoreRange] = useState<ScoreRange | null>(null)
  const { history, push: pushHistory, remove: removeHistory } = useSearchHistory()
  const [inputFocused, setInputFocused] = useState(false)
```

- [ ] **Build `videoTitles` map** — add after the `videos` state:

```typescript
  const videoTitles = Object.fromEntries(
    videos.filter((v) => v.title).map((v) => [v.video_id, v.title!])
  )
```

- [ ] **Build `objectSuggestions`** — add after `videoTitles`:

```typescript
  const objectSuggestions = scope === 'video'
    ? (videos.find((v) => v.video_id === selectedVideo)?.top_objects ?? [])
    : [...new Set(videos.flatMap((v) => v.top_objects ?? []))].slice(0, 20)
```

- [ ] **Update `handleSearch`** to save to history and capture `score_range`:

Replace the `setHits(res.hits)` line inside the try block with:
```typescript
      setHits(res.hits)
      setScoreRange(res.score_range ?? null)
      if (!isChain && effectiveQuery) {
        pushHistory(effectiveQuery, mode)
      }
```

- [ ] **Pass `scoreRange` to `ResultCard`** — find the `<ResultCard ... />` render and update:

```tsx
              {hits.map((hit, i) => (
                <ResultCard
                  key={`${hit.video_id}-${hit.start}-${i}`}
                  hit={hit}
                  index={i}
                  title={videoTitles[hit.video_id]}
                  scoreRange={scoreRange}
                />
              ))}
```

- [ ] **Pass `objectSuggestions` to `FilterPanel`**:

Find `<FilterPanel` and add the prop:
```tsx
        <FilterPanel
          videos={videos}
          selectedVideo={selectedVideo}
          onVideoChange={setSelectedVideo}
          scope={scope}
          onScopeChange={setScope}
          filterObjects={filterObjects}
          onFilterObjectsChange={setFilterObjects}
          objectSuggestions={objectSuggestions}
        />
```

- [ ] **Add history chips** — find the search input `<div className="relative flex-1">` wrapper and add focus/blur handlers to the `<input>` element:

```tsx
                onFocus={() => setInputFocused(true)}
                onBlur={() => setTimeout(() => setInputFocused(false), 150)}
```

Then add history chips directly below the search input div (after the closing `</div>` of the flex gap-2 row containing the input and Search button):

```tsx
        {inputFocused && history.length > 0 && !query && (
          <div className="flex flex-wrap gap-1.5">
            {history.map((entry) => (
              <button
                key={`${entry.query}-${entry.mode}`}
                onMouseDown={(e) => {
                  e.preventDefault()
                  setQuery(entry.query)
                  setMode(entry.mode)
                  setTimeout(() => handleSearch(), 0)
                }}
                className="group inline-flex items-center gap-1.5 rounded-full border border-border bg-surface px-2.5 py-1 text-xs text-muted hover:border-border-strong hover:text-fg transition-colors"
              >
                <span>{entry.query}</span>
                <span className="text-[10px] text-dim">{entry.mode}</span>
                <span
                  onMouseDown={(e) => { e.stopPropagation(); removeHistory(entry.query, entry.mode) }}
                  className="ml-0.5 text-dim hover:text-red-400 cursor-pointer"
                >
                  ×
                </span>
              </button>
            ))}
          </div>
        )}
```

- [ ] **Commit**

```bash
git add frontend/src/pages/SearchPage.tsx
git commit -m "feat: SearchPage adds history chips, score normalization, object suggestions"
```

---

## Task 19: Update `frontend/src/components/VideoLibrary.tsx`

**Files:** Modify `frontend/src/components/VideoLibrary.tsx`

Add title column, source URL link, and Re-index context button.

- [ ] **Add `onReindex` prop** to the interface:

```typescript
interface VideoLibraryProps {
  videos: VideoMeta[]
  loading: boolean
  onSelect?: (videoId: string) => void
  onDeleted?: (videoId: string) => void
  onReindex?: (videoId: string) => void
}
```

- [ ] **Add `reindexing` state** inside `VideoLibrary`:

```typescript
  const [reindexing, setReindexing] = useState<string | null>(null)
```

- [ ] **Add `handleReindex` function**:

```typescript
  const handleReindex = async (videoId: string) => {
    setReindexing(videoId)
    try {
      await api.buildContexts([videoId])
    } finally {
      setReindexing(null)
    }
  }
```

- [ ] **Update the table `<thead>`** — replace the `Video` column header text with `Title / ID`.

- [ ] **Replace the `video_id` cell** in the row render with title + id + source URL + re-index button:

Find this block inside the `<tbody>` row:
```tsx
                <td className="px-4 py-3">
                  <span className="font-mono text-sm text-fg">{v.video_id}</span>
                </td>
```

Replace with:
```tsx
                <td className="px-4 py-3">
                  <div className="flex flex-col gap-0.5">
                    <span className="text-sm font-medium text-fg truncate max-w-xs">
                      {v.title ?? v.video_id}
                    </span>
                    {v.title && (
                      <span className="font-mono text-[10px] text-dim">{v.video_id}</span>
                    )}
                    {v.source_url && (
                      <a href={v.source_url} target="_blank" rel="noreferrer noopener"
                         onClick={(e) => e.stopPropagation()}
                         className="inline-flex items-center gap-1 text-[10px] text-subtle hover:text-fg transition-colors">
                        Source <ExternalLink className="h-2.5 w-2.5" />
                      </a>
                    )}
                  </div>
                </td>
```

- [ ] **Add Re-index button** to the actions cell — inside the `<div className="flex items-center justify-end gap-3">`, add before the delete button:

```tsx
                    <button
                      onClick={(e) => { e.stopPropagation(); handleReindex(v.video_id) }}
                      disabled={reindexing === v.video_id}
                      title="Rebuild context index"
                      className="rounded-md p-1 text-subtle hover:text-fg hover:bg-surface2 transition-colors disabled:opacity-40"
                    >
                      {reindexing === v.video_id
                        ? <Spinner size="sm" />
                        : <RefreshCw className="h-3.5 w-3.5" />}
                    </button>
```

- [ ] **Add `RefreshCw` to imports** at the top (it's already imported in `LibraryPage.tsx` but not in `VideoLibrary.tsx`):

Change the lucide import line to:
```typescript
import { ExternalLink, RefreshCw, Trash2 } from 'lucide-react'
```

- [ ] **Pass `onReindex` in `LibraryPage.tsx`** — find `<VideoLibrary` and add the prop:

```tsx
      <VideoLibrary
        videos={videos}
        loading={loading}
        onSelect={(id) => navigate(`/?video=${encodeURIComponent(id)}`)}
        onDeleted={() => fetchVideos()}
        onReindex={(id) => console.log('reindex', id)}
      />
```

- [ ] **Commit**

```bash
git add frontend/src/components/VideoLibrary.tsx frontend/src/pages/LibraryPage.tsx
git commit -m "feat: VideoLibrary shows title, source URL, and Re-index context button"
```

---

## Task 20: Final end-to-end verification

- [ ] **Start backend**

```bash
cd backend && uvicorn app:app --reload --port 8000
```

- [ ] **Start frontend**

```bash
cd frontend && npm run dev
```

- [ ] **Verify in browser:**

1. Open `http://localhost:5173` — page loads, no console errors
2. Click "Add video", paste a YouTube URL → modal shows stage checklist advancing through three stages → "Done" on completion
3. After ingest, search with text/visual/action mode → results appear with titles (not raw IDs) on cards
4. Scores show meaningful spread (best result ≈ 100%, lowest ≈ 0%)
5. Focus search box with empty query → history chips appear if previous searches exist
6. Open Library page → title and source URL visible on each row; Re-index button spins then resolves
7. Open Filters → object filter shows autocomplete suggestions from the video's indexed objects
8. Delete a video → row disappears, FAISS cache is evicted (no stale results)

- [ ] **Commit any remaining fixes**, then tag the work done

```bash
git add -p  # stage any last fixes
git commit -m "fix: final integration tweaks"
```
