import os, sqlite3, faiss, numpy as np, json
from typing import Optional, List, Tuple, Dict
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import re

_nlp = None
def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["ner"]) 
    return _nlp

# --- Paths ---
BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "data")
os.makedirs(os.path.join(DATA, "indexes"), exist_ok=True)
os.makedirs(os.path.join(DATA, "transcripts"), exist_ok=True)

DB_PATH = os.path.join(DATA, "indexes", "meta.sqlite")

# Text (ASR) index paths
INDEX_PATH = lambda vid: os.path.join(DATA, "indexes", f"{vid}.faiss")
META_PATH  = lambda vid: os.path.join(DATA, "indexes", f"{vid}.json")

# Visual (frames) index paths
VINDEX_PATH = lambda vid: os.path.join(DATA, "indexes", f"{vid}.vfaiss")

# Action index paths
ACLIP_PATH = lambda vid: os.path.join(DATA, "indexes", f"{vid}.aclip.faiss")

# Initialize embedding model for video context
EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# --- SQLite helpers ---
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks(
            video_id TEXT,
            idx INTEGER,
            start REAL,
            end REAL,
            text TEXT
        )
        """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS visual_chunks(
            video_id TEXT,
            idx INTEGER,
            start REAL,
            end REAL,
            frame TEXT,
            objects TEXT,
            UNIQUE(video_id, idx) ON CONFLICT REPLACE
        )             
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS visual_clips(
            video_id TEXT,
            idx INTEGER,
            start REAL,
            end REAL,
            objects TEXT,
            UNIQUE(video_id, idx) ON CONFLICT REPLACE
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS video_context(
            video_id TEXT PRIMARY KEY,
            title TEXT,
            source_url TEXT,
            summary TEXT,
            topics TEXT,
            objects_topk TEXT,
            actions_topk TEXT,
            lang TEXT,
            emb BLOB,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )             
    """)
    
    # Create index for video_context lookups
    conn.execute("CREATE INDEX IF NOT EXISTS idx_video_context_topics ON video_context (video_id)")
    return conn

def clear_video(video_id: str):
    conn = get_conn()
    conn.execute("DELETE FROM chunks WHERE video_id=?", (video_id,))
    conn.execute("DELETE FROM visual_chunks WHERE video_id=?", (video_id,))
    conn.execute("DELETE FROM visual_clips  WHERE video_id=?", (video_id,))
    conn.execute("DELETE FROM video_context WHERE video_id=?", (video_id,))
    conn.commit()
    conn.close()

    # Remove index files
    for p in (INDEX_PATH(video_id), ACLIP_PATH(video_id), VINDEX_PATH(video_id), META_PATH(video_id)):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass

# --- TEXT (ASR) INDEX ---
def save_index(video_id: str, embeddings: np.ndarray, chunks: list[dict]):
    # FAISS index
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    vecs = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    index.add(vecs.astype('float32'))
    faiss.write_index(index, INDEX_PATH(video_id))

    # Meta json
    with open(META_PATH(video_id), "w") as f:
        json.dump({"video_id": video_id, "n": len(chunks)}, f)
    
    # SQLite rows
    conn = get_conn()
    conn.executemany(
        "INSERT INTO chunks(video_id, idx, start, end, text) VALUES(?, ?, ?, ?, ?)",
        [(video_id, i, c["start"], c["end"], c["text"]) for i, c in enumerate(chunks)]
    )
    conn.commit()
    conn.close()

def load_index(video_id: str):
    index = faiss.read_index(INDEX_PATH(video_id))
    conn = get_conn()
    rows = conn.execute("""
        SELECT idx, start, end, text 
        FROM chunks 
        WHERE video_id=?
        ORDER BY idx
    """, (video_id,)).fetchall()
    conn.close()
    return index, rows

# --- VISUAL INDEX ---
def save_visual_index(video_id: str, embeddings: np.ndarray, chunks: list[dict]):
    # FAISS index
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    vecs = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    index.add(vecs.astype('float32'))
    faiss.write_index(index, VINDEX_PATH(video_id))

    # SQLite rows
    conn = get_conn()
    conn.executemany(
        "INSERT INTO visual_chunks(video_id, idx, start, end, frame, objects) VALUES(?, ?, ?, ?, ?, ?)",
        [(video_id, i, c["start"], c["end"], c["frame"], json.dumps(c.get("objects", []))) 
        for i, c in enumerate(chunks)]
    )
    conn.commit()
    conn.close()

def load_visual_index(video_id: str):
    index = faiss.read_index(VINDEX_PATH(video_id))
    conn = get_conn()
    rows = conn.execute("""
        SELECT idx, start, end, frame, objects 
        FROM visual_chunks 
        WHERE video_id=?
        ORDER BY idx
    """, (video_id,)).fetchall()
    conn.close()
    return index, rows

# --- ACTION CLIP INDEX ---
def save_action_clips_index(video_id: str, embeddings: np.ndarray, rows: list[dict]):
    d = embeddings.shape[1]

    if os.path.exists(ACLIP_PATH(video_id)):
        idx = faiss.read_index(ACLIP_PATH(video_id))
        if idx.d != d:
            os.remove(ACLIP_PATH(video_id))

    index = faiss.IndexFlatIP(d)
    vecs = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    index.add(vecs.astype('float32'))
    faiss.write_index(index, ACLIP_PATH(video_id))

    conn = get_conn()
    conn.executemany(
        "INSERT INTO visual_clips(video_id, idx, start, end, objects) VALUES(?,?,?,?,?)",
        [(video_id, i, r["start"], r["end"], json.dumps(r.get("objects", []))) for i, r in enumerate(rows)]
    )
    conn.commit()
    conn.close()

def load_action_clips_index(video_id: str):
    index = faiss.read_index(ACLIP_PATH(video_id))
    conn = get_conn()
    rows = conn.execute("""
        SELECT idx, start, end, objects 
        FROM visual_clips 
        WHERE video_id=?
        ORDER BY idx
    """, (video_id,)).fetchall()
    conn.close()
    return index, rows

# --- VIDEO CONTEXT FUNCTIONS ---

def derive_text_summary_for(video_id: str, conn: sqlite3.Connection) -> str:
    """Derive a short summary from ASR text for this video."""
    rows = conn.execute("""
        SELECT text FROM chunks 
        WHERE video_id=? 
        ORDER BY start
        LIMIT 5
    """, (video_id,)).fetchall()
    
    if not rows:
        return ""
    
    # Take first few chunks and concatenate
    texts = [row[0] for row in rows if row[0] and row[0].strip()]
    combined = " ".join(texts)
    
    # Simple truncation to keep summary reasonable
    if len(combined) > 200:
        combined = combined[:200] + "..."
    
    return combined

def top_objects_for(video_id: str, conn: sqlite3.Connection, k: int = 20) -> Dict[str, int]:
    """Get top K objects from visual clips for this video."""
    rows = conn.execute("""
        SELECT objects FROM visual_clips 
        WHERE video_id=?
    """, (video_id,)).fetchall()
    
    if not rows:
        return {}
    
    object_counts = Counter()
    for row in rows:
        if row[0]:
            try:
                objects = json.loads(row[0])
                for obj in objects:
                    object_counts[obj] += 1
            except (json.JSONDecodeError, TypeError):
                continue
    
    return dict(object_counts.most_common(k))

def top_actions_for(video_id: str, conn: sqlite3.Connection, k: int = 20) -> Dict[str, int]:
    """Get top K action-related terms from ASR text for this video."""
    rows = conn.execute("""
        SELECT text FROM chunks 
        WHERE video_id=?
    """, (video_id,)).fetchall()
    
    if not rows:
        return {}
    
    nlp = _get_nlp()
    verb_counts = Counter()
    vo_counts = Counter()
    
    for (txt,) in rows:
        if not txt:
            continue
        doc = nlp(txt)
        for tok in doc:
            # Verbs
            if tok.pos_ == "VERB" and tok.lemma_ not in ("be", "have", "do"):
                v = tok.lemma_.lower()
                verb_counts[v] += 1

            # Capture verb + its object
            dobj = None
            for child in tok.children:
                if child.dep_ in ("dobj", "obj") and child.pos_ in ("PROPN", "NOUN", "PRON"):
                    dobj = child.lemma_.lower()
                    break
            if dobj:
                vo_counts[f"{v} {dobj}"] += 1

    # Merge, prefer verb object over verb
    out = {}
    for key, count in vo_counts.most_common(k):
        out[key] = count
    if len(out) < k:
        for verb, count in verb_counts.most_common(k - len(out)):
            out[verb] = count
    return out

def derive_topics(summary: str, objects_topk: dict, actions_topk: dict, texts: list[str] = None, topn=10):
    texts = texts or []
    corpus = [t for t in texts if t] + ([summary] if summary else [])
    topics = set()

    # seed with structured signals
    topics.update(list(objects_topk.keys())[:5])
    topics.update(list(actions_topk.keys())[:5])

    if corpus:
        vec = TfidfVectorizer(
            ngram_range=(1,2),
            min_df=2,
            max_features=1000,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
        )
        X = vec.fit_transform(corpus)
        scores = X.sum(axis=0).A1
        vocab  = np.array(vec.get_feature_names_out())
        order  = scores.argsort()[::-1]
        for term in vocab[order]:
            # skip overly generic words & those already present
            if term in topics: 
                continue
            if re.match(r"^\d+$", term):
                continue
            topics.add(term)
            if len(topics) >= topn + 10:  # allow some headroom
                break

    return list(topics)

from bertopic import BERTopic

def _fetch_texts_for_video(video_id: str, conn: sqlite3.Connection, max_chunks: int = 300) -> list[str]:
    rows = conn.execute("""
        SELECT text FROM chunks
        WHERE video_id=?
        ORDER BY start
        LIMIT ?
    """, (video_id, max_chunks)).fetchall()
    texts = [t for (t,) in rows if t and t.strip()]
    # light cleanup
    texts = [re.sub(r"\s+", " ", t).strip() for t in texts]
    return texts

def derive_topics_bertopic(texts: list[str], topn: int = 10, min_topic_size: int = 5) -> list[str]:
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return []

    # reuse your global embedding model by passing its name 
    topic_model = BERTopic(min_topic_size=min_topic_size, calculate_probabilities=False, verbose=False,
                           embedding_model="sentence-transformers/all-MiniLM-L6-v2")

    topics, _ = topic_model.fit_transform(texts)
    info = topic_model.get_topic_info()

    out = []
    # top 5 largest topics → top 3 terms each
    for _, row in info.sort_values("Count", ascending=False).head(5).iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue  # skip noise topic
        for term, _ in (topic_model.get_topic(tid) or [])[:3]:
            out.append(term)

    # dedupe & cap
    final, seen = [], set()
    for t in out:
        t = t.strip().lower()
        if t and t not in seen:
            final.append(t); seen.add(t)
        if len(final) >= topn:
            break
    return final

def build_video_context(video_id: str) -> None:
    """Build and store video context summary for better search filtering."""
    conn = get_conn()
    
    try:
        # 1) Derive a short summary from ASR
        summary = derive_text_summary_for(video_id, conn)
        
        # 2) Get frequent objects/actions
        objects_topk = top_objects_for(video_id, conn, k=20)
        actions_topk = top_actions_for(video_id, conn, k=20)
        
        # 3) Derive topics
        texts = _fetch_texts_for_video(video_id, conn, max_chunks=300)
        if len(texts) < 10:
            topics = derive_topics(summary, objects_topk, actions_topk, texts, topn=10)
        else:
            topics = derive_topics_bertopic(texts, topn=10, min_topic_size=5)

        fused_topics = []
        seen = set()
        for t in (list(objects_topk.keys())[:5] + list(actions_topk.keys())[:5] + topics):
            t0 = t.strip().lower()
            if t0 and t0 not in seen:
                fused_topics.append(t0); seen.add(t0)
        
        # 4) Create embedding
        text_for_emb = summary or " ".join(fused_topics) or ""
        if text_for_emb.strip():
            emb = EMB.encode([text_for_emb], normalize_embeddings=True).astype("float32")[0]
            emb_blob = emb.tobytes()
        else:
            emb_blob = None
        
        # 5) Store in database
        conn.execute("""
            INSERT INTO video_context (video_id, title, source_url, summary, topics,
                                     objects_topk, actions_topk, lang, emb, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(video_id) DO UPDATE SET
              summary=excluded.summary,
              topics=excluded.topics,
              objects_topk=excluded.objects_topk,
              actions_topk=excluded.actions_topk,
              emb=excluded.emb,
              updated_at=CURRENT_TIMESTAMP
        """, (
            video_id,
            None,  # title - can be set later
            None,  # source_url - can be set later
            summary,
            json.dumps(fused_topics),
            json.dumps(objects_topk),
            json.dumps(actions_topk),
            "en",  # lang - could detect later
            emb_blob
        ))
        conn.commit()
        print(f"Built video context for {video_id}: {len(fused_topics)} topics, {len(objects_topk)} objects, {len(actions_topk)} actions")
        
    except Exception as e:
        print(f"Error building video context for {video_id}: {e}")
    finally:
        conn.close()

def _fetch_contexts(video_ids: Optional[List[str]] = None) -> List[Dict]:
    """Fetch video contexts from database."""
    conn = get_conn()
    try:
        if video_ids:
            qmarks = ",".join(["?"] * len(video_ids))
            rows = conn.execute(f"""
                SELECT video_id, topics, objects_topk, actions_topk, emb
                FROM video_context
                WHERE video_id IN ({qmarks})
            """, video_ids).fetchall()
        else:
            rows = conn.execute("""
                SELECT video_id, topics, objects_topk, actions_topk, emb
                FROM video_context
            """).fetchall()
        
        contexts = []
        for vid, topics, objs, acts, emb in rows:
            contexts.append({
                "video_id": vid,
                "topics": json.loads(topics or "[]"),
                "objects_topk": json.loads(objs or "{}"),
                "actions_topk": json.loads(acts or "{}"),
                "emb": np.frombuffer(emb, dtype=np.float32) if emb else None
            })
        return contexts
    finally:
        conn.close()

def filter_videos_by_context(query: str,
                           restrict_videos: Optional[List[str]] = None,
                           topn: int = 50,
                           min_cos: float = 0.28) -> List[str]:
    """Filter videos by context similarity before detailed search."""
    # 1) Embed the query once
    qv = EMB.encode([query], normalize_embeddings=True).astype("float32")[0]
    
    # 2) Fetch contexts
    contexts = _fetch_contexts(restrict_videos)
    
    scored: List[Tuple[str, float]] = []
    for c in contexts:
        if c["emb"] is None:
            # If no embedding, give it a small score to include it
            scored.append((c["video_id"], 0.1))
            continue
            
        # Cosine similarity
        cos = float(np.dot(qv, c["emb"]))
        
        # Keyword boost if query matches topics/objects/actions
        query_lower = query.lower()
        if any(t.lower() in query_lower for t in c["topics"]):
            cos += 0.05
        if any(obj.lower() in query_lower for obj in c["objects_topk"].keys()):
            cos += 0.03
        if any(act.lower() in query_lower for act in c["actions_topk"].keys()):
            cos += 0.03
            
        scored.append((c["video_id"], cos))
    
    # 3) Keep only sufficiently relevant videos
    scored.sort(key=lambda x: x[1], reverse=True)
    keep = [vid for vid, s in scored if s >= min_cos]
    return keep[:topn]

def passes_hard_context(video_id: str, 
                       require_any: Optional[List[str]] = None, 
                       require_all: Optional[List[str]] = None) -> bool:
    """Hard verification that video context contains certain objects/actions."""
    contexts = _fetch_contexts([video_id])
    if not contexts:
        return True  # If no context, don't filter out
    
    c = contexts[0]
    # Build bag of all terms
    bag = set([*c["topics"], *c["objects_topk"].keys(), *c["actions_topk"].keys()])
    bag = {w.lower() for w in bag}
    
    if require_all and not all(w.lower() in bag for w in require_all):
        return False
    if require_any and not any(w.lower() in bag for w in require_any):
        return False
    return True
