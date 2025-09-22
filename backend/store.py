import os, sqlite3, faiss, numpy as np, json

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

# Global index paths
GLOBAL_ACLIP_PATH = os.path.join(DATA, "indexes", "_global_aclips.faiss")

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
            objects TEXT
        )             
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS visual_clips(
            video_id TEXT,
            idx INTEGER,
            start REAL,
            end REAL,
            objects TEXT
        )
    """)
    return conn

def clear_video(video_id: str):
    conn = get_conn()
    conn.execute("DELETE FROM chunks WHERE video_id=?", (video_id,))
    conn.execute("DELETE FROM visual_chunks WHERE video_id=?", (video_id,))
    conn.execute("DELETE FROM visual_clips  WHERE video_id=?", (video_id,))
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
    vecs = _safe_l2norm(embeddings)
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

# --- GLOBAL ACTION CLIP INDEX ---
# --- Deterministic 64-bit ids for clips: gid = crc32(video_id)<<32 | clip_idx
import zlib
def _vid_crc(video_id: str) -> int:
    return zlib.crc32(video_id.encode("utf-8")) & 0xffffffff

def pack_gid(video_id: str, clip_idx: int) -> int:
    crc = _vid_crc(video_id)  # 32-bit
    # Use 31 bits for crc, 32 bits for idx → stays under 2^63
    return ( (crc & 0x7fffffff) << 32 ) | (clip_idx & 0xffffffff)

def unpack_gid(gid: int) -> tuple[int, int]:
    return (gid >> 32) & 0xffffffff, gid & 0xffffffff  # (vid_crc, clip_idx)

# --- Map crc->video_id (so we can go from gid back to the actual video_id)
def ensure_meta_tables(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vid_crc(
            video_id TEXT PRIMARY KEY,
            crc INTEGER UNIQUE
        )
    """)

def upsert_vid_crc(video_id: str):
    conn = get_conn()
    ensure_meta_tables(conn)
    crc = _vid_crc(video_id)
    try:
        conn.execute("INSERT OR IGNORE INTO vid_crc(video_id, crc) VALUES(?, ?)", (video_id, crc))
        # If another video already used this CRC (collision), fall back to replacing the mapping.
        if conn.total_changes == 0:
            conn.execute("UPDATE vid_crc SET video_id=? WHERE crc=?", (video_id, crc))
        conn.commit()
    finally:
        conn.close()


def video_id_from_crc(crc: int) -> str | None:
    conn = get_conn()
    row = conn.execute("SELECT video_id FROM vid_crc WHERE crc=?", (int(crc),)).fetchone()
    conn.close()
    return row[0] if row else None

# --- Global action index helpers ---
def _load_or_create_global_index(d: int):
    if os.path.exists(GLOBAL_ACLIP_PATH):
        idx = faiss.read_index(GLOBAL_ACLIP_PATH)

        # If dim mismatches, start fresh (better than crashing)
        if idx.d != d:
            # WARNING: dropping old file to avoid segfaults / corruption
            os.remove(GLOBAL_ACLIP_PATH)
            base = faiss.IndexFlatIP(d)
            return faiss.IndexIDMap2(base)

        # Ensure it supports add_with_ids
        if not isinstance(idx, (faiss.IndexIDMap, faiss.IndexIDMap2)):
            idx = faiss.IndexIDMap2(idx)
        return idx

    base = faiss.IndexFlatIP(d)
    return faiss.IndexIDMap2(base)

def _safe_l2norm(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32", copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / (norms + 1e-12)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite values in embeddings after normalization")
    return x

def append_global_action_index(video_id: str, clip_embeddings: np.ndarray):
    """
    Add this video's action clip vectors to the global index, with stable gids.
    """
    d = clip_embeddings.shape[1]
    index = _load_or_create_global_index(d)
    vecs = _safe_l2norm(clip_embeddings)

    # assign gids
    n = vecs.shape[0]
    gids = np.array([pack_gid(video_id, i) for i in range(n)], dtype="int64")

    # add
    vecs = np.ascontiguousarray(vecs, dtype=np.float32)
    gids = np.ascontiguousarray(gids, dtype=np.int64)

    index.add_with_ids(vecs, gids)
    faiss.write_index(index, GLOBAL_ACLIP_PATH)
    upsert_vid_crc(video_id)

def load_global_action_index():
    if not os.path.exists(GLOBAL_ACLIP_PATH):
        raise FileNotFoundError("Global action index not found")
    idx = faiss.read_index(GLOBAL_ACLIP_PATH)
    return idx