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
    return conn

def clear_video(video_id: str):
    conn = get_conn()
    conn.execute("DELETE FROM chunks WHERE video_id=?", (video_id,))
    conn.execute("DELETE FROM visual_chunks WHERE video_id=?", (video_id,))
    conn.commit()
    conn.close()

    # Remove index files
    for p in (INDEX_PATH(video_id), VINDEX_PATH(video_id), META_PATH(video_id)):
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