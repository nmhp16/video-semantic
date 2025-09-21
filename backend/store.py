import os, sqlite3, faiss, numpy as np, json

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "data")
os.makedirs(os.path.join(DATA, "indexes"), exist_ok=True)
os.makedirs(os.path.join(DATA, "transcripts"), exist_ok=True)

DB_PATH = os.path.join(DATA, "indexes", "meta.sqlite")
INDEX_PATH = lambda vid: os.path.join(DATA, "indexes", f"{vid}.faiss")
META_PATH  = lambda vid: os.path.join(DATA, "indexes", f"{vid}.json")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks(
            video_id TEXT,
            idx INTEGER,
            start REAL,
            end REAL,
            text TEXT
        )
        """)
    return conn

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