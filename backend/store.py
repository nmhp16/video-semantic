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
    # Use 30 bits for crc, 32 bits for idx → safer for signed int64
    return ( (crc & 0x3fffffff) << 32 ) | (clip_idx & 0xffffffff)

def unpack_gid(gid: int) -> tuple[int, int]:
    return (gid >> 32) & 0x3fffffff, gid & 0xffffffff  # (vid_crc, clip_idx)

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
    # Store only the truncated CRC that we actually use in GIDs
    truncated_crc = crc & 0x3fffffff
    try:
        conn.execute("INSERT OR IGNORE INTO vid_crc(video_id, crc) VALUES(?, ?)", (video_id, truncated_crc))
        # If another video already used this CRC (collision), fall back to replacing the mapping.
        if conn.total_changes == 0:
            conn.execute("UPDATE vid_crc SET video_id=? WHERE crc=?", (video_id, truncated_crc))
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
        try:
            idx = faiss.read_index(GLOBAL_ACLIP_PATH)

            # If dim mismatches, start fresh (better than crashing)
            if idx.d != d:
                print(f"WARNING: Global index dimension mismatch ({idx.d} vs {d}), rebuilding")
                os.remove(GLOBAL_ACLIP_PATH)
                base = faiss.IndexFlatIP(d)
                return faiss.IndexIDMap2(base)

            # Ensure it supports add_with_ids
            if not isinstance(idx, (faiss.IndexIDMap, faiss.IndexIDMap2)):
                print(f"WARNING: Converting index to IDMap2 (was {type(idx)})")
                # Convert to IDMap2
                base = faiss.IndexFlatIP(d)
                new_idx = faiss.IndexIDMap2(base)
                # If the old index has data, we'd need to migrate, but for now just start fresh
                if hasattr(idx, 'ntotal') and idx.ntotal > 0:
                    print("WARNING: Old index has data but wrong type, starting fresh")
                    os.remove(GLOBAL_ACLIP_PATH)
                return new_idx
            
            return idx
            
        except Exception as e:
            print(f"ERROR loading global index: {e}, creating new one")
            # Remove corrupted file
            try:
                os.remove(GLOBAL_ACLIP_PATH)
            except:
                pass

    # Create fresh index
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
    if clip_embeddings.size == 0:
        print(f"WARNING: No embeddings to add for video {video_id}")
        return
        
    d = clip_embeddings.shape[1]
    
    try:
        index = _load_or_create_global_index(d)
        vecs = _safe_l2norm(clip_embeddings)

        # assign gids
        n = vecs.shape[0]
        gids = np.array([pack_gid(video_id, i) for i in range(n)], dtype="int64")

        # Ensure arrays are contiguous and proper type
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        gids = np.ascontiguousarray(gids, dtype=np.int64)

        # Validate inputs
        if not np.isfinite(vecs).all():
            raise ValueError("Non-finite values in embeddings")
        if len(gids) != len(vecs):
            raise ValueError("Mismatch between embeddings and IDs")

        # Add to index
        index.add_with_ids(vecs, gids)
        
        # Save index with proper error handling
        try:
            faiss.write_index(index, GLOBAL_ACLIP_PATH)
        except Exception as e:
            print(f"ERROR writing global index: {e}")
            # Try to recover by removing the file and retrying once
            try:
                os.remove(GLOBAL_ACLIP_PATH)
                faiss.write_index(index, GLOBAL_ACLIP_PATH)
                print("Successfully wrote index after removing old file")
            except Exception as e2:
                print(f"ERROR writing index even after cleanup: {e2}")
                raise e2
                
        upsert_vid_crc(video_id)
        print(f"Successfully added {n} clips for video {video_id} to global index")
        
    except Exception as e:
        print(f"ERROR in append_global_action_index for {video_id}: {e}")
        raise e

def rebuild_global_action_index():
    """
    Rebuild the global action index from all existing video action indexes.
    """
    print("Rebuilding global action index...")
    
    # Remove existing global index
    if os.path.exists(GLOBAL_ACLIP_PATH):
        os.remove(GLOBAL_ACLIP_PATH)
    
    # Get all video IDs that have action indexes
    videos_with_actions = []
    index_dir = os.path.join(DATA, "indexes")
    for filename in os.listdir(index_dir):
        if filename.endswith(".aclip.faiss"):
            video_id = filename.replace(".aclip.faiss", "")
            videos_with_actions.append(video_id)
    
    print(f"Found {len(videos_with_actions)} videos with action indexes")
    
    total_added = 0
    for video_id in videos_with_actions:
        try:
            # Load the individual video's action index
            aclip_path = ACLIP_PATH(video_id)
            if not os.path.exists(aclip_path):
                continue
                
            index = faiss.read_index(aclip_path)
            n = index.ntotal
            
            if n == 0:
                continue
                
            # Reconstruct embeddings from the index
            embeddings = np.zeros((n, index.d), dtype=np.float32)
            for i in range(n):
                embeddings[i] = index.reconstruct(i)
            
            # Add to global index
            append_global_action_index(video_id, embeddings)
            total_added += n
            print(f"Added {n} clips from {video_id}")
            
        except Exception as e:
            print(f"Error processing {video_id}: {e}")
            continue
    
    print(f"Global index rebuild complete: {total_added} total clips indexed")
    return total_added

def get_global_index_stats():
    """Get statistics about the global action index."""
    if not os.path.exists(GLOBAL_ACLIP_PATH):
        return {"exists": False, "total": 0, "dimension": 0}
    
    try:
        index = faiss.read_index(GLOBAL_ACLIP_PATH)
        return {
            "exists": True,
            "total": index.ntotal,
            "dimension": index.d,
            "type": type(index).__name__
        }
    except Exception as e:
        return {"exists": False, "error": str(e), "total": 0, "dimension": 0}

def load_global_action_index():
    """Load the global action index with proper error handling."""
    if not os.path.exists(GLOBAL_ACLIP_PATH):
        raise FileNotFoundError("Global action index not found")
    
    try:
        idx = faiss.read_index(GLOBAL_ACLIP_PATH)
        
        # Ensure it's an IDMap for proper search with IDs
        if not isinstance(idx, (faiss.IndexIDMap, faiss.IndexIDMap2)):
            print(f"WARNING: Global index is not IDMap type ({type(idx)}), needs rebuild")
            raise ValueError("Invalid index type - needs rebuild")
            
        # Basic validation
        if idx.ntotal == 0:
            raise ValueError("Empty global index")
            
        print(f"Loaded global action index: {idx.ntotal} clips, {idx.d}D")
        return idx
    except Exception as e:
        print(f"ERROR loading global action index: {e}")
        # Remove corrupted index to force rebuild
        try:
            os.remove(GLOBAL_ACLIP_PATH)
            print("Removed corrupted global index file")
        except:
            pass
        raise e