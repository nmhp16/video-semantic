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
    conn.commit()
    conn.close()


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
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
    exts = [".faiss", ".json", ".svfaiss", ".saclip.faiss", ".xaclip.faiss"]
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
