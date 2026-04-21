import os, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from supabase_client import get_sb

FRAMES_BUCKET  = "frames"
INDEXES_BUCKET = "indexes"

def _batch(lst, n=500):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# ---- Frame images ----

def upload_frame(video_id: str, local_path: str) -> str | None:
    sb = get_sb()
    bucket_path = f"{video_id}/{os.path.basename(local_path)}"
    with open(local_path, "rb") as f:
        data = f.read()
    try:
        sb.storage.from_(FRAMES_BUCKET).upload(bucket_path, data, {"upsert": "true"})
    except Exception:
        try:
            sb.storage.from_(FRAMES_BUCKET).update(bucket_path, data)
        except Exception as e:
            print(f"    frame upload failed {os.path.basename(local_path)}: {e}")
            return None
    return sb.storage.from_(FRAMES_BUCKET).get_public_url(bucket_path)

def upload_frames_parallel(video_id: str, local_paths: list[str], workers: int = 8) -> dict[str, str]:
    """Upload frames in parallel. Returns {local_abs_path: public_url}."""
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(upload_frame, video_id, p): p for p in local_paths}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                url = fut.result()
                if url:
                    results[path] = url
            except Exception as e:
                print(f"    upload error {path}: {e}")
    return results

# ---- FAISS index files ----

def push_faiss(video_id: str, ext: str, local_path: str):
    sb = get_sb()
    with open(local_path, "rb") as f:
        data = f.read()
    bucket_path = f"{video_id}{ext}"
    try:
        sb.storage.from_(INDEXES_BUCKET).upload(bucket_path, data, {"upsert": "true"})
    except Exception:
        try:
            sb.storage.from_(INDEXES_BUCKET).update(bucket_path, data)
        except Exception as e:
            print(f"    push_faiss failed {bucket_path}: {e}")

def pull_faiss(video_id: str, ext: str, dest_path: str) -> bool:
    """Download FAISS index from Supabase Storage. Returns True if found."""
    try:
        sb = get_sb()
        data = sb.storage.from_(INDEXES_BUCKET).download(f"{video_id}{ext}")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(data)
        print(f"  pulled {video_id}{ext} from Supabase")
        return True
    except Exception:
        return False

# ---- Metadata tables ----

def push_chunks(video_id: str, chunks: list[dict]):
    sb = get_sb()
    sb.table("chunks").delete().eq("video_id", video_id).execute()
    rows = [{"video_id": video_id, "idx": i,
             "start_sec": c["start"], "end_sec": c["end"], "text": c.get("text", "")}
            for i, c in enumerate(chunks)]
    for batch in _batch(rows):
        sb.table("chunks").insert(batch).execute()

def pull_chunks(video_id: str) -> list:
    """Returns (idx, start, end, text) tuples — same format as SQLite."""
    sb = get_sb()
    res = sb.table("chunks").select("idx,start_sec,end_sec,text").eq("video_id", video_id).order("idx").execute()
    return [(r["idx"], r["start_sec"], r["end_sec"], r["text"]) for r in res.data]

def push_visual_frames(video_id: str, frames: list[dict], frame_urls: dict[str, str] | None = None):
    sb = get_sb()
    sb.table("visual_frames").delete().eq("video_id", video_id).execute()
    rows = []
    for i, f in enumerate(frames):
        url = (frame_urls or {}).get(f["frame"]) or f["frame"]
        rows.append({"video_id": video_id, "idx": i,
                     "start_sec": f["start"], "end_sec": f["end"],
                     "frame_url": url,
                     "objects": f.get("objects", []),
                     "caption": f.get("caption", "")})
    for batch in _batch(rows):
        sb.table("visual_frames").insert(batch).execute()

def pull_visual_frames(video_id: str) -> list:
    """Returns (idx, start, end, frame_url, objects_json, caption) tuples."""
    sb = get_sb()
    res = sb.table("visual_frames").select("idx,start_sec,end_sec,frame_url,objects,caption") \
            .eq("video_id", video_id).order("idx").execute()
    return [(r["idx"], r["start_sec"], r["end_sec"], r["frame_url"],
             json.dumps(r.get("objects") or []), r.get("caption") or "") for r in res.data]

def push_action_clips(video_id: str, clips: list[dict]):
    sb = get_sb()
    sb.table("action_clips").delete().eq("video_id", video_id).execute()
    rows = [{"video_id": video_id, "idx": i,
             "start_sec": c["start"], "end_sec": c["end"],
             "objects": c.get("objects", []), "caption": c.get("caption", "")}
            for i, c in enumerate(clips)]
    for batch in _batch(rows):
        sb.table("action_clips").insert(batch).execute()

def pull_action_clips(video_id: str) -> list:
    """Returns (idx, start, end, objects_json, caption) tuples."""
    sb = get_sb()
    res = sb.table("action_clips").select("idx,start_sec,end_sec,objects,caption") \
            .eq("video_id", video_id).order("idx").execute()
    return [(r["idx"], r["start_sec"], r["end_sec"],
             json.dumps(r.get("objects") or []), r.get("caption") or "") for r in res.data]

def list_video_ids() -> dict[str, dict]:
    """Returns {video_id: {has_text, has_visual, has_action}} from Supabase tables."""
    sb = get_sb()
    result: dict[str, dict] = {}
    try:
        for table, key in [("chunks", "has_text"), ("visual_frames", "has_visual"), ("action_clips", "has_action")]:
            rows = sb.table(table).select("video_id").execute().data
            for r in rows:
                vid = r["video_id"]
                if vid not in result:
                    result[vid] = {"has_text": False, "has_visual": False, "has_action": False}
                result[vid][key] = True
    except Exception as e:
        print(f"supabase list_video_ids error: {e}")
    return result
