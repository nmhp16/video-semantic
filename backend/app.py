from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np, json, os, threading, time as _time
from sentence_transformers import SentenceTransformer
from typing import Optional
from store import load_index, load_visual_index, load_action_clips_index, build_video_context
from models import VideoIngestRequest, UnifiedSearchRequest, UnifiedSearchHit, UnifiedSearchResponse
from utils_unified import extract_video_id
from supabase_client import sb_enabled

# In-memory ingest job tracker  { video_id: {status, error?} }
_ingest_jobs: dict = {}
_ingest_lock = threading.Lock()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

_data_dir = os.path.join(os.path.dirname(__file__), "data")
# Only serve local frames when Supabase is not handling frame storage
if not sb_enabled():
    os.makedirs(os.path.join(_data_dir, "frames"), exist_ok=True)
    app.mount("/frames", StaticFiles(directory=os.path.join(_data_dir, "frames")), name="frames")

EMB = SentenceTransformer("BAAI/bge-small-en-v1.5")

# --------------------
# Result post-processing
# --------------------
def dedupe_hits(hits: list[dict]) -> list[dict]:
    """Keep the highest-scoring result per (video_id, time window)."""
    best = {}
    for h in hits:
        key = (h["video_id"], round(h.get("start", 0.0), 3), round(h.get("end", h.get("start", 0.0)), 3))
        if key not in best or h["score"] > best[key]["score"]:
            best[key] = h
    return sorted(best.values(), key=lambda x: (-x["score"], x.get("start", 0.0)))

def nms_time(hits: list[dict], tol: float = 0.5) -> list[dict]:
    """Non-max suppression: drop results too close in time to a higher-scoring one."""
    kept = []
    for h in sorted(hits, key=lambda x: -x["score"]):
        mid = (h.get("start", 0.0) + h.get("end", 0.0)) * 0.5
        if all(
            h["video_id"] != k["video_id"] or
            abs(mid - (k.get("start", 0.0) + k.get("end", 0.0)) * 0.5) > tol
            for k in kept
        ):
            kept.append(h)
    return kept

# --------------------
# Index helpers
# --------------------
def list_video_ids_with(ext: str) -> list[str]:
    idx_dir = os.path.join(_data_dir, "indexes")
    local = set()
    if os.path.isdir(idx_dir):
        local = {fn[:-len(ext)] for fn in os.listdir(idx_dir) if fn.endswith(ext)}
    if sb_enabled():
        try:
            from supabase_store import list_video_ids as sb_list
            key = {".faiss": "has_text", ".vfaiss": "has_visual", ".aclip.faiss": "has_action"}.get(ext)
            if key:
                local |= {vid for vid, flags in sb_list().items() if flags.get(key)}
        except Exception as e:
            print(f"supabase list error: {e}")
    return list(local)

def have_indexes(video_id: str, need_text=False, need_visual=False, need_action=False) -> bool:
    idx_dir = os.path.join(_data_dir, "indexes")
    pairs = []
    if need_text:   pairs.append((f"{video_id}.faiss",       ".faiss"))
    if need_visual: pairs.append((f"{video_id}.vfaiss",      ".vfaiss"))
    if need_action: pairs.append((f"{video_id}.aclip.faiss", ".aclip.faiss"))
    for filename, ext in pairs:
        local = os.path.join(idx_dir, filename)
        if not os.path.exists(local) and sb_enabled():
            from supabase_store import pull_faiss
            if not pull_faiss(video_id, ext, local):
                return False
        elif not os.path.exists(local):
            return False
    return True

def ensure_ingested(video_url: str, video_id: str, need_text: bool, need_visual: bool, need_action: bool):
    from ingest import ingest as do_ingest
    from visual_ingest import ingest_visual as do_visual_ingest
    idx_dir = os.path.join(_data_dir, "indexes")
    if (need_visual or need_action) and not os.path.exists(os.path.join(idx_dir, f"{video_id}.vfaiss")):
        do_visual_ingest(video_url)
    if need_text and not os.path.exists(os.path.join(idx_dir, f"{video_id}.faiss")):
        do_ingest(video_url)

# --------------------
# Per-video search functions
# --------------------
def search_text(video_id: str, q: str, k: int) -> list[dict]:
    index, rows = load_index(video_id)
    qv = EMB.encode([q], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, k)
    return [
        {"video_id": video_id, "start": float(rows[i][1]), "end": float(rows[i][2]),
         "score": float(s), "text": rows[i][3]}
        for s, i in zip(D[0], I[0]) if i != -1
    ]

def search_visual(video_id: str, q: str, k: int) -> list[dict]:
    index, rows = load_visual_index(video_id)
    qv = EMB.encode([q], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, k)
    return [
        {"video_id": video_id, "start": float(rows[i][1]), "end": float(rows[i][2]),
         "score": float(s), "frame": rows[i][3],
         "objects": json.loads(rows[i][4]) if rows[i][4] else [],
         "caption": rows[i][5] or ""}
        for s, i in zip(D[0], I[0]) if i != -1
    ]

def search_action(video_id: str, q: str, k: int) -> list[dict]:
    index, rows = load_action_clips_index(video_id)
    qv = EMB.encode([q], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, k)
    return [
        {"video_id": video_id, "start": float(rows[i][1]), "end": float(rows[i][2]),
         "score": float(s),
         "objects": json.loads(rows[i][3]) if rows[i][3] else [],
         "caption": rows[i][4] or ""}
        for s, i in zip(D[0], I[0]) if i != -1
    ]

# --------------------
# Global search (across all ingested videos)
# --------------------
_SEARCH_FN = {"text": search_text, "visual": search_visual, "action": search_action}
_EXT_MAP    = {"text": ".faiss",   "visual": ".vfaiss",     "action": ".aclip.faiss"}

def search_global(mode: str, q: str, k: int, restrict: Optional[list[str]]) -> list[dict]:
    vids = list_video_ids_with(_EXT_MAP[mode])
    if restrict:
        vids = [v for v in vids if v in restrict]
    all_hits = []
    for vid in vids:
        try:
            all_hits.extend(_SEARCH_FN[mode](vid, q, k))
        except Exception as e:
            print(f"[{mode} global] skip {vid}: {e}")
    return sorted(all_hits, key=lambda h: -h["score"])

# --------------------
# Endpoints
# --------------------
@app.get("/videos")
def list_videos():
    idx_dir = os.path.join(_data_dir, "indexes")
    # Local index files
    meta: dict[str, dict] = {}
    if os.path.isdir(idx_dir):
        for fn in os.listdir(idx_dir):
            for ext, key in ((".faiss", "has_text_search"), (".vfaiss", "has_visual_search"), (".aclip.faiss", "has_action_search")):
                if fn.endswith(ext):
                    vid = fn[:-len(ext)]
                    if vid not in meta:
                        meta[vid] = {"has_text_search": False, "has_visual_search": False, "has_action_search": False}
                    meta[vid][key] = True
    # Merge Supabase
    if sb_enabled():
        try:
            from supabase_store import list_video_ids as sb_list
            for vid, flags in sb_list().items():
                if vid not in meta:
                    meta[vid] = {"has_text_search": False, "has_visual_search": False, "has_action_search": False}
                if flags.get("has_text"):   meta[vid]["has_text_search"]   = True
                if flags.get("has_visual"): meta[vid]["has_visual_search"] = True
                if flags.get("has_action"): meta[vid]["has_action_search"] = True
        except Exception as e:
            print(f"supabase /videos error: {e}")
    return {"videos": sorted([{"video_id": vid, **flags} for vid, flags in meta.items()], key=lambda x: x["video_id"])}


def _is_indexed(video_id: str) -> dict[str, bool]:
    """Check which indexes exist (local + Supabase)."""
    idx_dir = os.path.join(_data_dir, "indexes")
    def local(ext): return os.path.exists(os.path.join(idx_dir, f"{video_id}{ext}"))
    has_text   = local(".faiss")
    has_visual = local(".vfaiss")
    has_action = local(".aclip.faiss")
    if sb_enabled() and not (has_text and has_visual and has_action):
        try:
            from supabase_store import list_video_ids as sb_list
            sb = sb_list()
            if video_id in sb:
                flags = sb[video_id]
                has_text   = has_text   or flags.get("has_text", False)
                has_visual = has_visual or flags.get("has_visual", False)
                has_action = has_action or flags.get("has_action", False)
        except Exception:
            pass
    return {"text": has_text, "visual": has_visual, "action": has_action}


def _fake_processing(video_id: str, delay: float = 15.0):
    """Simulate processing for already-indexed videos (demo effect)."""
    _time.sleep(delay)
    with _ingest_lock:
        _ingest_jobs[video_id] = {"status": "completed"}


def _run_ingest_job(video_url: str, video_id: str):
    from ingest import ingest as do_ingest
    from visual_ingest import ingest_visual as do_visual_ingest
    try:
        idx_dir = os.path.join(_data_dir, "indexes")
        def has_local(ext): return os.path.exists(os.path.join(idx_dir, f"{video_id}{ext}"))

        segments = None
        if not has_local(".vfaiss") or not has_local(".aclip.faiss"):
            segments = do_visual_ingest(video_url)
        if not has_local(".faiss"):
            do_ingest(video_url, segments=segments)
        build_video_context(video_id)

        with _ingest_lock:
            _ingest_jobs[video_id] = {"status": "completed"}
        print(f"INGEST DONE: {video_id}")
    except Exception as e:
        with _ingest_lock:
            _ingest_jobs[video_id] = {"status": "failed", "error": str(e)}
        print(f"INGEST FAILED: {video_id}: {e}")


@app.post("/ingest")
def ingest_video(request: VideoIngestRequest, background_tasks: BackgroundTasks):
    video_url = request.video_url
    video_id  = request.video_id or extract_video_id(video_url)
    if not video_id:
        raise HTTPException(400, "Could not extract video ID from URL")

    with _ingest_lock:
        job = _ingest_jobs.get(video_id)
    if job and job["status"] == "processing":
        return {"success": True, "video_id": video_id, "status": "processing"}

    indexed = _is_indexed(video_id)
    if indexed["text"] and indexed["visual"] and indexed["action"]:
        # Already done — show fake 15s progress for demo
        with _ingest_lock:
            _ingest_jobs[video_id] = {"status": "processing"}
        background_tasks.add_task(_fake_processing, video_id, 15.0)
        return {"success": True, "video_id": video_id, "status": "processing"}

    with _ingest_lock:
        _ingest_jobs[video_id] = {"status": "processing"}
    background_tasks.add_task(_run_ingest_job, video_url, video_id)
    return {"success": True, "video_id": video_id, "status": "processing"}


@app.get("/ingest/status/{video_id}")
def get_ingest_status(video_id: str):
    with _ingest_lock:
        job = _ingest_jobs.get(video_id)
    if job is None:
        return {"video_id": video_id, "status": "unknown"}
    return {"video_id": video_id, **job}


@app.post("/query", response_model=UnifiedSearchResponse)
def unified_query(body: UnifiedSearchRequest):
    scope = (body.scope or "video").lower()
    mode  = body.mode
    q     = body.query or ""
    k     = body.k

    if mode not in _SEARCH_FN:
        raise HTTPException(400, f"Unknown mode '{mode}'. Use: text, visual, action")

    # ---- single video ----
    if scope == "video":
        vid = body.video_id or extract_video_id(body.video_url or "")
        if not vid:
            raise HTTPException(400, "video_id or video_url required for scope='video'")

        need = dict(need_text=mode == "text", need_visual=mode == "visual", need_action=mode == "action")
        if not have_indexes(vid, **need):
            job = _ingest_jobs.get(vid, {})
            if job.get("status") == "processing":
                raise HTTPException(503, f"Video '{vid}' is still being indexed — try again in a moment")
            if body.ingest_if_needed:
                ensure_ingested(body.video_url or "", vid, **need)
            else:
                ext = {"text": ".faiss", "visual": ".vfaiss", "action": ".aclip.faiss"}[mode]
                raise HTTPException(404, f"No {mode} index for '{vid}' ({ext} not found)")

        raw = _SEARCH_FN[mode](vid, q, k)

    # ---- all ingested videos ----
    elif scope == "global":
        raw = search_global(mode, q, k, body.videos)

    else:
        raise HTTPException(400, f"Unknown scope '{scope}'. Use: video, global")

    raw = dedupe_hits(raw)
    raw = nms_time(raw)[:k]
    return UnifiedSearchResponse(
        video_id=body.video_id if scope == "video" else None,
        mode=mode,
        hits=[UnifiedSearchHit(
            video_id=h["video_id"],
            start=float(h.get("start", 0.0)),
            end=float(h.get("end", 0.0)),
            score=float(h.get("score", 0.0)),
            text=h.get("text"),
            frame=h.get("frame"),
            objects=h.get("objects"),
            caption=h.get("caption"),
        ) for h in raw],
    )
