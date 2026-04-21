from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np, json, os
from sentence_transformers import SentenceTransformer
from typing import Optional
from store import load_index, load_visual_index, load_action_clips_index, build_video_context
from models import VideoIngestRequest, UnifiedSearchRequest, UnifiedSearchHit, UnifiedSearchResponse
from utils_unified import extract_video_id

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

_data_dir = os.path.join(os.path.dirname(__file__), "data")
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
    if not os.path.isdir(idx_dir):
        return []
    return [fn[:-len(ext)] for fn in os.listdir(idx_dir) if fn.endswith(ext)]

def have_indexes(video_id: str, need_text=False, need_visual=False, need_action=False) -> bool:
    idx_dir = os.path.join(_data_dir, "indexes")
    checks = []
    if need_text:   checks.append(f"{video_id}.faiss")
    if need_visual: checks.append(f"{video_id}.vfaiss")
    if need_action: checks.append(f"{video_id}.aclip.faiss")
    return all(os.path.exists(os.path.join(idx_dir, f)) for f in checks)

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
async def list_videos():
    idx_dir = os.path.join(_data_dir, "indexes")
    if not os.path.isdir(idx_dir):
        return {"videos": []}
    vids = set()
    for fn in os.listdir(idx_dir):
        for ext in (".faiss", ".vfaiss", ".aclip.faiss"):
            if fn.endswith(ext):
                vids.add(fn[:-len(ext)])
    return {"videos": sorted([{
        "video_id":          vid,
        "has_text_search":   os.path.exists(os.path.join(idx_dir, f"{vid}.faiss")),
        "has_visual_search": os.path.exists(os.path.join(idx_dir, f"{vid}.vfaiss")),
        "has_action_search": os.path.exists(os.path.join(idx_dir, f"{vid}.aclip.faiss")),
    } for vid in vids], key=lambda x: x["video_id"])}


@app.post("/ingest")
async def ingest_video(request: VideoIngestRequest):
    try:
        video_url = request.video_url
        video_id  = request.video_id or extract_video_id(video_url)
        if not video_id:
            raise ValueError("Could not extract video ID from URL")

        print(f"Ingesting video: {video_id} from {video_url}")

        if os.path.exists(os.path.join(_data_dir, "indexes", f"{video_id}.faiss")):
            return {"success": True, "video_id": video_id, "status": "already_exists"}

        from ingest import ingest as do_ingest
        from visual_ingest import ingest_visual as do_visual_ingest
        do_visual_ingest(video_url)  # download video → extract frames → visual + action indexes
        do_ingest(video_url)         # transcribe audio → text index
        build_video_context(video_id)

        return {"success": True, "video_id": video_id, "status": "completed"}
    except Exception as e:
        print(f"Ingestion error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


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
        if body.ingest_if_needed and not have_indexes(vid, **need):
            ensure_ingested(body.video_url or "", vid, **need)

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
