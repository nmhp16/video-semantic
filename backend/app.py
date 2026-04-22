from fastapi import FastAPI, Query, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np, json, os, re, logging
from urllib.parse import urlparse, parse_qs
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict, Any
from store import (
    load_index, db,
    load_siglip_visual_index, load_siglip_action_clips_index,
    get_cached_captions, put_cached_captions,
    filter_videos_by_context, build_video_context,
    clear_video,
)
from models import (
    SearchResponse, SearchHit, VideoIngestRequest, OVVerifyRequest,
    UnifiedSearchRequest, UnifiedSearchHit, UnifiedSearchResponse,
    MAX_K,
)
from utils_unified import extract_video_id
from gdino import detect_on_image

logger = logging.getLogger(__name__)

app = FastAPI()

# CORS: comma-separated origins in CORS_ORIGINS, falling back to common
# local-dev ports. Wildcard is incompatible with allow_credentials, so we
# only enable credentials when an explicit allowlist is configured.
_cors_raw = os.environ.get("CORS_ORIGINS", "").strip()
if _cors_raw:
    _cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
else:
    _cors_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_BASE_DIR = os.path.dirname(__file__)
_data_dir = os.path.join(_BASE_DIR, "data")
_frames_dir = os.path.join(_data_dir, "frames")
_media_dir = os.path.join(_data_dir, "media")
_indexes_dir = os.path.join(_data_dir, "indexes")
os.makedirs(_frames_dir, exist_ok=True)
os.makedirs(_media_dir, exist_ok=True)
app.mount("/frames", StaticFiles(directory=_frames_dir), name="frames")
app.mount("/media", StaticFiles(directory=_media_dir), name="media")

EMB = SentenceTransformer("BAAI/bge-small-en-v1.5")

# SigLIP text tower — loaded lazily so /search and /videos aren't slowed at startup
_SIGLIP = None
def _get_siglip():
    global _SIGLIP
    if _SIGLIP is None:
        from visual_ingest import SigLIPEncoder
        _SIGLIP = SigLIPEncoder("google/siglip-base-patch16-224")
    return _SIGLIP

# Florence-2 captioner — loaded lazily the first time a query needs captions.
_CAPTIONER = None
def _get_captioner():
    global _CAPTIONER
    if _CAPTIONER is None:
        from visual_ingest import Florence2Captioner
        _CAPTIONER = Florence2Captioner("microsoft/Florence-2-base")
    return _CAPTIONER

def _caption_hits_lazy(video_id: Optional[str], hits: list[dict]) -> None:
    """Fill in 'caption' and 'objects' on each hit dict in place.
    Reads from the sqlite caption_cache first; runs Florence-2 only for frames
    that aren't cached yet, then writes the new captions back to the cache.
    Pass video_id=None for mixed-video (global-scope) lists; the function then
    groups hits by hit['video_id']."""
    if not hits:
        return

    # Group hits by video_id so cache reads/writes are per-video
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
                abs_path = frame if os.path.isabs(frame) else os.path.join(_BASE_DIR, frame)
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

# --------------------
# Utilities
# --------------------
def _val(h, name, default=None):
    if isinstance(h, dict):
        return h.get(name, default)
    return getattr(h, name, default)

def dedupe_hits(hits, prefer="max", key_mode="auto"):
    """
    Collapse duplicates:
      - if a 'frame' field exists -> group by (video_id, frame)
      - else -> group by (video_id, rounded start/end)
    keep the max score within each group.
    """
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
        return (float(_val(h,"start",0.0)) + float(_val(h,"end",0.0))) * 0.5

    for h in sorted_hits:
        m, vid = mid(h), _val(h, "video_id")
        if all(not (vid == _val(k,"video_id") and abs(m - mid(k)) <= tol) for k in kept):
            kept.append(h)
    return kept


def have_indexes(video_id: str, need_text=False, need_visual=False, need_action=False) -> bool:
    ok = True
    if need_text:
        ok &= os.path.exists(os.path.join(_indexes_dir, f"{video_id}.faiss"))
    if need_visual:
        ok &= os.path.exists(os.path.join(_indexes_dir, f"{video_id}.svfaiss"))
    if need_action:
        ok &= os.path.exists(os.path.join(_indexes_dir, f"{video_id}.saclip.faiss"))
    return ok

def ensure_ingested(video_url: str, video_id: str, need_text: bool, need_visual: bool, need_action: bool):
    from ingest import ingest as do_ingest
    from visual_ingest import ingest_visual as do_visual_ingest
    need_visual_or_action = need_visual or need_action
    if need_text and not os.path.exists(os.path.join(_indexes_dir, f"{video_id}.faiss")):
        do_ingest(video_url)
    if need_visual_or_action and not os.path.exists(os.path.join(_indexes_dir, f"{video_id}.svfaiss")):
        do_visual_ingest(video_url)

def _parse_objects(raw: Optional[str]) -> list:
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []

def search_action_clips(video_id: str, q: str, k: int, filter_objects: str | None):
    """Legacy-named action search; now backed by the SigLIP action-clip index."""
    index, rows = load_siglip_action_clips_index(video_id)
    qv = _get_siglip().encode_text([q])
    D, I = index.search(qv, k)
    hits = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        _, start, end, objects_json, caption = rows[idx]
        objs = _parse_objects(objects_json)
        if filter_objects and objs and (filter_objects not in objs):
            continue
        hits.append({
            "video_id": video_id,
            "start": float(start), "end": float(end),
            "score": float(score), "objects": objs,
            "caption": caption or "",
        })
    hits.sort(key=lambda h: (h["start"], -h["score"]))
    return hits

def chain_actions(video_id: str, steps: list[str], k_per_step=40, max_gap=8.0, filter_objects: str | None = None):
    cand = [search_action_clips(video_id, q, k_per_step, filter_objects) for q in steps]
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

    def total_score(p): return sum(x["score"] for x in p)
    paths = sorted(paths, key=total_score, reverse=True)
    full = [p for p in paths if len(p) == len(steps)]
    chosen = full[0] if full else (paths[0] if paths else [])
    return chosen, cand

def representative_frame_for_segment(video_id: str, seg_start: float, seg_end: float) -> Optional[str]:
    mid = 0.5 * (float(seg_start) + float(seg_end))
    with db() as conn:
        # Exact containment of midpoint
        row = conn.execute("""
            SELECT frame, start, end
            FROM visual_chunks
            WHERE video_id=? AND start <= ? AND end >= ?
            ORDER BY ABS((start+end)/2.0 - ?) ASC
            LIMIT 1
        """, (video_id, mid, mid, mid)).fetchone()

        if not row:
            # Any overlap, closest to midpoint
            row = conn.execute("""
                SELECT frame, start, end
                FROM visual_chunks
                WHERE video_id=? AND end >= ? AND start <= ?
                ORDER BY
                  CASE
                    WHEN ? BETWEEN start AND end THEN 0
                    ELSE MIN(ABS(start-?), ABS(end-?))
                  END ASC
                LIMIT 1
            """, (video_id, seg_start, seg_end, mid, mid, mid)).fetchone()

    return row[0] if row else None
    
def action_frame_resolver(hit: Dict[str, Any]) -> Optional[str]:
    if hit.get("frame"):
        return hit["frame"]
    return representative_frame_for_segment(hit["video_id"], hit["start"], hit["end"])
# --------------------
# per-video + global helpers
# --------------------
def list_video_ids_with(ext: str) -> list[str]:
    if not os.path.isdir(_indexes_dir):
        return []
    vids = []
    for fn in os.listdir(_indexes_dir):
        if fn.endswith(ext):
            vids.append(fn[:-len(ext)])
    return vids

def search_text_single(video_id: str, q: str, k: int):
    index, rows = load_index(video_id)
    qv = EMB.encode([q], normalize_embeddings=True).astype('float32')
    D, I = index.search(qv, k)
    out = []
    for s, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1: continue
        _, start, end, text = rows[idx]
        out.append({
            "video_id": video_id,
            "start": float(start), "end": float(end),
            "score": float(s), "text": text
        })
    return out

def _apply_filter_objects(objs: List[str], filter_objects: Optional[str]) -> bool:
    """Word-list filter used inside SigLIP search (before captioning).
    No-op when filter_objects is empty or objs is empty; otherwise requires a
    case-insensitive exact word match. Kept for legacy /vsearch /asearch paths."""
    if not filter_objects:
        return True
    if not objs:
        return True
    needle = filter_objects.strip().lower()
    if not needle:
        return True
    return any(needle == str(o).strip().lower() for o in objs)

def _hit_matches_filter(hit: dict, filter_objects: Optional[str]) -> bool:
    """Post-caption filter for /query: case-insensitive substring match against
    the caption *and* the object keyword list. Graceful no-op if the hit has
    neither caption nor objects populated yet. This is what actually runs after
    lazy Florence-2 captioning, so it catches 'knives' in captions that say
    'using a knife to cut', 'Knife' typed with a capital, etc."""
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
    if not caption and not objs:
        return True  # nothing to filter against yet
    return False

def search_visual_single(video_id: str, q: str, k: int, filter_objects: Optional[str]):
    """Visual frame search backed by SigLIP vision-text embeddings."""
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
        out.append({
            "video_id": video_id,
            "start": float(start), "end": float(end),
            "score": float(s), "frame": frame, "objects": objs,
            "caption": caption or "",
        })
    return out

def search_action_single(video_id: str, q: str, k: int, filter_objects: Optional[str]):
    """Action-clip search backed by SigLIP vision-text embeddings."""
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
        out.append({
            "video_id": video_id,
            "start": float(start), "end": float(end),
            "score": float(s), "objects": objs,
            "caption": caption or "",
        })
    return out

def _globally(needle_ext: str, restrict: Optional[list[str]]) -> list[str]:
    vids = list_video_ids_with(needle_ext)
    return [v for v in vids if (not restrict or v in restrict)]

def search_text_global(q: str, k: int, restrict_videos: Optional[list[str]] = None):
    # Context pre-filter
    candidates = filter_videos_by_context(q, restrict_videos, topn=100, min_cos=0.18)
    # Fall back to everything if nothing passes context filter
    vids = candidates if candidates else _globally(".faiss", restrict_videos)
    
    all_hits = []
    for vid in vids:
        try:
            all_hits.extend(search_text_single(vid, q, k))
        except Exception:
            logger.warning("text global search skipped %s", vid, exc_info=True)
    all_hits.sort(key=lambda h: h["score"], reverse=True)
    return all_hits

def search_visual_global(q: str, k: int, filter_objects: Optional[str] = None,
                         restrict_videos: Optional[list[str]] = None):
    # Context pre-filter
    candidates = filter_videos_by_context(q, restrict_videos, topn=100, min_cos=0.18)
    # Fall back to SigLIP-indexed videos if nothing passes context filter
    vids = candidates if candidates else _globally(".svfaiss", restrict_videos)

    all_hits = []
    for vid in vids:
        try:
            all_hits.extend(search_visual_single(vid, q, k, filter_objects))
        except Exception:
            logger.warning("visual global search skipped %s", vid, exc_info=True)
    all_hits.sort(key=lambda h: h["score"], reverse=True)
    return all_hits

def search_action_global(q: str, k: int, filter_objects: Optional[str] = None,
                         restrict_videos: Optional[list[str]] = None):
    # Context pre-filter
    candidates = filter_videos_by_context(q, restrict_videos, topn=100, min_cos=0.18)
    # Fall back to SigLIP-indexed videos if nothing passes context filter
    vids = candidates if candidates else _globally(".saclip.faiss", restrict_videos)

    all_hits = []
    for vid in vids:
        try:
            all_hits.extend(search_action_single(vid, q, k, filter_objects))
        except Exception:
            logger.warning("action global search skipped %s", vid, exc_info=True)
    all_hits.sort(key=lambda h: h["score"], reverse=True)
    return all_hits

# ---- Shared helpers ----
def _postproc_hits(hits: list[dict], *, key_mode: str, k: int | None) -> list[dict]:
    hits = dedupe_hits(hits, key_mode=key_mode)
    hits = nms_time(hits, tol=0.5)
    if k:
        hits = hits[:k]
    return hits

def _maybe_caption_rerank(
    hits: list[dict],
    *,
    verify_on: bool,
    prompts: list[str],
    require_all: list[str],
    w_base: float = 0.7,
    w_caption: float = 0.3,
) -> list[dict]:
    """Rerank hits by checking if captions mention the required terms.
    No model inference — pure text matching on pre-computed captions."""
    if not (verify_on and prompts and hits):
        return hits
    terms = [p.strip().lower() for p in prompts if p.strip()]
    req = [r.strip().lower() for r in (require_all or []) if r.strip()]

    for h in hits:
        cap = (h.get("caption") or "").lower()
        if not cap:
            h["verify_score"] = 0.0
            h["score_fused"] = w_base * float(h.get("score", 0.0))
            continue
        # Score: fraction of prompt terms found in caption
        matched = sum(1 for t in terms if t in cap)
        verify = matched / len(terms) if terms else 0.0
        # Penalize if require_all terms are missing
        if req and not all(r in cap for r in req):
            verify *= 0.3
        h["verify_score"] = verify
        h["score_fused"] = w_base * float(h.get("score", 0.0)) + w_caption * verify
    return hits

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

# --------------------
# Endpoints
# --------------------
@app.get("/search", response_model=SearchResponse)
async def search(
    video_id: str = Query(...),
    q: str = Query(...),
    k: int = Query(5, ge=1, le=MAX_K),
):
    index, rows = load_index(video_id)
    qv = EMB.encode([q], normalize_embeddings=True).astype('float32')
    D, I = index.search(qv, k)
    hits = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        _, start, end, text = rows[idx]
        hits.append(SearchHit(start=start, end=end, text=text, score=score))
    return SearchResponse(video_id=video_id, hits=hits)

@app.get("/vsearch")
async def vsearch(
    video_id: str = Query(...),
    q: str = Query(...),
    k: int = Query(6, ge=1, le=MAX_K),
    filter_objects: str | None = None,
):
    """Legacy visual search; now backed by the SigLIP frame index."""
    try:
        hits = search_visual_single(video_id, q, k, filter_objects)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    hits = dedupe_hits(hits, key_mode="auto")
    hits = nms_time(hits, tol=0.5)
    if k:
        hits = hits[:k]
    return {"video_id": video_id, "hits": hits}

@app.get("/asearch")
async def asearch(
    video_id: str = Query(...),
    q: str = Query(...),
    k: int = Query(40, ge=1, le=MAX_K),
    filter_objects: str | None = None
):
    try:
        raw = search_action_clips(video_id, q, k, filter_objects)
        hits = [{
            "start": h["start"], "end": h["end"], "score": h["score"],
            "objects": h.get("objects"), "video_id": video_id
        } for h in raw]
        hits = dedupe_hits(hits, key_mode="time")
        hits = nms_time(hits, tol=0.5)
        if k:
            hits = hits[:k]
        return {"video_id": video_id, "hits": hits}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/asearch_chain")
async def asearch_chain(
    video_id: str = Query(...),
    steps: List[str] = Query(..., description="Ordered list of action prompts"),
    k_per_step: int = Query(40, ge=1, le=MAX_K),
    max_gap: float = Query(8.0, ge=0.0, le=60.0),
    filter_objects: str | None = None
):
    try:
        path, cand = chain_actions(video_id, steps, k_per_step=k_per_step,
                                   max_gap=max_gap, filter_objects=filter_objects)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Action clip index missing: {e}")

    hits = [{
        "start": h["start"], "end": h["end"], "score": h["score"],
        "objects": h.get("objects"), "video_id": video_id
    } for h in path]
    hits = dedupe_hits(hits, key_mode="time")
    hits = nms_time(hits, tol=0.5)

    return {
        "video_id": video_id,
        "steps": steps,
        "best_path": hits,
        "candidates_preview": [c[:5] for c in cand]
    }

def _thumbnail_url_for(video_id: str) -> Optional[str]:
    """Return the mounted URL of the first sampled frame, or None if missing."""
    frames_subdir = os.path.join(_frames_dir, video_id)
    if not os.path.isdir(frames_subdir):
        return None
    try:
        jpgs = sorted(
            f for f in os.listdir(frames_subdir)
            if f.startswith("frame-") and f.endswith(".jpg")
        )
    except OSError:
        return None
    if not jpgs:
        return None
    return f"/frames/{video_id}/{jpgs[0]}"

@app.get("/videos")
async def list_videos():
    try:
        if not os.path.exists(_media_dir):
            return {"videos": []}
        videos = []
        for filename in os.listdir(_media_dir):
            if filename.endswith('.mp4'):
                vid = filename[:-4]
                has_text = os.path.exists(os.path.join(_indexes_dir, f"{vid}.faiss"))
                has_visual = os.path.exists(os.path.join(_indexes_dir, f"{vid}.svfaiss"))
                has_actions = os.path.exists(os.path.join(_indexes_dir, f"{vid}.saclip.faiss"))
                videos.append({
                    "video_id": vid,
                    "has_text_search": has_text,
                    "has_visual_search": has_visual,
                    "has_action_search": has_actions,
                    "thumbnail_url": _thumbnail_url_for(vid),
                })
        return {"videos": sorted(videos, key=lambda x: x['video_id'])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing videos: {e}")

# Video IDs must be alphanumeric + [_-] to stop path traversal / injection when
# they're spliced into filenames and FAISS paths.
_VIDEO_ID_RE = r"^[A-Za-z0-9_-]{1,64}$"

@app.delete("/videos/{video_id}")
def delete_video(video_id: str = Path(..., pattern=_VIDEO_ID_RE)):
    """Remove every trace of a video: DB rows, FAISS indexes, media, frames."""
    try:
        clear_video(video_id)
    except Exception as e:
        logger.exception("delete_video failed for %s", video_id)
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")
    return {"success": True, "video_id": video_id}

@app.get("/asearch_all")
def asearch_all(
    q: str = Query(...),
    k: int = Query(50, ge=1, le=MAX_K),
    filter_objects: str | None = None,
    videos: list[str] | None = Query(None, description="Optional list of video_ids to restrict to")
):
    try:
        hits = search_action_global(q, k, filter_objects=filter_objects, restrict_videos=videos)
        return {"query": q, "hits": hits[:k]}
    except Exception as e:
        logger.exception("/asearch_all failed")
        raise HTTPException(status_code=500, detail=f"Search error: {e}")

@app.post("/ingest")
async def ingest_video(request: VideoIngestRequest):
    try:
        video_url = request.video_url

        if request.video_id:
            video_id = request.video_id
        else:
            if 'youtube.com' in video_url or 'youtu.be' in video_url:
                if 'youtu.be/' in video_url:
                    video_id = video_url.split('youtu.be/')[-1].split('?')[0]
                else:
                    parsed = urlparse(video_url)
                    video_id = parse_qs(parsed.query).get('v', [None])[0]
                if not video_id:
                    raise ValueError("Could not extract video ID from YouTube URL")
            else:
                video_id = re.sub(r'[^a-zA-Z0-9_-]', '', video_url.split('/')[-1])[:11]

        logger.info("Ingesting video %s from %s", video_id, video_url)

        media_path = os.path.join(_media_dir, f"{video_id}.mp4")
        if os.path.exists(media_path):
            return {"success": True, "message": f"Video {video_id} already exists",
                    "video_id": video_id, "status": "already_exists"}

        from ingest import ingest as do_ingest
        from visual_ingest import ingest_visual as do_visual_ingest

        do_visual_ingest(video_url) # visual first: downloads full video, extracts wav
        do_ingest(video_url)        # text: reuses wav produced by visual ingest

        # Build video context for better search filtering
        build_video_context(video_id)

        return {"success": True, "message": f"Video {video_id} ingested successfully",
                "video_id": video_id, "status": "completed"}

    except Exception as e:
        logger.exception("Ingestion error")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/build_contexts")
async def rebuild_video_contexts(video_ids: Optional[List[str]] = None):
    """Build/rebuild video contexts for existing videos."""
    try:
        if video_ids is None:
            # Build contexts for all videos that have any index
            with db() as conn:
                rows = conn.execute("""
                    SELECT DISTINCT video_id FROM chunks
                    UNION
                    SELECT DISTINCT video_id FROM visual_chunks
                    UNION
                    SELECT DISTINCT video_id FROM visual_clips
                """).fetchall()
            video_ids = [row[0] for row in rows]

        results = []
        for video_id in video_ids:
            try:
                build_video_context(video_id)
                results.append({"video_id": video_id, "status": "success"})
            except Exception as e:
                logger.exception("build_video_context failed for %s", video_id)
                results.append({"video_id": video_id, "status": "error", "error": str(e)})

        success_count = sum(1 for r in results if r["status"] == "success")
        return {
            "success": True,
            "message": f"Built contexts for {success_count}/{len(results)} videos",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ov_verify")
def ov_verify(req: OVVerifyRequest):
    out = {}
    for f in req.frames:
        try:
            res = detect_on_image(
                image_path=os.path.abspath(f),
                prompts=req.prompts,
                box_threshold=req.box_threshold,
                text_threshold=req.text_threshold
            )
            out[f] = res
        except Exception as e:
            out[f] = {"detections": [], "debug": {"error": str(e)}}
    return {"results": out}

# --------------------
# Unified /query with scope switch
# --------------------
@app.post("/query", response_model=UnifiedSearchResponse)
def unified_query(body: UnifiedSearchRequest):
    scope = (body.scope or "video").lower()
    restrict = body.videos

    # ==== scope = video ====
    if scope == "video":
        vid = body.video_id or extract_video_id(body.video_url or "")
        if not vid:
            raise HTTPException(400, "video_id or video_url is required for scope='video'")

        need_text   = body.mode == "text"
        need_visual = body.mode == "visual"
        need_action = body.mode in ("action", "action_chain")

        if body.ingest_if_needed and not have_indexes(vid, need_text, need_visual, need_action):
            ensure_ingested(body.video_url or "", vid, need_text, need_visual, need_action)

        # ---- TEXT ----
        if body.mode == "text":
            raw = search_text_single(vid, body.query or "", body.k)
            hits = [UnifiedSearchHit(
                start=h["start"],
                end=h["end"],
                score=float(h["score"]),
                text=h["text"],
                video_id=h["video_id"]
            ) for h in raw]
            return UnifiedSearchResponse(video_id=vid, mode="text", hits=hits)

        # ---- VISUAL (frame-level) ----
        if body.mode == "visual":
            # Over-fetch so post-caption filter/rerank has headroom
            raw = search_visual_single(vid, body.query or "", body.k * 3, body.filter_objects)
            _caption_hits_lazy(vid, raw)
            # Post-caption filter_objects (now that object words are available)
            if body.filter_objects:
                raw = [h for h in raw if _hit_matches_filter(h, body.filter_objects)]
            raw = _maybe_caption_rerank(
                raw,
                verify_on=body.verify_with_gdino,
                prompts=body.verify_prompts or [],
                require_all=body.verify_require_all or [],
            )
            raw.sort(key=lambda h: float(h.get("score_fused", h.get("score", 0.0))), reverse=True)
            raw = _postproc_hits(raw, key_mode="auto", k=body.k)
            hits = [_as_unified(h) for h in raw]
            return UnifiedSearchResponse(video_id=vid, mode="visual", hits=hits)

        # ---- ACTION (segment-level) ----
        if body.mode == "action":
            raw = search_action_single(vid, body.query or "", body.k * 3, body.filter_objects)
            # Resolve a representative frame per segment, then lazy-caption those frames
            for h in raw:
                if not h.get("frame"):
                    h["frame"] = representative_frame_for_segment(vid, h["start"], h["end"])
            _caption_hits_lazy(vid, raw)
            if body.filter_objects:
                raw = [h for h in raw if _hit_matches_filter(h, body.filter_objects)]
            raw = _maybe_caption_rerank(
                raw,
                verify_on=body.verify_with_gdino,
                prompts=body.verify_prompts or [],
                require_all=body.verify_require_all or [],
            )
            raw.sort(key=lambda h: float(h.get("score_fused", h.get("score", 0.0))), reverse=True)
            raw = _postproc_hits(raw, key_mode="time", k=body.k)
            hits = [_as_unified(h) for h in raw]
            return UnifiedSearchResponse(video_id=vid, mode="action", hits=hits)

        # ---- ACTION CHAIN (within a single video) ----
        if body.mode == "action_chain":
            if not body.steps:
                raise HTTPException(400, "steps is required for mode=action_chain")

            path_hits, cand_per_step = chain_actions(
                vid, body.steps, k_per_step=body.k,
                max_gap=body.max_gap, filter_objects=body.filter_objects
            )
            path_hits = _maybe_caption_rerank(
                path_hits,
                verify_on=body.verify_with_gdino,
                prompts=body.verify_prompts or [],
                require_all=body.verify_require_all or [],
            )
            path_hits.sort(key=lambda h: float(h.get("score_fused", h.get("score", 0.0))), reverse=True)
            path_hits = _postproc_hits(path_hits, key_mode="time", k=body.k)
            hits = [_as_unified(h) for h in path_hits]
            return UnifiedSearchResponse(
                video_id=vid, mode="action_chain", hits=hits,
                info={"steps": body.steps, "preview_per_step": [c[:5] for c in cand_per_step]}
            )
        raise HTTPException(400, f"Unknown mode {body.mode}")

    # ==== scope = global ====
    if scope == "global":
        # ---- TEXT ----
        if body.mode == "text":
            raw = search_text_global(body.query or "", body.k, restrict_videos=restrict)
            hits = [UnifiedSearchHit(
                start=h["start"],
                end=h["end"],
                score=float(h["score"]),
                text=h["text"],
                video_id=h["video_id"]
            ) for h in raw]
            return UnifiedSearchResponse(video_id=None, mode="text", hits=hits)

        # ---- VISUAL (frame-level) ----
        if body.mode == "visual":
            raw = search_visual_global(
                body.query or "", body.k * 3,
                filter_objects=body.filter_objects,
                restrict_videos=restrict
            )
            _caption_hits_lazy(None, raw)
            if body.filter_objects:
                raw = [h for h in raw if _hit_matches_filter(h, body.filter_objects)]
            raw = _maybe_caption_rerank(
                raw,
                verify_on=body.verify_with_gdino,
                prompts=body.verify_prompts or [],
                require_all=body.verify_require_all or [],
            )
            raw.sort(key=lambda h: float(h.get("score_fused", h.get("score", 0.0))), reverse=True)
            raw = _postproc_hits(raw, key_mode="auto", k=body.k)
            hits = [_as_unified(h) for h in raw]
            return UnifiedSearchResponse(video_id=None, mode="visual", hits=hits)

        # ---- ACTION (segment-level) ----
        if body.mode == "action":
            raw = search_action_global(
                body.query or "", body.k * 3,
                filter_objects=body.filter_objects,
                restrict_videos=restrict
            )
            for h in raw:
                if not h.get("frame"):
                    h["frame"] = representative_frame_for_segment(h["video_id"], h["start"], h["end"])
            _caption_hits_lazy(None, raw)
            if body.filter_objects:
                raw = [h for h in raw if _hit_matches_filter(h, body.filter_objects)]
            raw = _maybe_caption_rerank(
                raw,
                verify_on=body.verify_with_gdino,
                prompts=body.verify_prompts or [],
                require_all=body.verify_require_all or [],
            )
            raw.sort(key=lambda h: float(h.get("score_fused", h.get("score", 0.0))), reverse=True)
            raw = _postproc_hits(raw, key_mode="time", k=body.k)
            hits = [_as_unified(h) for h in raw]
            return UnifiedSearchResponse(video_id=None, mode="action", hits=hits)

        # ---- ACTION CHAIN (global: per-video chain, pick best) ----
        if body.mode == "action_chain":
            if not body.steps:
                raise HTTPException(400, "steps is required for mode=action_chain")

            vids = _globally(".saclip.faiss", restrict)
            per_video = []  # (video_id, path_hits, cand_per_step, total_score)

            for vid in vids:
                try:
                    path_hits, cand_per_step = chain_actions(
                        vid, body.steps, k_per_step=body.k,
                        max_gap=body.max_gap, filter_objects=body.filter_objects
                    )
                    path_hits = _maybe_caption_rerank(
                        path_hits,
                        verify_on=body.verify_with_gdino,
                        prompts=body.verify_prompts or [],
                        require_all=body.verify_require_all or [],
                    )

                    # score the path (prefer fused)
                    def seg_score(h): return float(h.get("score_fused", h.get("score", 0.0)))
                    total = sum(seg_score(h) for h in path_hits)
                    per_video.append((vid, path_hits, cand_per_step, total))
                except Exception:
                    continue  # skip videos without indexes or failing chain

            if not per_video:
                return UnifiedSearchResponse(
                    video_id=None, mode="action_chain", hits=[],
                    info={"steps": body.steps}
                )

            per_video.sort(key=lambda x: x[3], reverse=True)
            best_vid, best_path, best_cands, _ = per_video[0]
            best_path.sort(key=lambda h: float(h.get("score_fused", h.get("score", 0.0))), reverse=True)
            best_path = _postproc_hits(best_path, key_mode="time", k=body.k)
            hits = [_as_unified(h) for h in best_path]

            return UnifiedSearchResponse(
                video_id=best_vid,
                mode="action_chain",
                hits=hits,
                info={
                    "steps": body.steps,
                    "selected_video": best_vid,
                    "preview_per_step": [c[:5] for c in best_cands]
                }
            )

        raise HTTPException(400, "Unknown or unsupported mode for scope='global'")

    raise HTTPException(400, f"Unknown scope {scope}")

