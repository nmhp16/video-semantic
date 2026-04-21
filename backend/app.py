from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np, json, os
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict, Any
from store import (
    load_index, get_conn,
    load_siglip_visual_index, load_siglip_action_clips_index,
    get_cached_captions, put_cached_captions,
    filter_videos_by_context, passes_hard_context, build_video_context,
)
from models import (
    SearchResponse, SearchHit, VideoIngestRequest, OVVerifyRequest,
    UnifiedSearchRequest, UnifiedSearchHit, UnifiedSearchResponse,
)
from utils_unified import extract_video_id
from gdino import detect_on_image

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

_data_dir = os.path.join(os.path.dirname(__file__), "data")
_frames_dir = os.path.join(_data_dir, "frames")
_media_dir = os.path.join(_data_dir, "media")
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
            for h in missing:
                frame = h["frame"]
                abs_path = os.path.join(os.path.dirname(_data_dir), frame) if not os.path.isabs(frame) else frame
                try:
                    from PIL import Image as _PilImage
                    img = _PilImage.open(abs_path).convert("RGB")
                    result = captioner.process_image(img)
                except Exception as e:
                    print(f"[lazy caption] failed for {frame}: {e}")
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
    idx_dir = os.path.join(os.path.dirname(__file__), "data", "indexes")
    ok = True
    if need_text:
        ok &= os.path.exists(os.path.join(idx_dir, f"{video_id}.faiss"))
    if need_visual:
        ok &= os.path.exists(os.path.join(idx_dir, f"{video_id}.svfaiss"))
    if need_action:
        ok &= os.path.exists(os.path.join(idx_dir, f"{video_id}.saclip.faiss"))
    return ok

def ensure_ingested(video_url: str, video_id: str, need_text: bool, need_visual: bool, need_action: bool):
    from ingest import ingest as do_ingest
    from visual_ingest import ingest_visual as do_visual_ingest
    need_visual_or_action = need_visual or need_action
    idx_dir = os.path.join(os.path.dirname(__file__), "data", "indexes")
    if need_text and not os.path.exists(os.path.join(idx_dir, f"{video_id}.faiss")):
        do_ingest(video_url)
    if need_visual_or_action and not os.path.exists(os.path.join(idx_dir, f"{video_id}.svfaiss")):
        do_visual_ingest(video_url)

def expand_prompt(prompt: str) -> list[str]:
    return [prompt]

def encode_prompt_set(model, prompts: list[str]) -> np.ndarray:
    X = model.encode(prompts, normalize_embeddings=True).astype('float32')
    v = X.mean(axis=0)
    v /= (np.linalg.norm(v) + 1e-12)
    return v.reshape(1, -1)

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
        objs = json.loads(objects_json) if objects_json else []
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
    conn = get_conn()

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
    
    conn.close()
    if row:
        return row[0]
    else:
        return None
    
def action_frame_resolver(hit: Dict[str, Any]) -> Optional[str]:
    if hit.get("frame"):
        return hit["frame"]
    return representative_frame_for_segment(hit["video_id"], hit["start"], hit["end"])
# --------------------
# per-video + global helpers
# --------------------
def list_video_ids_with(ext: str) -> list[str]:
    idx_dir = os.path.join(os.path.dirname(__file__), "data", "indexes")
    if not os.path.isdir(idx_dir):
        return []
    vids = []
    for fn in os.listdir(idx_dir):
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
    """Return True if this hit should be kept. If objects list is empty (no
    captions generated yet), the filter is a no-op rather than rejecting the hit."""
    if not filter_objects:
        return True
    if not objs:
        return True
    return filter_objects in objs

def search_visual_single(video_id: str, q: str, k: int, filter_objects: Optional[str]):
    """Visual frame search backed by SigLIP vision-text embeddings."""
    index, rows = load_siglip_visual_index(video_id)
    qv = _get_siglip().encode_text([q])
    D, I = index.search(qv, k)
    out = []
    import json as _json
    for s, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        _, start, end, frame, objects, caption = rows[idx]
        objs = _json.loads(objects) if objects else []
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
    import json as _json
    index, rows = load_siglip_action_clips_index(video_id)
    qv = _get_siglip().encode_text([q])
    D, I = index.search(qv, k)
    out = []
    for s, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        _, start, end, objects_json, caption = rows[idx]
        objs = _json.loads(objects_json) if objects_json else []
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
        except Exception as e:
            print(f"[text global] skip {vid}: {e}")
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
        except Exception as e:
            print(f"[visual global] skip {vid}: {e}")
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
        except Exception as e:
            print(f"[action global] skip {vid}: {e}")
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
async def search(video_id: str = Query(...), q: str = Query(...), k: int = 5):
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
async def vsearch(video_id: str = Query(...), q: str = Query(...), k: int = 6, filter_objects: str | None = None):
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
    k: int = 40,
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
    k_per_step: int = 40,
    max_gap: float = 8.0,
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

def fetch_clip_row(video_id: str, clip_idx: int):
    conn = get_conn()
    row = conn.execute("""
        SELECT idx, start, end, objects
        FROM visual_clips
        WHERE video_id=? AND idx=?
    """, (video_id, clip_idx)).fetchone()
    conn.close()
    return row

@app.get("/videos")
async def list_videos():
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        media_dir = os.path.join(data_dir, "media")
        if not os.path.exists(media_dir):
            return {"videos": []}
        videos = []
        for filename in os.listdir(media_dir):
            if filename.endswith('.mp4'):
                vid = filename[:-4]
                indexes_dir = os.path.join(data_dir, "indexes")
                has_text = os.path.exists(os.path.join(indexes_dir, f"{vid}.faiss"))
                has_visual = os.path.exists(os.path.join(indexes_dir, f"{vid}.svfaiss"))
                has_actions = os.path.exists(os.path.join(indexes_dir, f"{vid}.saclip.faiss"))
                videos.append({
                    "video_id": vid,
                    "has_text_search": has_text,
                    "has_visual_search": has_visual,
                    "has_action_search": has_actions
                })
        return {"videos": sorted(videos, key=lambda x: x['video_id'])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing videos: {e}")

@app.get("/asearch_all")
def asearch_all(
    q: str = Query(...), k: int = 50,
    filter_objects: str | None = None,
    videos: list[str] | None = Query(None, description="Optional list of video_ids to restrict to")
):
    try:
        hits = search_action_global(q, k, filter_objects=filter_objects, restrict_videos=videos)
        return {"query": q, "hits": hits[:k]}
    except Exception as e:
        print(f"Error in /asearch_all: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {e}")

@app.post("/ingest")
async def ingest_video(request: VideoIngestRequest):
    try:
        import re
        from urllib.parse import urlparse, parse_qs
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

        print(f"Ingesting video: {video_id} from {video_url}")

        data_dir = os.path.join(os.path.dirname(__file__), "data")
        media_path = os.path.join(data_dir, "media", f"{video_id}.mp4")
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
        print(f"Ingestion error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/build_contexts")
async def rebuild_video_contexts(video_ids: Optional[List[str]] = None):
    """Build/rebuild video contexts for existing videos."""
    try:
        if video_ids is None:
            # Build contexts for all videos that have any index
            from store import get_conn
            conn = get_conn()
            rows = conn.execute("""
                SELECT DISTINCT video_id FROM chunks
                UNION
                SELECT DISTINCT video_id FROM visual_chunks
                UNION  
                SELECT DISTINCT video_id FROM visual_clips
            """).fetchall()
            conn.close()
            video_ids = [row[0] for row in rows]
        
        results = []
        for video_id in video_ids:
            try:
                build_video_context(video_id)
                results.append({"video_id": video_id, "status": "success"})
            except Exception as e:
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
                raw = [h for h in raw if _apply_filter_objects(h.get("objects") or [], body.filter_objects)]
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
                raw = [h for h in raw if _apply_filter_objects(h.get("objects") or [], body.filter_objects)]
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
                raw = [h for h in raw if _apply_filter_objects(h.get("objects") or [], body.filter_objects)]
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
                raw = [h for h in raw if _apply_filter_objects(h.get("objects") or [], body.filter_objects)]
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

