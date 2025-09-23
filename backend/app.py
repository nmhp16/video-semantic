from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np, json, os
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict, Any
from store import (
    load_index, get_conn, load_visual_index, load_action_clips_index,
)
from models import (
    SearchResponse, SearchHit, VideoIngestRequest, OVVerifyRequest,
    UnifiedSearchRequest, UnifiedSearchHit, UnifiedSearchResponse,
)
from utils_unified import extract_video_id
from gdino import detect_on_image
from gdino_helper import rerank_with_gdino

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
CLIP_TXT = SentenceTransformer("clip-ViT-B-32")

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
    """
    Greedy time-NMS: keep top-scoring hits and drop another hit whose
    midpoint is within `tol` seconds of a kept hit.
    """
    sorted_hits = sorted(hits, key=lambda x: -float(_val(x, "score", 0.0)))
    kept = []
    def mid(h): 
        return (float(_val(h, "start", 0.0)) + float(_val(h, "end", 0.0))) * 0.5

    for h in sorted_hits:
        m = mid(h)
        if all(abs(m - mid(k)) > tol for k in kept):
            kept.append(h)
    return kept

def have_indexes(video_id: str, need_text=False, need_visual=False, need_action=False) -> bool:
    idx_dir = os.path.join(os.path.dirname(__file__), "data", "indexes")
    ok = True
    if need_text:
        ok &= os.path.exists(os.path.join(idx_dir, f"{video_id}.faiss"))
    if need_visual:
        ok &= os.path.exists(os.path.join(idx_dir, f"{video_id}.vfaiss"))
    if need_action:
        ok &= os.path.exists(os.path.join(idx_dir, f"{video_id}.aclip.faiss"))
    return ok

def ensure_ingested(video_url: str, video_id: str, need_text: bool, need_visual: bool, need_action: bool):
    from ingest import ingest as do_ingest
    from visual_ingest import ingest_visual as do_visual_ingest
    need_visual_or_action = need_visual or need_action
    idx_dir = os.path.join(os.path.dirname(__file__), "data", "indexes")
    if need_text and not os.path.exists(os.path.join(idx_dir, f"{video_id}.faiss")):
        do_ingest(video_url)
    if need_visual_or_action and not os.path.exists(os.path.join(idx_dir, f"{video_id}.vfaiss")):
        do_visual_ingest(video_url)

def expand_prompt(prompt: str) -> list[str]:
    return [prompt, f"a person {prompt}", f"someone {prompt}"]

def encode_prompt_set(clip_txt_model, prompts: list[str]) -> np.ndarray:
    X = clip_txt_model.encode(prompts, normalize_embeddings=True).astype('float32')
    v = X.mean(axis=0)
    v /= (np.linalg.norm(v) + 1e-12)
    return v.reshape(1, -1)

def search_action_clips(video_id: str, q: str, k: int, filter_objects: str | None):
    index, rows = load_action_clips_index(video_id)
    qv = encode_prompt_set(CLIP_TXT, expand_prompt(q))
    D, I = index.search(qv, k)
    hits = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1: 
            continue
        _, start, end, objects_json = rows[idx]
        objs = json.loads(objects_json) if objects_json else []
        if filter_objects and (filter_objects not in objs):
            continue
        hits.append({
            "video_id": video_id,
            "start": float(start), "end": float(end),
            "score": float(score), "objects": objs
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

def search_visual_single(video_id: str, q: str, k: int, filter_objects: Optional[str]):
    index, rows = load_visual_index(video_id)
    qv = CLIP_TXT.encode([q], normalize_embeddings=True).astype('float32')
    D, I = index.search(qv, k)
    out = []
    import json as _json
    for s, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1: continue
        _, start, end, frame, objects = rows[idx]
        objs = _json.loads(objects) if objects else []
        if filter_objects and filter_objects not in objs:
            continue
        out.append({
            "video_id": video_id,
            "start": float(start), "end": float(end),
            "score": float(s), "frame": frame, "objects": objs
        })
    return out

def search_action_single(video_id: str, q: str, k: int, filter_objects: Optional[str]):
    hits = search_action_clips(video_id, q, k, filter_objects)
    return [
        {
            "video_id": video_id,
            "start": h["start"], "end": h["end"],
            "score": h["score"], "objects": h.get("objects")
        }
        for h in hits
    ]

def _globally(needle_ext: str, restrict: Optional[list[str]]) -> list[str]:
    vids = list_video_ids_with(needle_ext)
    return [v for v in vids if (not restrict or v in restrict)]

def search_text_global(q: str, k: int, restrict_videos: Optional[list[str]] = None):
    vids = _globally(".faiss", restrict_videos)
    all_hits = []
    for vid in vids:
        try:
            all_hits.extend(search_text_single(vid, q, k))
        except Exception as e:
            print(f"[text global] skip {vid}: {e}")
    all_hits.sort(key=lambda h: h["score"], reverse=True)
    return all_hits[:k]

def search_visual_global(q: str, k: int, filter_objects: Optional[str] = None,
                         restrict_videos: Optional[list[str]] = None):
    vids = _globally(".vfaiss", restrict_videos)
    all_hits = []
    for vid in vids:
        try:
            all_hits.extend(search_visual_single(vid, q, k, filter_objects))
        except Exception as e:
            print(f"[visual global] skip {vid}: {e}")
    all_hits = dedupe_hits(all_hits, key_mode="auto")
    all_hits = nms_time(all_hits, tol=0.5)
    all_hits.sort(key=lambda h: h["score"], reverse=True)
    return all_hits[:k]

def search_action_global(q: str, k: int, filter_objects: Optional[str] = None,
                         restrict_videos: Optional[list[str]] = None):
    vids = _globally(".aclip.faiss", restrict_videos)
    all_hits = []
    for vid in vids:
        try:
            all_hits.extend(search_action_single(vid, q, k, filter_objects))
        except Exception as e:
            print(f"[action global] skip {vid}: {e}")
    all_hits = dedupe_hits(all_hits, key_mode="time")
    all_hits = nms_time(all_hits, tol=0.5)
    all_hits.sort(key=lambda h: h["score"], reverse=True)
    return all_hits[:k]

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
    try:
        index, rows = load_visual_index(video_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    qv = CLIP_TXT.encode([q], normalize_embeddings=True).astype('float32')
    D, I = index.search(qv, k)
    hits = []
    import json as _json
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        _, start, end, frame, objects = rows[idx]
        objs = _json.loads(objects) if objects else []
        if filter_objects and filter_objects not in objs:
            continue
        hits.append({
            "start": float(start), "end": float(end), "frame": frame,
            "objects": objs, "score": float(score), "video_id": video_id
        })

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
                has_visual = os.path.exists(os.path.join(indexes_dir, f"{vid}.vfaiss"))
                has_actions = os.path.exists(os.path.join(indexes_dir, f"{vid}.aclip.faiss"))
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

        do_ingest(video_url)        # text (ASR + embeddings)
        do_visual_ingest(video_url) # visual (frames + action clips)

        return {"success": True, "message": f"Video {video_id} ingested successfully",
                "video_id": video_id, "status": "completed"}

    except Exception as e:
        print(f"Ingestion error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

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
            raw = search_visual_single(vid, body.query or "", body.k, body.filter_objects)
            raw = dedupe_hits(raw, key_mode="auto")
            raw = nms_time(raw, tol=0.5)

            if getattr(body, "verify_with_gdino", False):
                topk = min(getattr(body, "verify_topk", 30) or 30, len(raw))
                cand_hits = raw[:topk]
                prompts = body.verify_prompts or []
                require = body.verify_require_all or []
                if prompts:
                    cand_hits = rerank_with_gdino(
                        cand_hits,
                        prompts=prompts,
                        require_all=require,
                        box_th=body.verify_box_threshold,
                        text_th=body.verify_text_threshold,
                        w_clip=0.6, w_gdino=0.4
                    )
                    raw = cand_hits + raw[topk:]

            if body.k:
                raw = raw[:body.k]

            hits = [UnifiedSearchHit(
                start=h["start"],
                end=h["end"],
                score=float(h.get("score_fused", h["score"])),
                frame=h["frame"],
                objects=h.get("objects"),
                video_id=h["video_id"]
            ) for h in raw]
            return UnifiedSearchResponse(video_id=vid, mode="visual", hits=hits)

        # ---- ACTION (segment-level) ----
        if body.mode == "action":
            raw = search_action_single(vid, body.query or "", body.k, body.filter_objects)
            raw = dedupe_hits(raw, key_mode="time")
            raw = nms_time(raw, tol=0.5)

            if getattr(body, "verify_with_gdino", False) and raw:
                topk = min(getattr(body, "verify_topk", 30) or 30, len(raw))
                cand_hits = raw[:topk]
                prompts = body.verify_prompts or []
                require = body.verify_require_all or []
                if prompts:
                    cand_hits = rerank_with_gdino(
                        cand_hits,
                        prompts=prompts,
                        require_all=require,
                        box_th=body.verify_box_threshold,
                        text_th=body.verify_text_threshold,
                        w_clip=0.6, w_gdino=0.4,
                        frame_resolver=action_frame_resolver
                    )
                    raw = cand_hits + raw[topk:]

            if body.k:
                raw = raw[:body.k]

            hits = [UnifiedSearchHit(
                start=h["start"],
                end=h["end"],
                score=float(h.get("score_fused", h["score"])),
                frame=h.get("frame"),          # may be None; GDINO used resolver
                objects=h.get("objects"),
                video_id=h["video_id"]
            ) for h in raw]
            return UnifiedSearchResponse(video_id=vid, mode="action", hits=hits)

        # ---- ACTION CHAIN (within a single video) ----
        if body.mode == "action_chain":
            if not body.steps:
                raise HTTPException(400, "steps is required for mode=action_chain")

            path_hits, cand_per_step = chain_actions(
                vid, body.steps, k_per_step=body.k,
                max_gap=body.max_gap, filter_objects=body.filter_objects
            )
            path_hits = dedupe_hits(path_hits, key_mode="time")
            path_hits = nms_time(path_hits, tol=0.5)

            if getattr(body, "verify_with_gdino", False) and path_hits:
                topk = min(getattr(body, "verify_topk", 30) or 30, len(path_hits))
                cand_hits = path_hits[:topk]
                prompts = body.verify_prompts or []
                require = body.verify_require_all or []
                if prompts:
                    cand_hits = rerank_with_gdino(
                        cand_hits,
                        prompts=prompts,
                        require_all=require,
                        box_th=body.verify_box_threshold,
                        text_th=body.verify_text_threshold,
                        w_clip=0.6, w_gdino=0.4,
                        frame_resolver=action_frame_resolver
                    )
                    path_hits = cand_hits + path_hits[topk:]

            if body.k:
                path_hits = path_hits[:body.k]

            hits = [UnifiedSearchHit(
                start=h["start"],
                end=h["end"],
                score=float(h.get("score_fused", h["score"])),
                frame=h.get("frame"),
                objects=h.get("objects"),
                video_id=vid
            ) for h in path_hits]

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
                body.query or "", body.k,
                filter_objects=body.filter_objects,
                restrict_videos=restrict
            )

            if getattr(body, "verify_with_gdino", False) and raw:
                topk = min(getattr(body, "verify_topk", 30) or 30, len(raw))
                cand_hits = raw[:topk]
                prompts = body.verify_prompts or []
                require = body.verify_require_all or []
                if prompts:
                    cand_hits = rerank_with_gdino(
                        cand_hits,
                        prompts=prompts,
                        require_all=require,
                        box_th=body.verify_box_threshold,
                        text_th=body.verify_text_threshold,
                        w_clip=0.6, w_gdino=0.4
                    )
                    raw = cand_hits + raw[topk:]

            hits = [UnifiedSearchHit(
                start=h["start"],
                end=h["end"],
                score=float(h.get("score_fused", h["score"])),
                frame=h.get("frame"),
                objects=h.get("objects"),
                video_id=h["video_id"]
            ) for h in raw]
            return UnifiedSearchResponse(video_id=None, mode="visual", hits=hits)

        # ---- ACTION (segment-level) ----
        if body.mode == "action":
            raw = search_action_global(
                body.query or "", body.k,
                filter_objects=body.filter_objects,
                restrict_videos=restrict
            )

            if getattr(body, "verify_with_gdino", False) and raw:
                topk = min(getattr(body, "verify_topk", 30) or 30, len(raw))
                cand_hits = raw[:topk]
                prompts = body.verify_prompts or []
                require = body.verify_require_all or []
                if prompts:
                    cand_hits = rerank_with_gdino(
                        cand_hits,
                        prompts=prompts,
                        require_all=require,
                        box_th=body.verify_box_threshold,
                        text_th=body.verify_text_threshold,
                        w_clip=0.6, w_gdino=0.4,
                        frame_resolver=action_frame_resolver
                    )
                    raw = cand_hits + raw[topk:]

            hits = [UnifiedSearchHit(
                start=h["start"],
                end=h["end"],
                score=float(h.get("score_fused", h["score"])),
                frame=h.get("frame"),
                objects=h.get("objects"),
                video_id=h["video_id"]
            ) for h in raw]
            return UnifiedSearchResponse(video_id=None, mode="action", hits=hits)

        # ---- ACTION CHAIN (global: per-video chain, pick best) ----
        if body.mode == "action_chain":
            if not body.steps:
                raise HTTPException(400, "steps is required for mode=action_chain")

            vids = _globally(".aclip.faiss", restrict)
            per_video = []  # (video_id, path_hits, cand_per_step, total_score)

            for vid in vids:
                try:
                    path_hits, cand_per_step = chain_actions(
                        vid, body.steps, k_per_step=body.k,
                        max_gap=body.max_gap, filter_objects=body.filter_objects
                    )
                    path_hits = dedupe_hits(path_hits, key_mode="time")
                    path_hits = nms_time(path_hits, tol=0.5)

                    if getattr(body, "verify_with_gdino", False) and path_hits:
                        topk = min(getattr(body, "verify_topk", 30) or 30, len(path_hits))
                        cand_hits = path_hits[:topk]
                        prompts = body.verify_prompts or []
                        require = body.verify_require_all or []
                        if prompts:
                            cand_hits = rerank_with_gdino(
                                cand_hits,
                                prompts=prompts,
                                require_all=require,
                                box_th=body.verify_box_threshold,
                                text_th=body.verify_text_threshold,
                                w_clip=0.6, w_gdino=0.4,
                                frame_resolver=action_frame_resolver
                            )
                            path_hits = cand_hits + path_hits[topk:]

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

            if body.k:
                best_path = best_path[:body.k]

            hits = [UnifiedSearchHit(
                start=h["start"],
                end=h["end"],
                score=float(h.get("score_fused", h["score"])),
                frame=h.get("frame"),
                objects=h.get("objects"),
                video_id=best_vid
            ) for h in best_path]

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

