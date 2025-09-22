from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np, json, os
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from store import (
    load_index, get_conn, load_visual_index, load_action_clips_index,
)
from models import (
    SearchResponse, SearchHit, VideoIngestRequest,
    UnifiedSearchRequest, UnifiedSearchHit, UnifiedSearchResponse,
)
from utils_unified import extract_video_id

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

# --------------------
# Unified /query with scope switch
# --------------------
@app.post("/query", response_model=UnifiedSearchResponse)
def unified_query(body: UnifiedSearchRequest):
    scope = (body.scope or "video").lower()
    restrict = body.videos

    # ==== scope = video (existing behavior) ====
    if scope == "video":
        vid = body.video_id or extract_video_id(body.video_url or "")
        if not vid:
            raise HTTPException(400, "video_id or video_url is required for scope='video'")

        need_text   = body.mode == "text"
        need_visual = body.mode == "visual"
        need_action = body.mode in ("action", "action_chain")

        if body.ingest_if_needed and not have_indexes(vid, need_text, need_visual, need_action):
            ensure_ingested(body.video_url or "", vid, need_text, need_visual, need_action)

        if body.mode == "text":
            raw = search_text_single(vid, body.query or "", body.k)
            hits = [UnifiedSearchHit(start=h["start"], end=h["end"], score=h["score"],
                                     text=h["text"], video_id=h["video_id"]) for h in raw]
            return UnifiedSearchResponse(video_id=vid, mode="text", hits=hits)

        if body.mode == "visual":
            raw = search_visual_single(vid, body.query or "", body.k, body.filter_objects)
            raw = dedupe_hits(raw, key_mode="auto")
            raw = nms_time(raw, tol=0.5)
            if body.k:
                raw = raw[:body.k]
            hits = [UnifiedSearchHit(start=h["start"], end=h["end"], score=h["score"],
                                     frame=h["frame"], objects=h.get("objects"),
                                     video_id=h["video_id"]) for h in raw]
            return UnifiedSearchResponse(video_id=vid, mode="visual", hits=hits)

        if body.mode == "action":
            raw = search_action_single(vid, body.query or "", body.k, body.filter_objects)
            raw = dedupe_hits(raw, key_mode="time")
            raw = nms_time(raw, tol=0.5)
            if body.k:
                raw = raw[:body.k]
            hits = [UnifiedSearchHit(start=h["start"], end=h["end"], score=h["score"],
                                     objects=h.get("objects"), video_id=h["video_id"]) for h in raw]
            return UnifiedSearchResponse(video_id=vid, mode="action", hits=hits)

        if body.mode == "action_chain":
            if not body.steps:
                raise HTTPException(400, "steps is required for mode=action_chain")
            path, cand = chain_actions(vid, body.steps, k_per_step=body.k,
                                       max_gap=body.max_gap,
                                       filter_objects=body.filter_objects)
            path = dedupe_hits(path, key_mode="time")
            path = nms_time(path, tol=0.5)
            if body.k:
                path = path[:body.k]
            hits = [UnifiedSearchHit(start=h["start"], end=h["end"], score=h["score"],
                                     objects=h.get("objects"), video_id=vid) for h in path]
            return UnifiedSearchResponse(video_id=vid, mode="action_chain", hits=hits,
                                         info={"steps": body.steps,
                                               "preview_per_step": [c[:5] for c in cand]})
        raise HTTPException(400, f"Unknown mode {body.mode}")

    # ==== scope = global  ====
    if scope == "global":
        if body.mode == "text":
            raw = search_text_global(body.query or "", body.k, restrict_videos=restrict)
            hits = [UnifiedSearchHit(start=h["start"], end=h["end"], score=h["score"],
                                     text=h["text"], video_id=h["video_id"]) for h in raw]
            return UnifiedSearchResponse(video_id=None, mode="text", hits=hits)

        if body.mode == "visual":
            raw = search_visual_global(body.query or "", body.k,
                                       filter_objects=body.filter_objects,
                                       restrict_videos=restrict)
            hits = [UnifiedSearchHit(start=h["start"], end=h["end"], score=h["score"],
                                     frame=h.get("frame"), objects=h.get("objects"),
                                     video_id=h["video_id"]) for h in raw]
            return UnifiedSearchResponse(video_id=None, mode="visual", hits=hits)

        if body.mode == "action":
            raw = search_action_global(body.query or "", body.k,
                                       filter_objects=body.filter_objects,
                                       restrict_videos=restrict)
            hits = [UnifiedSearchHit(start=h["start"], end=h["end"], score=h["score"],
                                     objects=h.get("objects"), video_id=h["video_id"]) for h in raw]
            return UnifiedSearchResponse(video_id=None, mode="action", hits=hits)

        # (Optional) global action_chain could run per-video and pick best; not supported here.
        raise HTTPException(400, "action_chain not supported with scope='global'")

    raise HTTPException(400, f"Unknown scope {scope}")
