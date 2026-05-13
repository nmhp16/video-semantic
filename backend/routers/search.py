# backend/routers/search.py
import os, json, logging
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
from models import (
    UnifiedSearchRequest,
    UnifiedSearchHit, UnifiedSearchResponse, ScoreRange, MAX_K,

)
from indexes import (
    load_index, load_siglip_visual_index,
    load_xclip_action_index, has_xclip_action_index,
)
from db import get_cached_captions
from context import filter_videos_by_context
from embeddings import get_emb

router = APIRouter()
logger = logging.getLogger(__name__)

BASE = os.path.dirname(os.path.dirname(__file__))
_indexes_dir = os.path.join(BASE, "data", "indexes")
_frames_dir  = os.path.join(BASE, "data", "frames")

# X-CLIP encoder — lazy
_XCLIP = None
def _get_xclip():
    global _XCLIP
    if _XCLIP is None:
        from visual_ingest import XCLIPEncoder
        _XCLIP = XCLIPEncoder()
    return _XCLIP



def _val(h, name, default=None):
    return h.get(name, default) if isinstance(h, dict) else getattr(h, name, default)


def dedupe_hits(hits, prefer="max", key_mode="auto"):
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
        return (float(_val(h, "start", 0.0)) + float(_val(h, "end", 0.0))) * 0.5
    for h in sorted_hits:
        m, vid = mid(h), _val(h, "video_id")
        if all(not (vid == _val(k, "video_id") and abs(m - mid(k)) <= tol) for k in kept):
            kept.append(h)
    return kept


def _caption_hits_from_cache(video_id: Optional[str], hits: list) -> None:
    if not hits:
        return
    by_vid: dict = {}
    for h in hits:
        vid = video_id or h.get("video_id")
        if not vid or not h.get("frame"):
            continue
        by_vid.setdefault(vid, []).append(h)
    for vid, vhits in by_vid.items():
        frames = [h["frame"] for h in vhits]
        cached = get_cached_captions(vid, frames)
        for h in vhits:
            entry = cached.get(h["frame"]) or {"caption": "", "objects": []}
            h["caption"] = entry["caption"]
            h["objects"] = entry.get("objects") or h.get("objects") or []


def _parse_objects(raw: Optional[str]) -> list:
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


def _apply_filter_objects(objs: list, filter_objects: Optional[str]) -> bool:
    if not filter_objects or not objs:
        return True
    needle = filter_objects.strip().lower()
    return not needle or any(needle == str(o).strip().lower() for o in objs)


def _hit_matches_filter(hit: dict, filter_objects: Optional[str]) -> bool:
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
    return not caption and not objs



def _postproc_hits(hits, *, key_mode, k):
    hits = dedupe_hits(hits, key_mode=key_mode)
    hits = nms_time(hits, tol=0.5)
    return hits[:k] if k else hits


def _as_unified(h: dict) -> UnifiedSearchHit:
    return UnifiedSearchHit(
        start=float(h.get("start", 0.0)),
        end=float(h.get("end", h.get("start", 0.0))),
        score=float(h.get("score", 0.0)),
        frame=h.get("frame"),
        objects=h.get("objects"),
        caption=h.get("caption"),
        text=h.get("text"),
        video_id=h.get("video_id"),
    )


def _score_range(hits: list) -> ScoreRange:
    scores = [float(h.get("score", 0.0)) for h in hits]
    if not scores:
        return ScoreRange(min=0.0, max=0.0)
    return ScoreRange(min=min(scores), max=max(scores))


def have_indexes(video_id: str, need_text=False, need_visual=False, need_action=False) -> bool:
    ok = True
    if need_text:
        ok &= os.path.exists(os.path.join(_indexes_dir, f"{video_id}.faiss"))
    if need_visual:
        ok &= os.path.exists(os.path.join(_indexes_dir, f"{video_id}.svfaiss"))
    if need_action:
        ok &= (os.path.exists(os.path.join(_indexes_dir, f"{video_id}.xaclip.faiss")) or
               os.path.exists(os.path.join(_indexes_dir, f"{video_id}.saclip.faiss")))
    return ok


def list_video_ids_with(ext: str) -> list:
    if not os.path.isdir(_indexes_dir):
        return []
    return [fn[:-len(ext)] for fn in os.listdir(_indexes_dir) if fn.endswith(ext)]


def _globally(needle_ext: str, restrict: Optional[list]) -> list:
    vids = list_video_ids_with(needle_ext)
    return [v for v in vids if (not restrict or v in restrict)]


def representative_frame_for_segment(video_id: str, seg_start: float, seg_end: float) -> Optional[str]:
    from db import db
    mid = 0.5 * (float(seg_start) + float(seg_end))
    with db() as conn:
        row = conn.execute("""
            SELECT frame FROM visual_chunks
            WHERE video_id=? AND start <= ? AND end >= ?
            ORDER BY ABS((start+end)/2.0 - ?) ASC LIMIT 1
        """, (video_id, mid, mid, mid)).fetchone()
        if not row:
            row = conn.execute("""
                SELECT frame FROM visual_chunks
                WHERE video_id=? AND end >= ? AND start <= ?
                ORDER BY CASE WHEN ? BETWEEN start AND end THEN 0
                              ELSE MIN(ABS(start-?), ABS(end-?)) END ASC LIMIT 1
            """, (video_id, seg_start, seg_end, mid, mid, mid)).fetchone()
    return row[0] if row else None


def search_text_single(video_id: str, q: str, k: int) -> list:
    index, rows = load_index(video_id)
    qv = get_emb().encode([q], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, k)
    return [{"video_id": video_id, "start": float(rows[idx][1]), "end": float(rows[idx][2]),
             "score": float(s), "text": rows[idx][3]}
            for s, idx in zip(D[0].tolist(), I[0].tolist()) if idx != -1]


def search_visual_single(video_id: str, q: str, k: int, filter_objects: Optional[str]) -> list:
    index, rows = load_siglip_visual_index(video_id)
    qv = _get_xclip().encode_text([q])
    D, I = index.search(qv, k)
    out = []
    for s, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        _, start, end, frame, objects, caption = rows[idx]
        objs = _parse_objects(objects)
        if not _apply_filter_objects(objs, filter_objects):
            continue
        out.append({"video_id": video_id, "start": float(start), "end": float(end),
                    "score": float(s), "frame": frame, "objects": objs, "caption": caption or ""})
    return out


def search_action_single(video_id: str, q: str, k: int, filter_objects: Optional[str]) -> list:
    index, rows = load_xclip_action_index(video_id)
    qv = _get_xclip().encode_text([q])
    D, I = index.search(qv, k)
    out = []
    for s, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        _, start, end, objects_json, caption = rows[idx]
        objs = _parse_objects(objects_json)
        if not _apply_filter_objects(objs, filter_objects):
            continue
        out.append({"video_id": video_id, "start": float(start), "end": float(end),
                    "score": float(s), "objects": objs, "caption": caption or ""})
    return out


def search_text_global(q: str, k: int, restrict_videos=None) -> list:
    candidates = filter_videos_by_context(q, restrict_videos, topn=100, min_cos=0.18)
    vids = candidates if candidates else _globally(".faiss", restrict_videos)
    all_hits = []
    for vid in vids:
        try:
            all_hits.extend(search_text_single(vid, q, k))
        except Exception:
            logger.warning("text global skipped %s", vid, exc_info=True)
    all_hits.sort(key=lambda h: h["score"], reverse=True)
    return all_hits


def search_visual_global(q: str, k: int, filter_objects=None, restrict_videos=None) -> list:
    candidates = filter_videos_by_context(q, restrict_videos, topn=100, min_cos=0.18)
    vids = candidates if candidates else _globally(".svfaiss", restrict_videos)
    all_hits = []
    for vid in vids:
        try:
            all_hits.extend(search_visual_single(vid, q, k, filter_objects))
        except Exception:
            logger.warning("visual global skipped %s", vid, exc_info=True)
    all_hits.sort(key=lambda h: h["score"], reverse=True)
    return all_hits


def search_action_global(q: str, k: int, filter_objects=None, restrict_videos=None) -> list:
    xclip_vids = set(_globally(".xaclip.faiss", restrict_videos))
    candidates = filter_videos_by_context(q, restrict_videos, topn=100, min_cos=0.18)
    vids = [v for v in candidates if v in xclip_vids] if candidates else list(xclip_vids)
    all_hits = []
    for vid in vids:
        try:
            all_hits.extend(search_action_single(vid, q, k, filter_objects))
        except Exception:
            logger.warning("action global skipped %s", vid, exc_info=True)
    all_hits.sort(key=lambda h: h["score"], reverse=True)
    return all_hits




# ── Caption enrichment (called by frontend after results render) ──




@router.get("/asearch_all")
def asearch_all(q: str = Query(...), k: int = Query(50, ge=1, le=MAX_K),
                filter_objects: str = None,
                videos: list = Query(None)):
    try:
        hits = search_action_global(q, k, filter_objects=filter_objects, restrict_videos=videos)
        return {"query": q, "hits": hits[:k]}
    except Exception as e:
        logger.exception("/asearch_all failed")
        raise HTTPException(status_code=500, detail=f"Search error: {e}")


# ── Unified /query ──

@router.post("/query", response_model=UnifiedSearchResponse)
def unified_query(body: UnifiedSearchRequest):
    from utils_unified import extract_video_id
    scope = (body.scope or "video").lower()
    restrict = body.videos

    if scope == "video":
        vid = body.video_id or extract_video_id(body.video_url or "")
        if not vid:
            raise HTTPException(400, "video_id or video_url required for scope='video'")
        if body.ingest_if_needed and not have_indexes(
            vid,
            need_text=body.mode == "text",
            need_visual=body.mode == "visual",
            need_action=body.mode == "action",
        ):
            from ingest import ingest as do_ingest
            from visual_ingest import ingest_visual as do_visual
            if body.mode == "text":
                do_ingest(body.video_url or "")
            else:
                do_visual(body.video_url or "")

        if body.mode == "text":
            try:
                raw = search_text_single(vid, body.query or "", body.k)
            except FileNotFoundError:
                raise HTTPException(404, f"No text index for video {vid} — ingest first")
            hits = [UnifiedSearchHit(start=h["start"], end=h["end"], score=h["score"],
                                     text=h["text"], video_id=h["video_id"]) for h in raw]
            return UnifiedSearchResponse(video_id=vid, mode="text", hits=hits,
                                         score_range=_score_range(raw))

        if body.mode == "visual":
            try:
                raw = search_visual_single(vid, body.query or "", body.k * 3, body.filter_objects)
            except FileNotFoundError:
                raise HTTPException(404, f"No visual index for video {vid} — ingest first")
            _caption_hits_from_cache(vid, raw)
            if body.filter_objects:
                raw = [h for h in raw if _hit_matches_filter(h, body.filter_objects)]
            raw.sort(key=lambda h: float(h.get("score", 0.0)), reverse=True)
            sr = _score_range(raw)
            raw = _postproc_hits(raw, key_mode="auto", k=body.k)
            return UnifiedSearchResponse(video_id=vid, mode="visual",
                                         hits=[_as_unified(h) for h in raw], score_range=sr)

        if body.mode == "action":
            try:
                raw = search_action_single(vid, body.query or "", body.k * 3, body.filter_objects)
            except FileNotFoundError:
                raise HTTPException(404, f"No action index for video {vid} — ingest first")
            for h in raw:
                if not h.get("frame"):
                    h["frame"] = representative_frame_for_segment(vid, h["start"], h["end"])
            _caption_hits_from_cache(vid, raw)
            if body.filter_objects:
                raw = [h for h in raw if _hit_matches_filter(h, body.filter_objects)]
            raw.sort(key=lambda h: float(h.get("score", 0.0)), reverse=True)
            sr = _score_range(raw)
            raw = _postproc_hits(raw, key_mode="time", k=body.k)
            return UnifiedSearchResponse(video_id=vid, mode="action",
                                         hits=[_as_unified(h) for h in raw], score_range=sr)

        raise HTTPException(400, f"Unknown mode {body.mode}")

    if scope == "global":
        if body.mode == "text":
            raw = search_text_global(body.query or "", body.k, restrict_videos=restrict)
            sr = _score_range(raw)
            hits = [UnifiedSearchHit(start=h["start"], end=h["end"], score=h["score"],
                                     text=h["text"], video_id=h["video_id"]) for h in raw]
            return UnifiedSearchResponse(video_id=None, mode="text", hits=hits, score_range=sr)

        if body.mode == "visual":
            raw = search_visual_global(body.query or "", body.k * 3,
                                       filter_objects=body.filter_objects,
                                       restrict_videos=restrict)
            _caption_hits_from_cache(None, raw)
            if body.filter_objects:
                raw = [h for h in raw if _hit_matches_filter(h, body.filter_objects)]
            raw.sort(key=lambda h: float(h.get("score", 0.0)), reverse=True)
            sr = _score_range(raw)
            raw = _postproc_hits(raw, key_mode="auto", k=body.k)
            return UnifiedSearchResponse(video_id=None, mode="visual",
                                         hits=[_as_unified(h) for h in raw], score_range=sr)

        if body.mode == "action":
            raw = search_action_global(body.query or "", body.k * 3,
                                       filter_objects=body.filter_objects,
                                       restrict_videos=restrict)
            for h in raw:
                if not h.get("frame"):
                    h["frame"] = representative_frame_for_segment(h["video_id"], h["start"], h["end"])
            _caption_hits_from_cache(None, raw)
            if body.filter_objects:
                raw = [h for h in raw if _hit_matches_filter(h, body.filter_objects)]
            raw.sort(key=lambda h: float(h.get("score", 0.0)), reverse=True)
            sr = _score_range(raw)
            raw = _postproc_hits(raw, key_mode="time", k=body.k)
            return UnifiedSearchResponse(video_id=None, mode="action",
                                         hits=[_as_unified(h) for h in raw], score_range=sr)

        raise HTTPException(400, "Unknown or unsupported mode for scope='global'")

    raise HTTPException(400, f"Unknown scope {scope}")
