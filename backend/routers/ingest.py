# backend/routers/ingest.py
import os, uuid, time, logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from models import VideoIngestRequest
from utils_unified import extract_video_id as _parse_video_id

router = APIRouter()
logger = logging.getLogger(__name__)

BASE = os.path.dirname(os.path.dirname(__file__))
_media_dir = os.path.join(BASE, "data", "media")

_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, "_JobState"] = {}
_JOB_TTL = 600  # seconds to keep terminal jobs


@dataclass
class _JobState:
    job_id: str
    video_id: str
    status: str        # queued | running | done | error
    stage: str
    error: Optional[str] = None
    created_at: float = field(default_factory=time.monotonic)


def _set(job_id: str, stage: str, status: str = "running"):
    if job_id in _jobs:
        _jobs[job_id].status = status
        _jobs[job_id].stage = stage


def _sweep_jobs():
    now = time.monotonic()
    stale = [jid for jid, j in _jobs.items()
             if j.status in ("done", "error") and (now - j.created_at) > _JOB_TTL]
    for jid in stale:
        del _jobs[jid]


def _run_ingest(job_id: str, url: str, video_id: str):
    try:
        _set(job_id, "Downloading & extracting frames…")
        # Fetch title before downloading (no-download metadata call)
        title, source_url = None, None
        try:
            import yt_dlp
            with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get("title")
                source_url = info.get("webpage_url") or url
        except Exception:
            logger.warning("Could not fetch yt-dlp metadata for %s", url)

        from visual_ingest import ingest_visual as do_visual
        do_visual(url)

        _set(job_id, "Transcribing audio…")
        from ingest import ingest as do_ingest
        do_ingest(url)

        _set(job_id, "Building search index…")
        from context import build_video_context
        build_video_context(video_id)

        if title or source_url:
            from db import store_video_meta
            store_video_meta(video_id, title, source_url)

        _jobs[job_id].status = "done"
        _jobs[job_id].stage = "Done"
    except Exception as e:
        logger.exception("Ingest failed for %s", video_id)
        _jobs[job_id].status = "error"
        _jobs[job_id].error = str(e)


@router.post("/ingest")
async def ingest_video(request: VideoIngestRequest):
    _sweep_jobs()
    video_id = request.video_id or _parse_video_id(request.video_url)

    _indexes_dir = os.path.join(BASE, "data", "indexes")
    fully_indexed = (
        os.path.exists(os.path.join(_indexes_dir, f"{video_id}.faiss")) and
        os.path.exists(os.path.join(_indexes_dir, f"{video_id}.svfaiss")) and
        os.path.exists(os.path.join(_indexes_dir, f"{video_id}.xaclip.faiss"))
    )
    if fully_indexed:
        return {"job_id": None, "video_id": video_id,
                "status": "already_exists", "message": f"Video {video_id} already indexed"}

    job_id = str(uuid.uuid4())
    _jobs[job_id] = _JobState(job_id=job_id, video_id=video_id,
                               status="queued", stage="Queued…")
    _executor.submit(_run_ingest, job_id, request.video_url, video_id)
    return {"job_id": job_id, "video_id": video_id, "status": "queued"}


@router.get("/ingest/status/{job_id}")
async def ingest_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found or expired")
    if job.status in ("done", "error") and (time.monotonic() - job.created_at) > _JOB_TTL:
        del _jobs[job_id]
        raise HTTPException(404, "Job expired")
    return {
        "job_id": job.job_id,
        "video_id": job.video_id,
        "status": job.status,
        "stage": job.stage,
        "error": job.error,
    }


@router.post("/build_contexts")
async def rebuild_video_contexts(video_ids: Optional[List[str]] = None):
    from db import db
    from context import build_video_context
    if video_ids is None:
        with db() as conn:
            rows = conn.execute("""
                SELECT DISTINCT video_id FROM chunks
                UNION SELECT DISTINCT video_id FROM visual_chunks
                UNION SELECT DISTINCT video_id FROM visual_clips
            """).fetchall()
        video_ids = [r[0] for r in rows]
    results = []
    for vid in video_ids:
        try:
            build_video_context(vid)
            results.append({"video_id": vid, "status": "success"})
        except Exception as e:
            logger.exception("build_video_context failed for %s", vid)
            results.append({"video_id": vid, "status": "error", "error": str(e)})
    ok = sum(1 for r in results if r["status"] == "success")
    return {"success": True, "message": f"Built contexts for {ok}/{len(results)} videos",
            "results": results}
