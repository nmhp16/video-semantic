# backend/routers/ingest.py
import os, uuid, time, logging, json, multiprocessing
from dataclasses import dataclass, field
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from models import VideoIngestRequest
from utils_unified import extract_video_id as _parse_video_id

router = APIRouter()
logger = logging.getLogger(__name__)

BASE = os.path.dirname(os.path.dirname(__file__))
_JOBS_DIR = os.path.join(BASE, "data", "jobs")
os.makedirs(_JOBS_DIR, exist_ok=True)

_jobs: dict[str, "_JobHandle"] = {}
_JOB_TTL = 3600  # 1 hour — ingest can be slow


@dataclass
class _JobHandle:
    job_id: str
    video_id: str
    status_file: str
    created_at: float = field(default_factory=time.monotonic)

    def read(self) -> dict:
        try:
            with open(self.status_file) as f:
                return json.load(f)
        except Exception:
            return {"status": "running", "stage": "Starting…", "error": None}


def _write_status(path: str, status: str, stage: str, error: Optional[str] = None):
    with open(path, "w") as f:
        json.dump({"status": status, "stage": stage, "error": error}, f)


def _run_ingest_proc(status_file: str, url: str, video_id: str):
    import sys, os as _os
    _backend = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _backend not in sys.path:
        sys.path.insert(0, _backend)

    try:
        _write_status(status_file, "running", "Fetching metadata…")
        title, source_url = None, None
        try:
            import yt_dlp
            with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get("title")
                source_url = info.get("webpage_url") or url
        except Exception:
            pass

        _write_status(status_file, "running", "Downloading & extracting frames…")
        from visual_ingest import ingest_visual
        ingest_visual(url)

        _write_status(status_file, "running", "Transcribing audio…")
        from ingest import ingest
        ingest(url)

        _write_status(status_file, "running", "Building search index…")
        from context import build_video_context
        build_video_context(video_id)

        if title or source_url:
            from db import store_video_meta
            store_video_meta(video_id, title, source_url)

        _write_status(status_file, "done", "Done")
    except Exception as e:
        import traceback
        _write_status(status_file, "error", "Failed", error=traceback.format_exc()[-2000:])


def _sweep_jobs():
    now = time.monotonic()
    stale = [jid for jid, h in list(_jobs.items())
             if (now - h.created_at) > _JOB_TTL]
    for jid in stale:
        h = _jobs.pop(jid)
        try:
            os.remove(h.status_file)
        except Exception:
            pass


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
    status_file = os.path.join(_JOBS_DIR, f"{job_id}.json")
    _write_status(status_file, "queued", "Queued…")

    p = multiprocessing.Process(
        target=_run_ingest_proc,
        args=(status_file, request.video_url, video_id),
        daemon=True,
    )
    p.start()

    _jobs[job_id] = _JobHandle(job_id=job_id, video_id=video_id, status_file=status_file)
    return {"job_id": job_id, "video_id": video_id, "status": "queued"}


@router.get("/ingest/status/{job_id}")
async def ingest_status(job_id: str):
    handle = _jobs.get(job_id)
    if handle is None:
        status_file = os.path.join(_JOBS_DIR, f"{job_id}.json")
        if not os.path.exists(status_file):
            raise HTTPException(404, "Job not found or expired")
        handle = _JobHandle(job_id=job_id, video_id="unknown", status_file=status_file)

    data = handle.read()
    return {
        "job_id": job_id,
        "video_id": handle.video_id,
        "status": data.get("status", "unknown"),
        "stage": data.get("stage", ""),
        "error": data.get("error"),
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
