import os, uuid, time, logging, json, re, shutil, queue, threading, subprocess, sys
from dataclasses import dataclass, field
from typing import Optional, List
from fastapi import APIRouter, HTTPException, UploadFile, File
from models import VideoIngestRequest
from utils import extract_video_id as _parse_video_id

router = APIRouter()
logger = logging.getLogger(__name__)

BASE = os.path.dirname(os.path.dirname(__file__))
_JOBS_DIR = os.path.join(BASE, "data", "jobs")
os.makedirs(_JOBS_DIR, exist_ok=True)

_jobs: dict[str, "_JobHandle"] = {}
_JOB_TTL = 3600

# Thread-managed queue; each job runs as an isolated subprocess so ingest
# models (Florence-2, YOLO) live in a separate process from the search server
# (X-CLIP, SentenceTransformer) — avoids shared-MPS OOM crashes on macOS.
_job_queue: queue.Queue = queue.Queue()
_worker_thread: Optional[threading.Thread] = None
_WORKER_SCRIPT = os.path.join(BASE, "worker.py")
_WAV_WAIT_TIMEOUT = 180


def _worker_loop():
    wlog = logging.getLogger("ingest-worker")
    wlog.info("Worker started, waiting for jobs…")
    while True:
        try:
            item = _job_queue.get(timeout=60)
        except queue.Empty:
            continue
        if item is None:
            break
        status_file, url, video_id = item
        wlog.info("Starting job: %s", video_id)
        env = {
            **os.environ,
            "PYTHONPATH": BASE,
            # Required on macOS — prevents numpy RecursionError on fork
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OBJC_DISABLE_INITIALIZE_FORK_SAFETY": "YES",
        }
        proc = subprocess.Popen(
            [sys.executable, _WORKER_SCRIPT, status_file, url, video_id],
            cwd=BASE, env=env,
        )
        try:
            proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            proc.kill()
            wlog.error("Job timed out, process killed: %s", video_id)
            raise
        wlog.info("Job done: %s (exit=%d)", video_id, proc.returncode)


def start_worker():
    global _worker_thread
    if _worker_thread is not None and _worker_thread.is_alive():
        return
    _worker_thread = threading.Thread(
        target=_worker_loop, daemon=True, name="ingest-worker"
    )
    _worker_thread.start()
    logger.info("Ingest worker started")


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


def _video_id_from_filename(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", stem)[:64].strip("_") or "upload"
    return sanitized


_MEDIA_DIR = os.path.join(BASE, "data", "media")
os.makedirs(_MEDIA_DIR, exist_ok=True)

_VALID_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv"}


def _run_ingest_proc(status_file: str, url: str, video_id: str):
    import sys, os as _os
    _backend = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _backend not in sys.path:
        sys.path.insert(0, _backend)

    is_local = _os.path.isfile(url)

    try:
        title, source_url = None, None
        if is_local:
            title = _os.path.splitext(_os.path.basename(url))[0]
            _write_status(status_file, "running", "Extracting frames…")
        else:
            _write_status(status_file, "running", "Fetching metadata…")
            try:
                import yt_dlp
                with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                    info = ydl.extract_info(url, download=False)
                    title = info.get("title")
                    source_url = info.get("webpage_url") or url
            except Exception:
                pass

        # Audio waits for the .wav written by the visual pipeline to avoid
        # an ffmpeg race — visual writes the wav file in its first ~20 s.
        import threading
        _media_dir = _os.path.join(_backend, "data", "media")
        wav_path = _os.path.join(_media_dir, f"{video_id}.wav")

        visual_exc: list = []
        audio_exc: list = []

        def _run_visual():
            try:
                from visual_ingest import ingest_visual
                _write_status(status_file, "running", "Downloading & extracting frames…")
                ingest_visual(url, progress_cb=lambda s: _write_status(status_file, "running", s))
            except Exception as e:
                visual_exc.append(e)

        def _run_audio():
            try:
                import time as _time
                deadline = _time.monotonic() + _WAV_WAIT_TIMEOUT
                while not _os.path.exists(wav_path) and _time.monotonic() < deadline:
                    _time.sleep(1)
                from pipeline import pipeline
                ingest(url)
            except Exception as e:
                audio_exc.append(e)

        t_vis = threading.Thread(target=_run_visual, daemon=True)
        t_aud = threading.Thread(target=_run_audio, daemon=True)
        t_vis.start()
        t_aud.start()
        t_vis.join()
        t_aud.join()

        if visual_exc:
            raise visual_exc[0]
        if audio_exc:
            raise audio_exc[0]

        _write_status(status_file, "running", "Building search index…")
        from video_context import build_video_context
        build_video_context(video_id)

        if title or source_url:
            from db import store_video_meta
            store_video_meta(video_id, title, source_url)

        _write_status(status_file, "done", "Done")
    except Exception:
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


def _active_ingest_job() -> Optional[str]:
    """Return job_id of any currently running/queued ingest, or None."""
    for jid, h in list(_jobs.items()):
        data = h.read()
        if data.get("status") in ("running", "queued"):
            return jid
    return None


def _is_fully_indexed(video_id: str) -> bool:
    """True only if all FAISS files AND DB rows exist for this video."""
    _indexes_dir = os.path.join(BASE, "data", "indexes")
    files_ok = (
        os.path.exists(os.path.join(_indexes_dir, f"{video_id}.faiss")) and
        os.path.exists(os.path.join(_indexes_dir, f"{video_id}.svfaiss")) and
        os.path.exists(os.path.join(_indexes_dir, f"{video_id}.xaclip.faiss"))
    )
    if not files_ok:
        return False
    from db import db
    with db() as conn:
        has_audio = conn.execute(
            "SELECT 1 FROM chunks WHERE video_id=? LIMIT 1", (video_id,)
        ).fetchone()
        has_visual = conn.execute(
            "SELECT 1 FROM visual_chunks WHERE video_id=? LIMIT 1", (video_id,)
        ).fetchone()
    return bool(has_audio and has_visual)


@router.post("/ingest")
async def ingest_video(request: VideoIngestRequest):
    _sweep_jobs()
    video_id = request.video_id or _parse_video_id(request.video_url)

    if _is_fully_indexed(video_id):
        return {"job_id": None, "video_id": video_id,
                "status": "already_exists", "message": f"Video {video_id} already indexed"}

    busy_job = _active_ingest_job()
    if busy_job:
        raise HTTPException(409, f"Another ingest is already running (job {busy_job}). Wait for it to finish.")

    job_id = str(uuid.uuid4())
    status_file = os.path.join(_JOBS_DIR, f"{job_id}.json")
    _write_status(status_file, "queued", "Queued…")

    _job_queue.put((status_file, request.video_url, video_id))

    _jobs[job_id] = _JobHandle(job_id=job_id, video_id=video_id, status_file=status_file)
    return {"job_id": job_id, "video_id": video_id, "status": "queued"}


@router.post("/ingest/upload")
async def upload_and_ingest(file: UploadFile = File(...)):
    _sweep_jobs()
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in _VALID_EXTS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(_VALID_EXTS))}")

    video_id = _video_id_from_filename(file.filename or "upload")

    # Make video_id unique if already taken by a different (non-indexed) upload
    if _is_fully_indexed(video_id):
        return {"job_id": None, "video_id": video_id,
                "status": "already_exists", "message": f"Video {video_id} already indexed"}

    busy_job = _active_ingest_job()
    if busy_job:
        raise HTTPException(409, f"Another ingest is already running (job {busy_job}). Wait for it to finish.")

    dest_path = os.path.join(_MEDIA_DIR, f"{video_id}{ext}")
    try:
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    job_id = str(uuid.uuid4())
    status_file = os.path.join(_JOBS_DIR, f"{job_id}.json")
    _write_status(status_file, "queued", "Queued…")

    _job_queue.put((status_file, dest_path, video_id))

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
    from video_context import build_video_context
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
