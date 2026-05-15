import os, json, logging
from fastapi import APIRouter, HTTPException, Path
from typing import Optional, List
from db import clear_video
from index_store import evict_video

router = APIRouter()
logger = logging.getLogger(__name__)

BASE = os.path.dirname(os.path.dirname(__file__))
_data_dir    = os.path.join(BASE, "data")
_frames_dir  = os.path.join(_data_dir, "frames")
_media_dir   = os.path.join(_data_dir, "media")
_indexes_dir = os.path.join(_data_dir, "indexes")

_VIDEO_ID_RE = r"^[A-Za-z0-9_-]{1,64}$"


def _thumbnail_url_for(video_id: str) -> Optional[str]:
    frames_subdir = os.path.join(_frames_dir, video_id)
    if not os.path.isdir(frames_subdir):
        return None
    try:
        jpgs = sorted(f for f in os.listdir(frames_subdir)
                      if f.startswith("frame-") and f.endswith(".jpg"))
    except OSError:
        return None
    return f"/frames/{video_id}/{jpgs[0]}" if jpgs else None


@router.get("/videos")
async def list_videos():
    try:
        if not os.path.exists(_media_dir):
            return {"videos": []}
        from db import db
        videos = []
        for filename in os.listdir(_media_dir):
            if not filename.endswith(".mp4"):
                continue
            vid = filename[:-4]
            has_text      = os.path.exists(os.path.join(_indexes_dir, f"{vid}.faiss"))
            has_visual    = os.path.exists(os.path.join(_indexes_dir, f"{vid}.svfaiss"))
            has_xclip     = os.path.exists(os.path.join(_indexes_dir, f"{vid}.xaclip.faiss"))
            has_action    = has_xclip or os.path.exists(os.path.join(_indexes_dir, f"{vid}.saclip.faiss"))
            title, source_url, top_objects = None, None, []
            with db() as conn:
                row = conn.execute(
                    "SELECT title, source_url, objects_topk FROM video_context WHERE video_id=?",
                    (vid,),
                ).fetchone()
            if row:
                title, source_url, obj_json = row
                try:
                    obj_dict = json.loads(obj_json or "{}")
                    top_objects = list(obj_dict.keys())[:10]
                except (json.JSONDecodeError, TypeError):
                    top_objects = []
            videos.append({
                "video_id": vid,
                "title": title,
                "source_url": source_url,
                "has_text_search": has_text,
                "has_visual_search": has_visual,
                "has_action_search": has_action,
                "has_xclip_action": has_xclip,
                "thumbnail_url": _thumbnail_url_for(vid),
                "top_objects": top_objects,
            })
        return {"videos": sorted(videos, key=lambda x: x["video_id"])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing videos: {e}")


@router.delete("/videos/{video_id}")
def delete_video(video_id: str = Path(..., pattern=_VIDEO_ID_RE)):
    try:
        result = clear_video(video_id)
        evict_video(video_id)
    except Exception as e:
        logger.exception("delete_video failed for %s", video_id)
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")
    return {"success": True, "video_id": video_id, **result}


