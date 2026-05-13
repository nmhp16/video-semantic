# backend/store.py — re-exports for backward compatibility
# ingest.py and visual_ingest.py import from here; do not remove.
from indexes import (
    DATA,
    save_index, load_index,
    save_visual_index, load_visual_index,
    save_action_clips_index, load_action_clips_index,
    save_visual_metadata, save_action_clips_metadata,
    save_siglip_visual_index, load_siglip_visual_index,
    save_siglip_action_clips_index, load_siglip_action_clips_index,
    evict_video,
)
from db import (
    db,
    clear_video,
    get_cached_captions,
    put_cached_captions,
    store_video_meta,
)
from context import (
    build_video_context,
    filter_videos_by_context,
    passes_hard_context,
)
