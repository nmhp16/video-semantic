#!/usr/bin/env python3
"""Pre-populate caption_cache for already-indexed videos."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from db import db, put_cached_captions, get_cached_captions

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "data")


def backfill(video_ids=None):
    with db() as conn:
        if video_ids:
            placeholders = ",".join(["?"] * len(video_ids))
            rows = conn.execute(
                f"SELECT video_id, frame FROM visual_chunks WHERE video_id IN ({placeholders})",
                video_ids,
            ).fetchall()
        else:
            rows = conn.execute("SELECT video_id, frame FROM visual_chunks").fetchall()

    by_video: dict = {}
    for vid, frame in rows:
        by_video.setdefault(vid, []).append(frame)

    if not by_video:
        print("No visual chunks found.")
        return

    captioner = None
    for vid, frames in by_video.items():
        cached = get_cached_captions(vid, frames)
        missing = [f for f in frames if f not in cached]
        if not missing:
            print(f"{vid}: all {len(frames)} captions already cached — skip")
            continue

        print(f"{vid}: captioning {len(missing)}/{len(frames)} frames...")
        if captioner is None:
            from visual_ingest import Florence2Captioner
            from PIL import Image
            captioner = Florence2Captioner()

        from PIL import Image
        entries: dict = {}
        for i, frame_rel in enumerate(missing):
            abs_path = os.path.join(os.path.dirname(DATA), frame_rel)
            if not os.path.exists(abs_path):
                print(f"  skip missing file: {abs_path}")
                continue
            try:
                img = Image.open(abs_path).convert("RGB")
                entries[frame_rel] = captioner.process_image(img)
            except Exception as e:
                print(f"  error on {frame_rel}: {e}")
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(missing)}")

        put_cached_captions(vid, entries)
        print(f"  done: {len(entries)} captions cached for {vid}")


if __name__ == "__main__":
    ids = sys.argv[1:] or None
    backfill(ids)
