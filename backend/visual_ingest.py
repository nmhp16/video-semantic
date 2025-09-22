import os, json, math, subprocess
from typing import List, Dict
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO
from store import DATA, save_visual_index, save_action_clips_index
import re
from pathlib import Path
from ingest import extract_audio

MEDIA = os.path.join(DATA, "media")
FRAMES = os.path.join(DATA, "frames")
os.makedirs(FRAMES, exist_ok=True)

YT_ID_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")

def ytdlp_video(url: str, out_mp4: str):
        subprocess.check_call([
            "yt-dlp", "-f", "bv*+ba/b", "--merge-output-format", "mp4",
            "-o", out_mp4, url
        ])

# --- Frame sampling ---
def sample_frames(video_path: str, out_dir: str, every_sec: float = 1.0) -> List[Dict]:
    os.makedirs(out_dir, exist_ok=True)

    # Infer duration using ffprobe
    probe = subprocess.check_output([
        "ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1", video_path
    ]).decode().strip()
    duration = float(probe) if probe else 0.0

    # Use fps = 1 / every_sec
    fps = max(0.0001, 1.0 / every_sec)

    # FFMPEG write frame-%06d.jpg starting at index 0
    subprocess.check_call([
        "ffmpeg","-y","-i", video_path, "-vf", f"fps={fps}", os.path.join(out_dir, "frame-%06d.jpg")
    ])

    # Collect files and compute timestamps by index * every_sec
    frames = sorted([f for f in os.listdir(out_dir) if f.startswith("frame-") and f.endswith(".jpg")])
    out = []
    for i, fn in enumerate(frames):
        t = i * every_sec
        out.append({
        "path": os.path.join(out_dir, fn), 
        "t": t, 
        "t_end": min(duration, (i+1)*every_sec)
        })
    
    return out

# --- CLIP embeddings with SentenceTransformer ---
class ClipEncoder:
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def encode_images(self, pil_images: List[Image.Image]) -> np.ndarray:
        X = self.model.encode(pil_images, normalize_embeddings=True)
        return np.array(X, dtype="float32")

    def encode_text(self, queries: List[str]) -> np.ndarray:
        X = self.model.encode(queries, normalize_embeddings=True)
        return np.array(X, dtype="float32")
    
# --- YOLO object detection ---
class YoloDetector:
    def __init__(self, mode_name: str = "yolov8n.pt"):
        self.model = YOLO(mode_name)

    def detect_labels(self, image_paths: List[str], conf: float = 0.25) -> List[List[str]]:
        labels_per_image = []
        for p in image_paths:
            res = self.model.predict(p, conf=conf, verbose=False) [0]
            labels = [res.names[int(c)] for c in (res.boxes.cls.cpu().numpy().tolist() if res.boxes is not None else [])]
            labels_per_image.append(labels)

        return labels_per_image

def build_clip_windows(frames, embs, labels, clip_len=2.0, stride=0.5):
    i, N = 0, len(frames)
    clip_vecs, rows = [], []
    while i < N:
        t0 = frames[i]["t"]
        # Collect [t0, t0 + clip_len]
        idxs = []
        j = i
        while j < N and frames[j]["t"] <= t0 + clip_len:
            idxs.append(j) 
            j += 1
        if idxs:
            V = embs[idxs]
            v = V.mean(axis=0)
            v /= (np.linalg.norm(v) + 1e-12)
            objs = set()
            for k in idxs:
                for o in labels[k]:
                    objs.add(o)
            clip_vecs.append(v)
            rows.append({
                "start": float(frames[idxs[0]]["t"]),
                "end":   float(frames[idxs[-1]]["t_end"]),
                "objects": sorted(list(objs))
            })
        # Stride
        t_next = t0 + stride
        while i < N and frames[i]["t"] < t_next:
            i += 1
    return np.stack(clip_vecs, axis=0).astype("float32"), rows

def ingest_visual(url_or_path: str, every_sec: float = 1.0):
    # Resolve video_id
    m = YT_ID_RE.search(url_or_path)
    if m:
        video_id = m.group(1)
    else:
        video_id = Path(url_or_path).stem

    # Download or copy
    if url_or_path.startswith("http"):
        tmp_media = os.path.join(MEDIA, f"{video_id}.mp4")
        ytdlp_video(url_or_path, tmp_media)
        audio_wav_path = os.path.join(MEDIA, f"{video_id}.wav")
        extract_audio(tmp_media, audio_wav_path)
    else:
        audio_wav_path = os.path.join(MEDIA, f"{video_id}.wav")
        extract_audio(url_or_path, audio_wav_path)

    # Find visual container
    base = os.path.splitext(audio_wav_path)[0]
    candidates = [base + ext for ext in (".mp4", ".mkv", ".webm", ".m4a")]
    src = None
    for c in candidates:
        if os.path.exists(c):
            src = c
            break

    if src is None:
        raise FileNotFoundError(f"Could not find video for {video_id}")
    
    out_dir = os.path.join(FRAMES, video_id)
    frames = sample_frames(src, out_dir, every_sec=every_sec)

    # Embeddings
    enc = ClipEncoder("clip-ViT-B-32")
    pil_images = [Image.open(f["path"]).convert("RGB") for f in frames]
    embs = enc.encode_images(pil_images) # NxD

    # Objects
    det = YoloDetector("yolov8n.pt")
    labels = det.detect_labels([f["path"] for f in frames])
    clip_vecs, clip_rows = build_clip_windows(frames, embs, labels, clip_len=2.0, stride=0.5)
    save_action_clips_index(video_id, clip_vecs, clip_rows)
    print(f"ACTION CLIPS OK: {video_id} | clips={len(clip_rows)}")

    # Rows for DB
    rows = []
    for i, f in enumerate(frames):
        rows.append({
            "start": float(f["t"]),
            "end":   float(f["t_end"]),
            "frame": os.path.relpath(f["path"], start=os.path.dirname(DATA)), 
            "objects": labels[i],
        })

    # Store
    save_visual_index(video_id, embs, rows)
    print(f"INGEST OK: {video_id} | frames={len(frames)}")

if __name__ == "__main__":
    import sys, os
    if len(sys.argv) < 2:
        print("Usage: ingest_visual.py <url_or_path> [every_sec]")
        sys.exit(1)

    url_or_path = sys.argv[1]
    every_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    ingest_visual(url_or_path, every_sec=every_sec)