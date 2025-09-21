import os, json, math, subprocess
from typing import List, Dict
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO
from store import DATA, save_visual_index

MEDIA = os.path.join(DATA, "media")
FRAMES = os.path.join(DATA, "frames")
os.makedirs(FRAMES, exist_ok=True)

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

def ingest_visual(video_id: str, audio_wav_path: str, every_sec: float = 1.0):
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