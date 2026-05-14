import os, subprocess, json
from typing import List, Dict
from PIL import Image
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from store import (
    DATA,
    save_visual_metadata, save_action_clips_metadata,
    save_siglip_visual_index, save_xclip_action_index,
)
from db import put_cached_captions
import re
from pathlib import Path
from ingest import extract_audio
from utils_unified import YT_ID_RE

MEDIA = os.path.join(DATA, "media")
FRAMES = os.path.join(DATA, "frames")
os.makedirs(FRAMES, exist_ok=True)

def _has_video_stream(path: str) -> bool:
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0", path
        ]).decode().strip()
        return out == "video"
    except Exception:
        return False

def ytdlp_video(url: str, out_mp4: str):
    if os.path.exists(out_mp4) and _has_video_stream(out_mp4):
        return  # already downloaded and valid
    if os.path.exists(out_mp4):
        os.remove(out_mp4)
    from ingest import _ytdlp_auth_args
    subprocess.check_call([
        "yt-dlp", "-f", "bv*+ba/b", "--merge-output-format", "mp4",
        "-o", out_mp4, *_ytdlp_auth_args(), url
    ])

# --- Frame sampling (scene-change + max-gap) ---
_META_FRAME_RE = re.compile(r"^frame:\d+.*pts_time:([0-9.]+)")
_META_SCENE_RE = re.compile(r"^lavfi\.scene_score=([0-9.]+)")

def sample_frames(video_path: str, out_dir: str,
                  max_gap_sec: float = 2.0,
                  scene_thresh: float = 0.3) -> List[Dict]:
    """Emit a frame when the scene changes OR when more than max_gap_sec
    has passed since the last emitted frame. The first frame is always emitted.
    Returns one dict per written JPEG with t, t_end, and is_scene_change.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Infer duration using ffprobe (for clipping the last frame's t_end)
    probe = subprocess.check_output([
        "ffprobe","-v","error","-show_entries","format=duration",
        "-of","default=noprint_wrappers=1:nokey=1", video_path
    ]).decode().strip()
    duration = float(probe) if probe else 0.0

    ts_path = os.path.join(out_dir, "_frames_ts.txt")
    if os.path.exists(ts_path):
        os.remove(ts_path)

    # select=<expr>: emit frame if scene-change OR first frame OR gap elapsed.
    # Commas inside function calls are escaped with \\, to survive the filter parser.
    filt = (
        f"select='gt(scene\\,{scene_thresh})"
        f"+eq(n\\,0)"
        f"+gte(t-prev_selected_t\\,{max_gap_sec})',"
        f"scale=-2:720,"
        f"metadata=print:file={ts_path}"
    )

    subprocess.check_call([
        "ffmpeg","-y","-i", video_path,
        "-vf", filt,
        "-fps_mode", "vfr",
        "-q:v", "3",
        os.path.join(out_dir, "frame-%06d.jpg")
    ])

    # Parse per-frame pts_time and scene score from the metadata sidecar.
    # Format: "frame:N pts:P pts_time:T" followed by "lavfi.scene_score=S"
    timestamps: List[float] = []
    scene_scores: List[float] = []
    if os.path.exists(ts_path):
        with open(ts_path) as f:
            pending_ts: float | None = None
            for line in f:
                m = _META_FRAME_RE.match(line)
                if m:
                    pending_ts = float(m.group(1))
                    continue
                m = _META_SCENE_RE.match(line)
                if m and pending_ts is not None:
                    timestamps.append(pending_ts)
                    scene_scores.append(float(m.group(1)))
                    pending_ts = None

    frame_files = sorted([
        f for f in os.listdir(out_dir)
        if f.startswith("frame-") and f.endswith(".jpg")
    ])

    out = []
    for i, fn in enumerate(frame_files):
        t = timestamps[i] if i < len(timestamps) else i * max_gap_sec
        score = scene_scores[i] if i < len(scene_scores) else 0.0
        if i + 1 < len(timestamps):
            t_end = min(duration, timestamps[i + 1])
        else:
            t_end = min(duration, t + max_gap_sec)
        # First frame (i==0) is forced by eq(n,0), treat as scene change
        is_scene_change = (i == 0) or (score > scene_thresh)
        out.append({
            "path": os.path.join(out_dir, fn),
            "t": t,
            "t_end": t_end,
            "is_scene_change": is_scene_change,
        })

    return out

# --- Caption keyword extraction (replaces Florence-2 <OD> pass) ---
_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_STOPWORDS = {
    "the","and","with","that","this","from","into","onto","over","under","near","upon",
    "very","some","many","much","more","most","less","just","also","their","there","here",
    "which","while","when","where","what","about","above","below","then","than","they",
    "them","these","those","have","has","had","are","was","were","been","being","its",
    "his","her","him","she","you","your","our","ours","theirs","itself",
    "appears","looking","visible","background","foreground","image","picture","photo",
}

def _caption_keywords(caption: str) -> List[str]:
    if not caption:
        return []
    words = (w.lower() for w in _WORD_RE.findall(caption))
    return sorted({w for w in words if w not in _STOPWORDS})

# --- SigLIP vision-text encoder for frame-level visual search ---
class SigLIPEncoder:
    def __init__(self, model_id: str = "google/siglip-base-patch16-224"):
        from transformers import SiglipModel, SiglipProcessor
        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        self.model = SiglipModel.from_pretrained(model_id).to(self._device).eval()
        self.processor = SiglipProcessor.from_pretrained(model_id)

    def encode_image_paths(self, paths: List[str], batch_size: int = 16) -> np.ndarray:
        total = len(paths)
        results = []
        for i in range(0, total, batch_size):
            batch_paths = paths[i:i + batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            try:
                inputs = self.processor(images=imgs, return_tensors="pt",
                                        padding="max_length").to(self._device)
                with torch.no_grad():
                    feats = self.model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                results.append(feats.float().cpu().numpy())
            finally:
                for img in imgs:
                    img.close()
            print(f"  siglip {min(i + batch_size, total)}/{total} frames")
        return np.concatenate(results, axis=0).astype("float32")

    def encode_text(self, texts: List[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt",
                                padding="max_length").to(self._device)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats.float().cpu().numpy().astype("float32")


# --- X-CLIP video-language encoder for action search ---
class XCLIPEncoder:
    N_FRAMES = 8

    def __init__(self, model_id: str = "microsoft/xclip-base-patch32"):
        from transformers import XCLIPProcessor, XCLIPModel
        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        self.processor = XCLIPProcessor.from_pretrained(model_id)
        self.model = XCLIPModel.from_pretrained(model_id).to(self._device).eval()

    def _uniform_sample(self, paths: List[str]) -> List[Image.Image]:
        n = self.N_FRAMES
        if len(paths) >= n:
            idxs = np.linspace(0, len(paths) - 1, n, dtype=int)
            selected = [paths[i] for i in idxs]
        else:
            reps = (n // len(paths)) + 1
            selected = (paths * reps)[:n]
        return [Image.open(p).convert("RGB") for p in selected]

    def encode_clip_paths(self, paths: List[str]) -> np.ndarray:
        frames = self._uniform_sample(paths)
        try:
            return self.encode_clip(frames)
        finally:
            for img in frames:
                img.close()

    def encode_clip(self, pil_frames: List[Image.Image]) -> np.ndarray:
        inputs = self.processor(videos=[pil_frames], return_tensors="pt").to(self._device)
        with torch.no_grad():
            feats = self.model.get_video_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats.float().cpu().numpy()[0]

    def encode_text(self, texts: List[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self._device)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats.float().cpu().numpy().astype("float32")

    def encode_frames_batch(self, paths: List[str], batch_size: int = 8) -> np.ndarray:
        """Encode individual frames by treating each as a duplicated 8-frame clip."""
        total = len(paths)
        results = []
        for i in range(0, total, batch_size):
            batch_paths = paths[i:i + batch_size]
            batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            try:
                videos = [[img] * self.N_FRAMES for img in batch_imgs]
                inputs = self.processor(videos=videos, return_tensors="pt").to(self._device)
                with torch.no_grad():
                    feats = self.model.get_video_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                results.append(feats.float().cpu().numpy())
            finally:
                for img in batch_imgs:
                    img.close()
            print(f"  xclip frames {min(i + batch_size, total)}/{total}")
        return np.concatenate(results, axis=0).astype("float32")


def build_xclip_clips(frames, xclip: XCLIPEncoder,
                      captions_data=None, clip_len: float = 4.0, stride: float = 2.0):
    """Sliding-window action clips encoded with X-CLIP video tower."""
    if not frames:
        return np.zeros((0, 512), dtype="float32"), []

    frame_to_idx = {f["path"]: i for i, f in enumerate(frames)}
    clip_vecs, clip_rows = [], []
    duration = frames[-1]["t_end"]
    t = frames[0]["t"]

    while t < duration:
        t_end = min(t + clip_len, duration)
        window = [f for f in frames if t <= f["t"] < t_end]
        if not window:
            t += stride
            continue

        emb = xclip.encode_clip_paths([f["path"] for f in window])
        clip_vecs.append(emb)

        mid = window[len(window) // 2]
        idx = frame_to_idx.get(mid["path"])
        caption = captions_data[idx]["caption"] if (captions_data and idx is not None) else ""
        objects = captions_data[idx]["objects"] if (captions_data and idx is not None) else []

        clip_rows.append({
            "start": float(window[0]["t"]),
            "end": float(min(window[-1]["t_end"], t_end)),
            "objects": objects,
            "caption": caption,
        })
        t += stride

    if not clip_vecs:
        return np.zeros((0, 512), dtype="float32"), []
    return np.stack(clip_vecs).astype("float32"), clip_rows


# --- Florence-2-large captioning ---
class Florence2Captioner:
    _MODEL_ID = "microsoft/Florence-2-large"

    def __init__(self):
        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        # float32 required — float16 has dtype mismatches on MPS with Florence-2
        self.model = AutoModelForCausalLM.from_pretrained(
            self._MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32,
        ).to(self._device).eval()
        self.processor = AutoProcessor.from_pretrained(self._MODEL_ID, trust_remote_code=True)

    def caption_batch(self, imgs: List[Image.Image]) -> List[Dict]:
        """One forward pass for the whole batch — much faster than one-at-a-time."""
        if not imgs:
            return []
        inputs = self.processor(
            text=["<MORE_DETAILED_CAPTION>"] * len(imgs),
            images=imgs,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=128, num_beams=1)
        raws = self.processor.batch_decode(ids, skip_special_tokens=False)
        results = []
        for raw, img in zip(raws, imgs):
            caption = self.processor.post_process_generation(
                raw, task="<MORE_DETAILED_CAPTION>", image_size=img.size
            )["<MORE_DETAILED_CAPTION>"]
            results.append({"caption": caption, "objects": _caption_keywords(caption)})
        return results


# --- YOLOv8 COCO object detector (augments Florence-2 objects list) ---
class YOLODetector:
    def __init__(self, model_id: str = "yolov8s.pt", conf: float = 0.15):
        from ultralytics import YOLO as _YOLO
        self._model = _YOLO(model_id)
        self._conf = conf

    def detect(self, image_path: str) -> List[str]:
        results = self._model.predict(image_path, conf=self._conf, verbose=False)
        labels: set = set()
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                labels.add(r.names[cls_id].lower())
        return sorted(labels)


# Module-level singletons — kept warm between jobs in the persistent worker process.
_captioner_instance: "Florence2Captioner | None" = None
_siglip_instance: "SigLIPEncoder | None" = None
_xclip_instance: "XCLIPEncoder | None" = None
_yolo_instance: "YOLODetector | None" = None


def _get_captioner() -> Florence2Captioner:
    global _captioner_instance
    if _captioner_instance is None:
        _captioner_instance = Florence2Captioner()
    return _captioner_instance


def _get_siglip() -> SigLIPEncoder:
    global _siglip_instance
    if _siglip_instance is None:
        _siglip_instance = SigLIPEncoder()
    return _siglip_instance


def _get_xclip() -> XCLIPEncoder:
    global _xclip_instance
    if _xclip_instance is None:
        _xclip_instance = XCLIPEncoder()
    return _xclip_instance


def _get_yolo() -> YOLODetector:
    global _yolo_instance
    if _yolo_instance is None:
        _yolo_instance = YOLODetector()
    return _yolo_instance




def ingest_visual(url_or_path: str, max_gap_sec: float = 2.0, scene_thresh: float = 0.3,
                  progress_cb=None):
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
    if progress_cb: progress_cb("Extracting frames…")
    frames = sample_frames(src, out_dir, max_gap_sec=max_gap_sec, scene_thresh=scene_thresh)

    # SigLIP frame embeddings — image-text contrastive space, better for per-frame visual search
    if progress_cb: progress_cb(f"Encoding {len(frames)} frames (SigLIP)…")
    siglip = _get_siglip()
    print(f"SigLIP encoding {len(frames)} frames...")
    frame_embs = siglip.encode_image_paths([f["path"] for f in frames], batch_size=16)

    # Resumable caption checkpoint — skip frames already captioned on a prior run.
    _ckpt_dir = os.path.join(DATA, "checkpoints")
    os.makedirs(_ckpt_dir, exist_ok=True)
    _ckpt_path = os.path.join(_ckpt_dir, f"{video_id}_captions.json")
    scene_captions: dict[int, Dict] = {}
    if os.path.exists(_ckpt_path):
        try:
            with open(_ckpt_path) as _f:
                scene_captions = {int(k): v for k, v in json.load(_f).items()}
            print(f"Resuming from checkpoint: {len(scene_captions)} captions already done")
        except Exception:
            scene_captions = {}

    scene_idxs = [i for i, f in enumerate(frames) if f.get("is_scene_change", True)]
    remaining = [i for i in scene_idxs if i not in scene_captions]
    total_scene = len(scene_idxs)

    if remaining:
        already_done = total_scene - len(remaining)
        if progress_cb: progress_cb(f"Captioning {already_done}/{total_scene} scene frames (Florence-2)…")
        print(f"Florence-2 captioning {len(remaining)}/{total_scene} scene-change frames...")
        captioner = _get_captioner()
        for rank, idx in enumerate(remaining):
            img = Image.open(frames[idx]["path"]).convert("RGB")
            scene_captions[idx] = captioner.caption_batch([img])[0]
            img.close()
            done = already_done + rank + 1
            if progress_cb: progress_cb(f"Captioning {done}/{total_scene} scene frames (Florence-2)…")
            if (rank + 1) % 5 == 0 or rank + 1 == len(remaining):
                print(f"  captioned {done}/{total_scene} scene frames")
                with open(_ckpt_path, "w") as _f:
                    json.dump(scene_captions, _f)
    else:
        print(f"All {total_scene} scene captions loaded from checkpoint")

    # Propagate scene captions to gap-fill frames (forward-fill)
    captions_data: List[Dict] = []
    last = {"caption": "", "objects": []}
    for i in range(len(frames)):
        if i in scene_captions:
            last = scene_captions[i]
        captions_data.append(last)

    # YOLO augmentation — detect COCO objects on every keyframe and merge into objects.
    # Each frame takes ~50ms so this adds only a few seconds total.
    # This catches objects Florence-2 hallucinated over (e.g. knife described as sandwich).
    if progress_cb: progress_cb("Detecting objects (YOLO)…")
    print(f"YOLO detecting objects on {len(frames)} frames…")
    yolo = _get_yolo()
    for i, f in enumerate(frames):
        detected = yolo.detect(f["path"])
        if detected:
            existing = set(captions_data[i].get("objects") or [])
            merged = sorted(existing | set(detected))
            captions_data[i] = {**captions_data[i], "objects": merged}

    frame_rel = [os.path.relpath(f["path"], start=os.path.dirname(DATA)) for f in frames]
    put_cached_captions(video_id, {k: v for k, v in zip(frame_rel, captions_data)})
    print(f"Cached {len(captions_data)} captions for {video_id}")

    # Action clips: X-CLIP sliding windows — reuse singleton
    if progress_cb: progress_cb("Encoding action clips (X-CLIP)…")
    xclip = _get_xclip()
    print(f"X-CLIP encoding action clips for {video_id}...")
    xclip_clip_vecs, clip_rows = build_xclip_clips(
        frames, xclip, captions_data=captions_data, clip_len=4.0, stride=2.0
    )
    save_xclip_action_index(video_id, xclip_clip_vecs)
    save_action_clips_metadata(video_id, clip_rows)
    print(f"ACTION CLIPS OK: {video_id} | clips={len(xclip_clip_vecs)}")

    rows = []
    for i, f in enumerate(frames):
        rows.append({
            "start": float(f["t"]),
            "end":   float(f["t_end"]),
            "frame": os.path.relpath(f["path"], start=os.path.dirname(DATA)),
            "objects": captions_data[i]["objects"],
            "caption": captions_data[i]["caption"],
        })

    save_siglip_visual_index(video_id, frame_embs)
    save_visual_metadata(video_id, rows)
    try:
        os.remove(_ckpt_path)
    except FileNotFoundError:
        pass
    print(f"INGEST OK: {video_id} | frames={len(frames)}")

if __name__ == "__main__":
    import sys, os
    if len(sys.argv) < 2:
        print("Usage: ingest_visual.py <url_or_path> [max_gap_sec] [scene_thresh]")
        sys.exit(1)

    url_or_path = sys.argv[1]
    max_gap_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    scene_thresh = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3

    ingest_visual(url_or_path, max_gap_sec=max_gap_sec, scene_thresh=scene_thresh)
