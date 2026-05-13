import os, subprocess
from typing import List, Dict
from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
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

def sample_frames(video_path: str, out_dir: str,
                  max_gap_sec: float = 5.0,
                  scene_thresh: float = 0.3) -> List[Dict]:
    """Emit a frame when the scene changes OR when more than max_gap_sec
    has passed since the last emitted frame. The first frame is always emitted.
    Returns one dict per written JPEG with its real pts_time-derived t / t_end.
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

    # Parse per-frame pts_time from the metadata sidecar
    timestamps: List[float] = []
    if os.path.exists(ts_path):
        with open(ts_path) as f:
            for line in f:
                m = _META_FRAME_RE.match(line)
                if m:
                    timestamps.append(float(m.group(1)))

    frame_files = sorted([
        f for f in os.listdir(out_dir)
        if f.startswith("frame-") and f.endswith(".jpg")
    ])

    out = []
    for i, fn in enumerate(frame_files):
        t = timestamps[i] if i < len(timestamps) else i * max_gap_sec
        if i + 1 < len(timestamps):
            t_end = min(duration, timestamps[i + 1])
        else:
            t_end = min(duration, t + max_gap_sec)
        out.append({
            "path": os.path.join(out_dir, fn),
            "t": t,
            "t_end": t_end,
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


# --- Moondream2 captioning (default) ---
class Moondream2Captioner:
    _REVISION = "2025-01-09"

    def __init__(self, model_id: str = "vikhyatk/moondream2"):
        if torch.cuda.is_available():
            self._device = "cuda"
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            self._device = "mps"
            dtype = torch.float16
        else:
            self._device = "cpu"
            dtype = torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=self._REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            revision=self._REVISION,
            torch_dtype=dtype,
        ).to(self._device).eval()

    def process_image(self, image: Image.Image) -> Dict:
        enc = self.model.encode_image(image)
        caption = self.model.answer_question(
            enc, "Describe what you see in this image in detail.", self.tokenizer
        )
        return {"caption": caption, "objects": _caption_keywords(caption)}

    @staticmethod
    def _resize_for_caption(img: Image.Image, max_side: int = 512) -> Image.Image:
        w, h = img.size
        if max(w, h) <= max_side:
            return img
        scale = max_side / max(w, h)
        return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    def process_images(self, pil_images: List[Image.Image]) -> List[Dict]:
        n = len(pil_images)
        results = []
        for i, img in enumerate(pil_images):
            enc = self.model.encode_image(self._resize_for_caption(img))
            caption = self.model.answer_question(
                enc, "Describe what you see in this image in detail.", self.tokenizer
            )
            del enc
            results.append({"caption": caption, "objects": _caption_keywords(caption)})
            if (i + 1) % 5 == 0 or i + 1 == n:
                print(f"  captioned {i+1}/{n} frames")
        return results




def ingest_visual(url_or_path: str, max_gap_sec: float = 5.0, scene_thresh: float = 0.3):
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
    frames = sample_frames(src, out_dir, max_gap_sec=max_gap_sec, scene_thresh=scene_thresh)

    # X-CLIP frame embeddings (visual search)
    xclip = XCLIPEncoder()
    print(f"X-CLIP encoding {len(frames)} frames...")
    frame_embs = xclip.encode_frames_batch([f["path"] for f in frames], batch_size=8)

    # Free X-CLIP from GPU before loading Moondream2 — both use MPS and would OOM together
    del xclip
    import gc; gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Moondream2 captions (encode+caption one frame at a time to avoid accumulating KV caches)
    print(f"Moondream2 captioning {len(frames)} frames...")
    captioner = Moondream2Captioner()
    pil_images = [Image.open(f["path"]).convert("RGB") for f in frames]
    captions_data = captioner.process_images(pil_images)
    del pil_images, captioner
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    frame_rel = [os.path.relpath(f["path"], start=os.path.dirname(DATA)) for f in frames]
    put_cached_captions(video_id, {k: v for k, v in zip(frame_rel, captions_data)})
    print(f"Cached {len(captions_data)} captions for {video_id}")

    # Action clips: X-CLIP sliding windows (reload X-CLIP with free GPU memory)
    xclip = XCLIPEncoder()
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
