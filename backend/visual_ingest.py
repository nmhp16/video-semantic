import os, json, subprocess
from typing import List, Dict
from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, SiglipModel
from store import (
    DATA,
    save_visual_metadata, save_action_clips_metadata,
    save_siglip_visual_index, save_siglip_action_clips_index,
)
import re
from pathlib import Path
from ingest import extract_audio

MEDIA = os.path.join(DATA, "media")
FRAMES = os.path.join(DATA, "frames")
os.makedirs(FRAMES, exist_ok=True)

YT_ID_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")

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
    if os.path.exists(out_mp4) and not _has_video_stream(out_mp4):
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

# --- Florence-2 captioning ---
class Florence2Captioner:
    def __init__(self, model_id: str = "microsoft/Florence-2-base"):
        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True
        ).to(self._device).eval()

    def _run_task(self, image: Image.Image, task: str, max_new_tokens: int = 128) -> dict:
        inputs = self.processor(text=task, images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            gen = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
            )
        text = self.processor.batch_decode(gen, skip_special_tokens=False)[0]
        return self.processor.post_process_generation(text, task=task, image_size=image.size)

    def process_image(self, image: Image.Image) -> Dict:
        """Return a detailed caption and caption-derived object keywords."""
        cap_result = self._run_task(image, "<MORE_DETAILED_CAPTION>", max_new_tokens=128)
        caption = cap_result.get("<MORE_DETAILED_CAPTION>", "")
        objects = _caption_keywords(caption)
        return {"caption": caption, "objects": objects}

    def process_images(self, pil_images: List[Image.Image]) -> List[Dict]:
        """Process a batch of images sequentially."""
        results = []
        for i, img in enumerate(pil_images):
            results.append(self.process_image(img))
            if (i + 1) % 10 == 0:
                print(f"  captioned {i+1}/{len(pil_images)} frames")
        return results


# --- SigLIP vision-text encoder (for direct frame embedding) ---
class SigLIPEncoder:
    def __init__(self, model_id: str = "google/siglip-base-patch16-224"):
        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = SiglipModel.from_pretrained(model_id).to(self._device).eval()

    def encode_images(self, pil_images: List[Image.Image], batch_size: int = 16) -> np.ndarray:
        all_embs = []
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self._device)
            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            all_embs.append(feats.float().cpu().numpy())
            if (i + batch_size) % 64 == 0 or i + batch_size >= len(pil_images):
                print(f"  siglip {min(i + batch_size, len(pil_images))}/{len(pil_images)} frames")
        return np.concatenate(all_embs, axis=0).astype("float32")

    def encode_image_paths(self, paths: List[str], batch_size: int = 16) -> np.ndarray:
        """Encode frames by path, opening only `batch_size` images at a time.
        Avoids holding every PIL image in memory simultaneously."""
        all_embs = []
        total = len(paths)
        for i in range(0, total, batch_size):
            batch_paths = paths[i:i + batch_size]
            batch = [Image.open(p).convert("RGB") for p in batch_paths]
            try:
                inputs = self.processor(images=batch, return_tensors="pt").to(self._device)
                with torch.no_grad():
                    feats = self.model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                all_embs.append(feats.float().cpu().numpy())
            finally:
                for img in batch:
                    img.close()
            if (i + batch_size) % 64 == 0 or i + batch_size >= total:
                print(f"  siglip {min(i + batch_size, total)}/{total} frames")
        return np.concatenate(all_embs, axis=0).astype("float32")

    def encode_text(self, texts: List[str]) -> np.ndarray:
        inputs = self.processor(
            text=texts, return_tensors="pt", padding="max_length", truncation=True
        ).to(self._device)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats.float().cpu().numpy().astype("float32")


# --- Build sliding-window action clips from caption embeddings ---
def build_caption_windows(frames, embs, captions_data, clip_len=2.0, stride=0.5):
    """Build sliding windows using caption embeddings (bge-small space)."""
    i, N = 0, len(frames)
    clip_vecs, rows = [], []
    while i < N:
        t0 = frames[i]["t"]
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
                for o in captions_data[k]["objects"]:
                    objs.add(o)

            # Use middle frame's caption as representative
            mid_idx = idxs[len(idxs) // 2]
            caption = captions_data[mid_idx]["caption"]

            clip_vecs.append(v)
            rows.append({
                "start": float(frames[idxs[0]]["t"]),
                "end":   float(frames[idxs[-1]]["t_end"]),
                "objects": sorted(list(objs)),
                "caption": caption,
            })
        # Stride
        t_next = t0 + stride
        while i < N and frames[i]["t"] < t_next:
            i += 1
    return np.stack(clip_vecs, axis=0).astype("float32"), rows


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

    # SigLIP frame embeddings (direct vision-text contrastive space — no captioning).
    # Encode in batches directly from disk so we never hold every frame in RAM.
    print(f"SigLIP-encoding {len(frames)} frames...")
    siglip = SigLIPEncoder("google/siglip-base-patch16-224")
    siglip_embs = siglip.encode_image_paths([f["path"] for f in frames], batch_size=16)

    # Placeholder captions_data — captions are generated lazily at query time.
    # build_caption_windows only uses the 'objects' field for aggregation, which
    # is an empty list when captions aren't pre-generated.
    captions_data = [{"caption": "", "objects": []} for _ in frames]

    # Action clips: sliding windows over SigLIP frame embeddings
    siglip_clip_vecs, clip_rows = build_caption_windows(
        frames, siglip_embs, captions_data, clip_len=2.0, stride=0.5
    )
    save_siglip_action_clips_index(video_id, siglip_clip_vecs)
    save_action_clips_metadata(video_id, clip_rows)
    print(f"ACTION CLIPS OK: {video_id} | clips={len(siglip_clip_vecs)}")

    # Frame-level metadata rows (captions/objects empty; filled lazily on query)
    rows = []
    for i, f in enumerate(frames):
        rows.append({
            "start": float(f["t"]),
            "end":   float(f["t_end"]),
            "frame": os.path.relpath(f["path"], start=os.path.dirname(DATA)),
            "objects": [],
            "caption": "",
        })

    save_siglip_visual_index(video_id, siglip_embs)
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
