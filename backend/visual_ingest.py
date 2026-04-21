import os, json, subprocess, tempfile
from typing import List, Dict
from PIL import Image
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForCausalLM
from store import DATA, save_visual_index, save_action_clips_index
import re
from pathlib import Path
from ingest import extract_audio

MEDIA  = os.path.join(DATA, "media")
FRAMES = os.path.join(DATA, "frames")
os.makedirs(MEDIA, exist_ok=True)
os.makedirs(FRAMES, exist_ok=True)

YT_ID_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")

# --- Frame sampling ---
def sample_frames(video_path: str, out_dir: str, every_sec: float = 1.0) -> List[Dict]:
    os.makedirs(out_dir, exist_ok=True)

    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]).decode().strip()
    duration = float(probe) if probe else 0.0

    fps = max(0.0001, 1.0 / every_sec)
    subprocess.check_call([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps},scale=-2:720",
        "-q:v", "3",
        os.path.join(out_dir, "frame-%06d.jpg")
    ])

    frames = sorted([f for f in os.listdir(out_dir) if f.startswith("frame-") and f.endswith(".jpg")])
    return [
        {"path": os.path.join(out_dir, fn), "t": i * every_sec, "t_end": min(duration, (i + 1) * every_sec)}
        for i, fn in enumerate(frames)
    ]


# --- Caption keyword extraction ---
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
        cap_result = self._run_task(image, "<MORE_DETAILED_CAPTION>", max_new_tokens=128)
        caption = cap_result.get("<MORE_DETAILED_CAPTION>", "")
        return {"caption": caption, "objects": _caption_keywords(caption)}

    def process_images(self, pil_images: List[Image.Image]) -> List[Dict]:
        results = []
        for i, img in enumerate(pil_images):
            results.append(self.process_image(img))
            if (i + 1) % 10 == 0:
                print(f"  captioned {i+1}/{len(pil_images)} frames")
        return results


# --- Sliding-window action clips ---
def build_caption_windows(frames, embs, captions_data, clip_len=2.0, stride=0.5):
    i, N = 0, len(frames)
    clip_vecs, rows = [], []
    while i < N:
        t0   = frames[i]["t"]
        idxs = [j for j in range(i, N) if frames[j]["t"] <= t0 + clip_len]
        if idxs:
            v = embs[idxs].mean(axis=0)
            v /= (np.linalg.norm(v) + 1e-12)
            objs = sorted({o for k in idxs for o in captions_data[k]["objects"]})
            mid  = idxs[len(idxs) // 2]
            clip_vecs.append(v)
            rows.append({
                "start":   float(frames[idxs[0]]["t"]),
                "end":     float(frames[idxs[-1]]["t_end"]),
                "objects": objs,
                "caption": captions_data[mid]["caption"],
            })
        t_next = t0 + stride
        while i < N and frames[i]["t"] < t_next:
            i += 1
    return np.stack(clip_vecs).astype("float32"), rows


def ingest_visual(url_or_path: str, every_sec: float = 1.0):
    m = YT_ID_RE.search(url_or_path)
    video_id = m.group(1) if m else Path(url_or_path).stem

    out_dir       = os.path.join(FRAMES, video_id)
    wav_handoff   = os.path.join(MEDIA, f"{video_id}.wav")  # temp wav for ingest.py

    if url_or_path.startswith("http"):
        # Download to a temp file — deleted after processing
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_video = tmp.name
        try:
            subprocess.check_call([
                "yt-dlp", "-f", "bv*+ba/b", "--merge-output-format", "mp4",
                "--extractor-args", "youtube:player_client=android",
                "-o", tmp_video, url_or_path
            ])
            extract_audio(tmp_video, wav_handoff)
            frames = sample_frames(tmp_video, out_dir, every_sec=every_sec)
        finally:
            if os.path.exists(tmp_video):
                os.remove(tmp_video)  # video not kept
    else:
        extract_audio(url_or_path, wav_handoff)
        frames = sample_frames(url_or_path, out_dir, every_sec=every_sec)

    # Caption every frame with Florence-2
    print(f"Captioning {len(frames)} frames with Florence-2...")
    captioner    = Florence2Captioner("microsoft/Florence-2-base")
    pil_images   = [Image.open(f["path"]).convert("RGB") for f in frames]
    captions_data = captioner.process_images(pil_images)

    # Embed captions
    print("Embedding captions...")
    emb_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embs = emb_model.encode([cd["caption"] for cd in captions_data], normalize_embeddings=True)
    embs = np.array(embs, dtype="float32")

    # Action clip index (sliding windows)
    clip_vecs, clip_rows = build_caption_windows(frames, embs, captions_data)
    save_action_clips_index(video_id, clip_vecs, clip_rows)
    print(f"ACTION CLIPS OK: {video_id} | clips={len(clip_rows)}")

    # Visual frame index
    rows = [{
        "start":   float(f["t"]),
        "end":     float(f["t_end"]),
        "frame":   os.path.relpath(f["path"], start=os.path.dirname(DATA)),
        "objects": captions_data[i]["objects"],
        "caption": captions_data[i]["caption"],
    } for i, f in enumerate(frames)]

    save_visual_index(video_id, embs, rows)
    print(f"VISUAL INGEST OK: {video_id} | frames={len(frames)}")
