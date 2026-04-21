import os, json, subprocess, tempfile, shutil
from typing import List, Dict
from PIL import Image
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForCausalLM
from store import DATA, save_visual_index, save_action_clips_index
from supabase_client import sb_enabled
import re
from pathlib import Path
from ingest import extract_audio

MEDIA  = os.path.join(DATA, "media")
FRAMES = os.path.join(DATA, "frames")

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

    def process_images(self, pil_images: List[Image.Image], batch_size: int = 8) -> List[Dict]:
        task = "<MORE_DETAILED_CAPTION>"
        results = []
        for start in range(0, len(pil_images), batch_size):
            batch = pil_images[start:start + batch_size]
            inputs = self.processor(
                text=[task] * len(batch),
                images=batch,
                return_tensors="pt",
                padding=True,
            ).to(self._device)
            with torch.no_grad():
                gen = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=128,
                    num_beams=1,
                    do_sample=False,
                )
            texts = self.processor.batch_decode(gen, skip_special_tokens=False)
            for text, img in zip(texts, batch):
                parsed  = self.processor.post_process_generation(text, task=task, image_size=img.size)
                caption = parsed.get(task, "")
                results.append({"caption": caption, "objects": _caption_keywords(caption)})
            print(f"  captioned {min(start + batch_size, len(pil_images))}/{len(pil_images)} frames")
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


def ingest_visual(url_or_path: str, every_sec: float = 2.0):
    m = YT_ID_RE.search(url_or_path)
    video_id = m.group(1) if m else Path(url_or_path).stem

    # All working files go into a single tmpdir — no permanent local dirs created
    with tempfile.TemporaryDirectory() as workdir:
        out_dir     = os.path.join(workdir, "frames") if sb_enabled() else os.path.join(FRAMES, video_id)
        wav_handoff = os.path.join(workdir, f"{video_id}.wav")

        def _run(video_path: str):
            nonlocal frames
            extract_audio(video_path, wav_handoff)
            frames = sample_frames(video_path, out_dir, every_sec=every_sec)

        frames = []

        if url_or_path.startswith("http"):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_video = os.path.join(tmpdir, "video.mp4")
                subprocess.check_call([
                    "yt-dlp", "-f", "bv*+ba/b", "--merge-output-format", "mp4",
                    "--extractor-args", "youtube:player_client=android",
                    "-o", tmp_video, url_or_path
                ])
                _run(tmp_video)
        else:
            _run(url_or_path)

        # Caption every frame with Florence-2
        print(f"Captioning {len(frames)} frames with Florence-2...")
        captioner     = Florence2Captioner("microsoft/Florence-2-base")
        pil_images    = [Image.open(f["path"]).convert("RGB") for f in frames]
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
        # abs path when Supabase (tempdir), relative to BASE when local
        rows = [{
            "start":   float(f["t"]),
            "end":     float(f["t_end"]),
            "frame":   f["path"] if sb_enabled() else os.path.relpath(f["path"], start=BASE),
            "objects": captions_data[i]["objects"],
            "caption": captions_data[i]["caption"],
        } for i, f in enumerate(frames)]

        save_visual_index(video_id, embs, rows)
        print(f"VISUAL INGEST OK: {video_id} | frames={len(frames)}")

        # Transcribe while wav is still alive inside this workdir
        from ingest import transcribe as do_transcribe
        print("Transcribing audio...")
        segments = do_transcribe(wav_handoff)
        return segments  # caller passes these to ingest() to skip re-download
