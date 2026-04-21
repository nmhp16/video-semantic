import os, json, subprocess, re, tempfile
from pathlib import Path
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
import numpy as np
from chunking import chunk_segments
from store import DATA, save_index
from supabase_client import sb_enabled

MEDIA = os.path.join(DATA, "media")
TRANS = os.path.join(DATA, "transcripts")

YT_ID_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")

def extract_audio(infile: str, outfile: str):
    subprocess.check_call([
        "ffmpeg", "-y", "-i", infile,
        "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        outfile
    ])

def transcribe(wav_path: str):
    # int8 quantization is 4-8x faster than fp32 on CPU with minimal quality loss
    model = WhisperModel("turbo", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(wav_path, word_timestamps=False, beam_size=1)
    return [
        {"start": s.start, "end": s.end, "text": s.text.strip()}
        for s in segments
    ]

def embed_texts(texts: list[str]):
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return np.array(model.encode(texts, normalize_embeddings=True), dtype="float32")

def ingest(url_or_path: str, segments: list | None = None):
    m = YT_ID_RE.search(url_or_path)
    video_id = m.group(1) if m else Path(url_or_path).stem

    if segments is None:
        if url_or_path.startswith("http"):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_audio = os.path.join(tmpdir, f"{video_id}_audio")
                subprocess.check_call([
                    "yt-dlp", "-f", "bestaudio/best",
                    "--extractor-args", "youtube:player_client=android",
                    "-o", tmp_audio, url_or_path
                ])
                tmp_wav = os.path.join(tmpdir, f"{video_id}.wav")
                downloaded = next(
                    (os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.startswith(f"{video_id}_audio")),
                    tmp_audio
                )
                extract_audio(downloaded, tmp_wav)
                segments = transcribe(tmp_wav)
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_wav = os.path.join(tmpdir, f"{video_id}.wav")
                extract_audio(url_or_path, tmp_wav)
                segments = transcribe(tmp_wav)

    if not sb_enabled():
        os.makedirs(TRANS, exist_ok=True)
        with open(os.path.join(TRANS, f"{video_id}.json"), "w") as f:
            json.dump(segments, f)

    chunks = chunk_segments(segments, max_sec=20, stride_sec=5)
    embeddings = embed_texts([c["text"] for c in chunks])
    save_index(video_id, embeddings, chunks)
    print(f"INGEST OK: {video_id} | chunks={len(chunks)}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: ingest.py <url_or_path>")
        sys.exit(1)
    ingest(sys.argv[1])
