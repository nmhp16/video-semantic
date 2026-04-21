import os, json, subprocess, re, tempfile
from pathlib import Path
import whisper
from sentence_transformers import SentenceTransformer
import numpy as np
from chunking import chunk_segments
from store import DATA, save_index

MEDIA = os.path.join(DATA, "media")
TRANS = os.path.join(DATA, "transcripts")
os.makedirs(MEDIA, exist_ok=True)
os.makedirs(TRANS, exist_ok=True)

YT_ID_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")

def extract_audio(infile: str, outfile: str):
    subprocess.check_call([
        "ffmpeg", "-y", "-i", infile,
        "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        outfile
    ])

def transcribe(wav_path: str):
    model = whisper.load_model("turbo")
    result = model.transcribe(wav_path, word_timestamps=False)
    return [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        for s in result["segments"]
    ]

def embed_texts(texts: list[str]):
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return np.array(model.encode(texts, normalize_embeddings=True), dtype="float32")

def ingest(url_or_path: str):
    m = YT_ID_RE.search(url_or_path)
    video_id = m.group(1) if m else Path(url_or_path).stem

    # visual_ingest leaves a wav here as a handoff — use it if present
    wav_handoff = os.path.join(MEDIA, f"{video_id}.wav")

    if os.path.exists(wav_handoff):
        segments = transcribe(wav_handoff)
        os.remove(wav_handoff)  # clean up after use
    elif url_or_path.startswith("http"):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_audio = os.path.join(tmpdir, f"{video_id}_audio")
            subprocess.check_call([
                "yt-dlp", "-f", "bestaudio/best",
                "--extractor-args", "youtube:player_client=android",
                "-o", tmp_audio, url_or_path
            ])
            tmp_wav = os.path.join(tmpdir, f"{video_id}.wav")
            # find whatever yt-dlp saved (extension varies)
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
