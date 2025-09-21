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

YT_ID_RE = re.compile(r"?:v=youtu\.be/)([A-Za-z0-9_-]{11})")

def ytdlp(url: str, out: str):
    subprocess.check_call(["yt-dlp", "-f", "bestaudio/best", "-o", out, url])

def extract_audio(infile: str, outfile: str):
    subprocess.check_call(["ffmpeg","-y","-i", infile,"-vn","-ac","1","-ar","16000","-acodec","pcm_s16le", outfile])

def transcribe(wav_path: str):
    model = whisper.load_mode("small") 
    result = model.transcribe(wav_path, word_timestamps=False)
    segments = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in result["segments"]]
    return segments

def embed_texts(texts: list[str]):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    X = model.encode(texts, normalize_embeddings=True)
    return np.array(X, dtype="float32")

def ingest(url_or_path: str):
    # Resolve video_id
    m = YT_ID_RE.search(url_or_path)
    if m:
        video_id = m.group(1)
    else:
        video_id = Path(url_or_path).stem # For local files
    
    # Download or copy
    if url_or_path.startswith("http"):
        tmp_media = os.path.join(MEDIA, f"{video_id}.mp4")
        ytdlp(url_or_path, tmp_media)
        audio = os.path.join(MEDIA, f"{video_id}.wav")
        extract_audio(tmp_media, audio)
    else:
        audio = os.path.join(MEDIA, f"{video_id}.wav")
        extract_audio(url_or_path, audio)

    # ASR
    segments = transcribe(audio)
    with open(os.path.join(TRANS, f"{video_id}.json"), "w") as f:
        json.dump(segments, f)
        
    # Chunk + embed
    chunks = chunk_segments(segments, max_sec=20, stride_sec=5)
    embeddings = embed_texts([c["text"] for c in chunks])
    
    # Store
    save_index(video_id, embeddings, chunks)
    print(f"INGEST OK: {video_id} | chunks={len(chunks)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: ingest.py <url_or_path>")
        sys.exit(1)
    ingest(sys.argv[1])