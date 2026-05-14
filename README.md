# video-semantic

Multi-modal video search. Ingest a YouTube URL or local file; search by transcript content, visual appearance, or on-screen activity.

## How it works

Ingestion runs two parallel pipelines:

- **Audio** — `mlx-whisper` transcribes the audio track. Chunks are embedded with `BAAI/bge-small-en-v1.5` and stored in a FAISS index.
- **Visual** — `ffmpeg` samples keyframes (scene-change detection + 2 s max gap). `Moondream2` generates a caption and keyword labels for each scene-change frame; `YOLOv8` (COCO, 80 classes) provides a second, calibrated pass at object detection. Frame embeddings are built with `SigLIP`; sliding-window clip embeddings with `X-CLIP`.

At query time the `auto` mode fuses all signals: transcript hits, SigLIP frame similarity, X-CLIP clip similarity, and direct YOLO object-label matching.

## Stack

| Layer | Choice |
|---|---|
| API | FastAPI |
| UI | Vite + React + TypeScript |
| Transcription | mlx-whisper |
| Frame captioning | Moondream2 |
| Object detection (ingest) | YOLOv8s (COCO 80-class) |
| Frame embeddings | SigLIP |
| Clip embeddings | X-CLIP |
| Text embeddings | BGE-small-en-v1.5 |
| Vector search | FAISS (inner-product, L2-normalised) |
| Metadata | SQLite |

## Project layout

```
video-semantic/
  backend/
    app.py              FastAPI entry point, startup warm-up
    models.py           Pydantic request/response types
    ingest.py           Audio pipeline: download, Whisper, text index
    visual_ingest.py    Visual pipeline: frame sampling, Moondream2,
                        YOLOv8, SigLIP + X-CLIP indexing
    ingest_worker.py    Per-job subprocess entry point (process isolation)
    context.py          Per-video context vectors for global pre-filtering
    indexes.py          FAISS read/write with LRU cache
    db.py               SQLite schema and queries
    embeddings.py       BGE singleton
    chunking.py         Transcript chunking
    utils_unified.py    Video-id extraction from URLs
    routers/
      ingest.py         Ingest endpoints and job queue
      search.py         Search and retrieval logic
      videos.py         Library management endpoints
    requirements.txt
    data/               Runtime artifacts — see Data layout below
  frontend/             Vite + React UI
```

## Requirements

- Python 3.10+
- `ffmpeg` and `ffprobe` on `PATH`
- Apple Silicon (MPS) or CUDA recommended; CPU-only is supported but slow

### Caption model

Ingest uses Moondream2 for scene captioning.

| | Moondream2 |
|---|---|
| RAM | ~3.7 GB (float16 on MPS/CPU), ~1.5 GB (4-bit NF4 on CUDA) |
| Speed | ~13–14 s/frame on Apple Silicon MPS, ~10–12 s/frame on CUDA |
| CPU only | Usable (~30–60 s/frame, bfloat16) |

## Setup

### Apple Silicon (MPS)

```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### CUDA (Blackwell / any NVIDIA GPU)

`mlx-whisper` is Apple-only and will fail on Linux. Install PyTorch for CUDA first, then skip it:

```bash
# PyTorch with CUDA 12.8 (Blackwell requires 12.8+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install everything else — mlx-whisper will error; that's expected, ignore it
pip install -r requirements.txt --ignore-requires-python || true

# If mlx-whisper blocks the install entirely, install packages individually:
# pip install fastapi uvicorn yt-dlp faster-whisper sentence-transformers faiss-cpu \
#   pydantic[dotenv] python-multipart sqlite-utils opencv-python pillow \
#   transformers>=4.57 accelerate bitsandbytes spacy bertopic einops timm \
#   sentencepiece protobuf pyvips ultralytics nltk

python -m spacy download en_core_web_sm
```

On first ingest, Moondream2 loads in 4-bit NF4 (~1.5 GB VRAM). Whisper turbo runs on CUDA automatically.

## Running

```bash
cd backend
uvicorn app:app --port 8000
```

The server pre-warms X-CLIP and the sentence encoder on startup. Model weights are downloaded on first use; the initial ingest takes a few minutes.

Start the UI in a separate terminal:

```bash
cd frontend
npm install && npm run dev
```

## Ingestion

### YouTube / remote URL

```bash
curl -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"video_url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

### Local file

```bash
curl -X POST http://localhost:8000/ingest/upload \
  -F 'file=@/path/to/video.mp4'
```

Both return a `job_id`. Poll for completion:

```bash
curl http://localhost:8000/ingest/status/<job_id>
```

`status` is one of `queued`, `running`, `done`, `error`. The `stage` field gives a progress description. Only one ingest runs at a time; submitting while one is active returns HTTP 409.

Ingest runs as a subprocess so the caption model and YOLOv8 live in a separate process from the search server, avoiding shared-MPS memory pressure on Apple Silicon.

## Search

All searches go through `POST /query`.

### Request fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `mode` | `text` \| `visual` \| `action` \| `auto` | required | |
| `scope` | `video` \| `global` | `video` | |
| `query` | string | required | Natural-language query |
| `k` | int | 50 | Max results (capped at 200) |
| `video_url` | string | — | Required when `scope=video` unless `video_id` given |
| `video_id` | string | — | Extracted from URL when absent |
| `filter_objects` | string | — | Keep only hits whose YOLO labels contain this string |
| `videos` | string[] | — | Restrict `global` scope to these video IDs |

### Modes

**`text`** — semantic search over Whisper transcript chunks.

**`visual`** — frame-level cosine similarity using SigLIP embeddings. Best for static scenes, objects, and appearances.

**`action`** — clip-level similarity using X-CLIP embeddings over 4-second sliding windows. Best for activities and motion.

**`auto`** — runs all three plus direct YOLO object-label matching, then merges and deduplicates. Use this by default.

### Examples

```bash
# Transcript search
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"video_id":"dQw4w9WgXcQ","mode":"text","query":"the chorus","k":5}'

# Visual search across all indexed videos
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"scope":"global","mode":"visual","query":"chef with a knife","k":10}'

# Auto search on a single video
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"video_id":"zPxQjuFoUBc","mode":"auto","query":"cutting tool","k":10}'

# Action search filtered to frames containing a person
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"video_id":"zPxQjuFoUBc","mode":"action","query":"chopping","filter_objects":"person","k":20}'
```

### Response

```json
{
  "video_id": "zPxQjuFoUBc",
  "mode": "auto",
  "score_range": {"min": 0.28, "max": 0.51},
  "hits": [
    {
      "video_id": "zPxQjuFoUBc",
      "start": 9.3,
      "end": 11.3,
      "score": 0.512,
      "frame": "frames/zPxQjuFoUBc/frame-000010.jpg",
      "objects": ["cutter", "cutting", "tortilla", "wooden", "gloves"],
      "caption": "A freshly cooked tortilla rests on a wooden cutting board. A hand wearing black gloves holds a silver tortilla cutter near the tortilla.",
      "text": null
    }
  ]
}
```

`frame` paths are served by the API under `/frames/`.

## Other endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/videos` | List indexed videos with metadata and top object labels |
| DELETE | `/videos/{video_id}` | Remove a video and all its indexes |
| POST | `/build_contexts` | Rebuild per-video context vectors (after manual DB edits) |

## Data layout

```
backend/data/
  media/     <video_id>.mp4   <video_id>.wav
  frames/    <video_id>/frame-*.jpg
  indexes/   <video_id>.faiss           transcript (BGE)
             <video_id>.svfaiss         frames (SigLIP)
             <video_id>.xaclip.faiss    clips (X-CLIP)
             meta.sqlite                chunks, captions, objects, context vectors
  jobs/      <job_id>.json              ingest job status
```

To fully reset, stop the server and delete `backend/data/`. To remove a single video, use `DELETE /videos/{video_id}` or remove its three index files and run `DELETE /videos/{video_id}` to clean the DB rows.

## Design notes

**Prompt ensembling.** The search vector is the mean of embeddings for four phrasings of the query (`{}`, `a photo of {}`, `a video of {}`, `close-up of {}`). This improves recall for short queries without requiring query expansion heuristics.

**Evidence filter.** In `auto` mode, when YOLO finds no object match for a query in a given video, FAISS hits must have caption or object-label corroboration to be returned. Hits with no caption evidence require an X-CLIP score above 0.50 to pass. This prevents unrelated videos from appearing via embedding drift alone. Because the filter uses caption text, queries that phrase things differently from the caption model's vocabulary (e.g. "knife" when the caption says "tortilla cutter") may return zero hits — use descriptive phrases like "cutting tool" or "slicing" instead.

**Process isolation.** Each ingest job runs as a subprocess via `ingest_worker.py`. Moondream2 and YOLOv8 (ingest-time models) and X-CLIP + BGE (search-time models) never share an MPS process pool, which prevents OOM crashes on Apple Silicon.
