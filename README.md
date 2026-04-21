# Video Semantic Search

A multi-modal video search service. Given a YouTube URL or local video file, the system produces an ASR transcript, per-frame captions, and frame/segment embeddings, then serves text, visual, and action search through a single FastAPI endpoint.

## Architecture

| Component | Model |
|-----------|-------|
| Speech-to-text | OpenAI Whisper |
| Text / caption embeddings | `BAAI/bge-small-en-v1.5` (SentenceTransformers) |
| Frame captioning and object labels | `microsoft/Florence-2-base` |
| On-demand open-vocabulary detection | `google/owlv2-base-patch16-ensemble` |
| Vector index | FAISS (inner product on L2-normalized vectors) |
| Metadata store | SQLite |

Captions and transcripts share the same embedding space, so text queries and visual queries retrieve against consistent vectors.

## Project Structure

```
video-semantic/
  backend/
    app.py            FastAPI application and search endpoints
    models.py         Pydantic request/response models
    ingest.py         Download, audio extraction, Whisper transcription, text index
    visual_ingest.py  Frame sampling, Florence-2 captioning, caption embedding,
                      action-clip windows
    gdino.py          OWLv2 open-vocabulary detector used by /ov_verify
    store.py          FAISS and SQLite persistence, context filters
    chunking.py       Transcript chunking utilities
    utils_unified.py  URL / video-id helpers
    requirements.txt
    data/             Runtime artifacts (media, frames, indexes, sqlite)
  frontend/           Vite + React UI (work in progress)
  extension/          Browser extension (placeholder)
  README.md
```

## Requirements

- Python 3.10 or newer
- `ffmpeg` and `ffprobe` on PATH
- `yt-dlp` (installed by `requirements.txt`)
- Optional GPU: CUDA for Florence-2 and OWLv2 will be used automatically when available

## Installation

```bash
cd backend
pip install -r requirements.txt
```

## Running the API

```bash
cd backend
uvicorn app:app --reload --port 8000
```

Wait for the log line `Application startup complete` before issuing requests. Model weights are downloaded on first use, so the initial ingest can take several minutes.

## Ingestion

Download, transcribe, sample frames, caption, and build all indexes in one call:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"video_url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

The pipeline:

1. `yt-dlp` downloads the video to `data/media/<video_id>.mp4`.
2. `ffmpeg` extracts audio to `data/media/<video_id>.wav`.
3. Whisper produces a transcript; chunks are embedded and written to `data/indexes/<video_id>.faiss`.
4. `ffmpeg` samples one frame per second into `data/frames/<video_id>/`.
5. Florence-2 produces a detailed caption and object labels for each frame.
6. Captions are embedded with `bge-small` and written to `<video_id>.vfaiss`.
7. Sliding windows over caption embeddings produce action clips in `<video_id>.aclip.faiss`.
8. A summary context vector is built for cross-video filtering.

List ingested videos and their available indexes:

```bash
curl http://127.0.0.1:8000/videos
```

## Unified Query Endpoint

All searches go through `POST /query` with a mode and scope.

### Modes

- `text` — semantic search over transcript chunks.
- `visual` — frame-level search over caption embeddings.
- `action` — segment-level search over sliding windows of captions.
- `action_chain` — ordered sequence of action queries with a maximum inter-step gap.

### Scopes

- `video` (default) — restrict to one video via `video_id` or `video_url`.
- `global` — search across all ingested videos; optionally restrict with `videos: [...]`.

### Text search

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "video_url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "mode":"text",
    "query":"chorus",
    "k":5
  }'
```

### Visual search

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "video_url":"https://youtu.be/zPxQjuFoUBc",
    "mode":"visual",
    "query":"chef chopping meat on a board",
    "filter_objects":"person",
    "verify_with_gdino": true,
    "verify_prompts": ["chef","knife","cutting board"],
    "k":12
  }'
```

When `verify_with_gdino` is true, hits are reranked by checking whether Florence-2 captions contain the `verify_prompts` terms. No extra detector inference is required at query time; the reranker reuses the captions stored at ingest time.

### Action search

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "video_url":"https://youtu.be/zPxQjuFoUBc",
    "mode":"action",
    "query":"chopping meat",
    "k":40
  }'
```

### Action-chain search

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "video_url":"https://youtu.be/zPxQjuFoUBc",
    "mode":"action_chain",
    "steps":["open fridge","take meat","chop meat"],
    "max_gap":8.0,
    "k":50
  }'
```

Parameters:
- `steps` — ordered list of action prompts.
- `max_gap` — maximum seconds between the end of one matched clip and the start of the next.

### Global scope

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "scope":"global",
    "mode":"action",
    "query":"cooking",
    "k":20
  }'
```

## Auxiliary Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/search`           | Legacy text search for a single video. |
| GET  | `/vsearch`          | Legacy frame-level visual search. |
| GET  | `/asearch`          | Legacy action-clip search for a single video. |
| GET  | `/asearch_chain`    | Legacy action-chain search. |
| GET  | `/asearch_all`      | Action search across all indexed videos. |
| GET  | `/videos`           | List ingested videos and available indexes. |
| POST | `/ingest`           | Run the ingestion pipeline for a video. |
| POST | `/build_contexts`   | Rebuild per-video context vectors used by the global scope. |
| POST | `/ov_verify`        | Run OWLv2 detection on specified frames for ad-hoc verification. |

Static file mounts:
- `/frames/<video_id>/...` serves sampled frames.
- `/media/<video_id>.mp4` serves source media.

## Request Models

The primary request body, `UnifiedSearchRequest`:

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `video_url` | string | — | Required for scope `video` unless `video_id` given. |
| `video_id` | string | — | Derived from URL when absent. |
| `query` | string | — | Required for `text`, `visual`, `action`. |
| `mode` | `text` \| `visual` \| `action` \| `action_chain` | — | |
| `k` | int | 50 | Top-k after deduping and NMS. |
| `filter_objects` | string | null | Keep hits whose object labels contain this string. |
| `steps` | string[] | null | Required for `action_chain`. |
| `max_gap` | float | 8.0 | Inter-step gap in seconds for `action_chain`. |
| `ingest_if_needed` | bool | true | Auto-ingest when indexes are missing. |
| `scope` | `video` \| `global` | `video` | |
| `videos` | string[] | null | Restrict global scope to these video_ids. |
| `verify_with_gdino` | bool | false | Enable caption-based verification rerank. |
| `verify_prompts` | string[] | null | Terms sought in captions. |
| `verify_require_all` | string[] | null | Hard-required terms for the rerank. |

## Data Layout

```
backend/data/
  media/      <video_id>.mp4, <video_id>.wav
  frames/     <video_id>/frame-XXXXXX.jpg
  indexes/    <video_id>.faiss         text
              <video_id>.vfaiss        visual (caption embeddings)
              <video_id>.aclip.faiss   action clips
              meta.sqlite              rows, captions, object labels, context
  transcripts/<video_id>.json
```

To reset the service, stop the server and remove everything under `backend/data/` except the directory itself.
