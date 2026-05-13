# Video Semantic — Backend & Frontend Improvements Design

**Date:** 2026-05-12  
**Scope:** Architecture (code quality), new user-facing features, performance & reliability  
**Approach:** C — focused restructure + high-impact features

---

## 1. Goals

1. **Code quality** — split monolithic `app.py` (37 KB) and `store.py` (24 KB) into focused modules without changing any external API contract.
2. **New features** — async ingestion with stage-level progress polling; video titles surfaced on cards and library; search history; object filter suggestions; better score display; richer library page.
3. **Performance & reliability** — FAISS in-memory LRU cache; SQLite schema init runs once; missing column indexes added; EMB model loaded lazily.

---

## 2. Backend Architecture

### 2.1 File layout after refactor

```
backend/
  app.py              # slim: startup, middleware, static mounts, router includes
  db.py               # NEW: init_db() (runs once), db() context manager, clear_video()
  indexes.py          # NEW: FAISS save/load with LRU in-memory cache (max 50 entries)
  context.py          # NEW: build_video_context, filter_videos_by_context, topic derivation
  routers/
    search.py         # NEW: /query (unified), legacy /search /vsearch /asearch /asearch_chain
    ingest.py         # NEW: /ingest (async), /ingest/status/{job_id}, /build_contexts
    videos.py         # NEW: GET /videos, DELETE /videos/{id}, POST /ov_verify
  store.py            # kept as thin re-exports during transition
```

No external API routes are added, removed, or renamed. Legacy endpoints (`/search`, `/vsearch`, `/asearch`, `/asearch_chain`) move to `routers/search.py` and receive a deprecation note in their docstrings — they still return the same responses.

### 2.2 `db.py`

- Exports `init_db()` — creates all tables and runs `ALTER TABLE` migrations exactly once, called from `app.py` on startup via `@app.on_event("startup")`.
- Exports `db()` context manager (same signature as today's `store.db()`).
- Exports `clear_video(video_id)`.
- Adds SQLite column indexes on first-run:
  - `chunks(video_id)`
  - `visual_chunks(video_id)`
  - `visual_clips(video_id)`
  - `caption_cache(video_id)`
- Removes the redundant `idx_video_context_topics` index (video_id is already the PRIMARY KEY on `video_context`).
- WAL mode and `synchronous=NORMAL` PRAGMA set once in `init_db()` and also in `db()` for safety.

### 2.3 `indexes.py`

- Wraps every `faiss.read_index()` behind a per-video LRU cache keyed by `(video_id, index_type)`, max 50 entries total.
- Cache is explicitly invalidated for a video when `clear_video()` runs.
- After ingest writes a new FAISS file, the relevant cache entry is warmed immediately.
- `SentenceTransformer("BAAI/bge-small-en-v1.5")` is loaded lazily on first use (instead of at module import in `store.py`).

### 2.4 `context.py`

- Contains `build_video_context`, `filter_videos_by_context`, `passes_hard_context`, `derive_topics`, `derive_topics_bertopic`, `top_objects_for`, `top_actions_for`, `derive_text_summary_for`.
- No logic changes — pure relocation from `store.py`.

### 2.5 `routers/ingest.py` — async ingestion

**Job state:**

```python
@dataclass
class JobState:
    job_id: str
    video_id: str
    status: Literal["queued", "running", "done", "error"]
    stage: str           # human label shown in frontend
    error: str | None
    created_at: float    # time.monotonic()
```

An in-process `_jobs: dict[str, JobState]` dict is the job store. Expiry is check-on-access: `GET /ingest/status/{job_id}` returns 404 if `time.monotonic() - job.created_at > 600` and the job is in a terminal state (`done` or `error`).

**Endpoints:**

`POST /ingest`
- Validates URL, extracts `video_id`.
- Checks if already ingested (`media_path` exists) → returns `{status: "already_exists", video_id, job_id: None}` immediately.
- Creates `JobState(status="queued")`, stores in `_jobs`.
- Submits `run_ingest(job_id, url, video_id)` to a module-level `ThreadPoolExecutor(max_workers=2)`.
- Returns `{job_id, video_id, status: "queued"}` immediately (HTTP 202).

`GET /ingest/status/{job_id}`
- Returns current `JobState` as JSON.
- HTTP 404 if unknown or expired.

`run_ingest(job_id, url, video_id)` (background thread):

```
stage("Downloading & extracting frames…")
  → visual_ingest(url)           # downloads video, samples frames, SigLIP index
stage("Transcribing audio…")
  → ingest(url)                  # Whisper ASR, text FAISS
stage("Building context…")
  → build_video_context(video_id)
→ JobState(status="done")
```

On any exception: `JobState(status="error", error=str(e))`.

**Video title capture:**  
`yt-dlp` already produces an `.info.json` during download (via `--write-info-json` or the Python API's `extract_info`). A `_extract_title(url)` helper in `routers/ingest.py` calls `yt-dlp` with `extract_info(url, download=False)` (Python API, no separate subprocess) to get the `title` and `webpage_url` fields. This is called before `visual_ingest` and stored into `video_context.title` and `video_context.source_url` via a `store_video_meta(video_id, title, source_url)` helper in `db.py`. On failure (non-YouTube URL or network error) the fields are left `None`.

### 2.6 `routers/videos.py`

`GET /videos` response gains two new optional fields per video:

```json
{
  "video_id": "dQw4w9WgXcQ",
  "title": "Rick Astley - Never Gonna Give You Up",
  "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "has_text_search": true,
  "has_visual_search": true,
  "has_action_search": true,
  "thumbnail_url": "/frames/dQw4w9WgXcQ/frame-000001.jpg",
  "top_objects": ["person", "chair", "microphone", "guitar", "keyboard"]
}
```

`title`, `source_url`, and `top_objects` are `null` / `[]` for videos ingested before this change.

### 2.7 `routers/search.py`

`POST /query` response gains one new optional field:

```json
{
  "video_id": "...",
  "mode": "visual",
  "hits": [...],
  "score_range": {"min": 0.21, "max": 0.38}
}
```

`score_range` is computed from the `score` field of the returned hits (after dedup/NMS, before fusion), letting the frontend normalize scores relative to the actual result set. All existing fields unchanged.

### 2.8 Error handling improvements

- `/query` returns HTTP 404 with `{"detail": "No index for video <id> — ingest first"}` instead of a raw FAISS file-not-found exception.
- `clear_video()` returns a structured result `{deleted: [...], failed: [...]}` so callers know what was and wasn't removed.

---

## 3. Frontend Features

### 3.1 Ingest progress

`IngestModal` flow:

1. POST `/ingest` → receive `{job_id, video_id, status}`.
2. If `status === "already_exists"` → show "Already indexed" immediately (existing behavior).
3. Otherwise start polling `GET /ingest/status/{job_id}` every 2 seconds.
4. Display current `stage` string with a spinner. Show a stage checklist:
   - Downloading & extracting frames
   - Transcribing audio
   - Building context
   Each stage gets a checkmark when the stage string advances past it.
5. On `status === "done"` → stop polling, show success + video title.
6. On `status === "error"` → stop polling, show error string.

### 3.2 Video titles

- `VideoMeta` interface gains `title: string | null` and `source_url: string | null`.
- `ResultCard`: receives a new optional `title?: string` prop (passed from `SearchPage` via a `videoTitles: Record<string, string>` map built from the `videos` list). Primary label becomes `title ?? hit.video_id`. YouTube external-link button remains. Video ID shown in smaller monospace below title.
- `LibraryPage`: card header shows title (or video_id fallback). Source URL shown as a small external link.

### 3.3 Object filter suggestions

`FilterPanel` replaces the plain text `<input>` for object filter with an `<input list="objects-datalist">` backed by `<datalist>`. The datalist options are populated from `video_context.objects_topk` for the selected video — fetched as part of `GET /videos` (the backend adds `objects_topk: string[]` to the per-video response, top 10 object labels). When scope is global, the datalist merges objects across all videos.

To support this, `GET /videos` adds `top_objects: string[]` per video (top 10 from `objects_topk` in `video_context`).

### 3.4 Search history

- `useSearchHistory` hook: reads/writes to `localStorage` key `vsearch:history`. Stores last 10 entries of `{query, mode, timestamp}`. Deduplicates on `(query, mode)`.
- Rendered as a compact row of clickable chips below the search input, visible only when the input is focused and empty.
- Clicking a chip sets `query` and `mode` state and immediately runs the search.
- An "×" icon on each chip removes it from history.

### 3.5 Score normalization

`ResultCard` computes display percentage as:

```
pct = score_range.max === score_range.min
  ? Math.round(hit.score * 100)
  : Math.round(((hit.score - score_range.min) / (score_range.max - score_range.min)) * 100)
```

`score_range` is passed down from `SearchPage` (received from the `/query` response). This makes "100%" mean the best result in this set and "0%" the weakest, regardless of raw cosine similarity scale.

### 3.6 Library page improvements

Each video card shows:
- Title (or video_id fallback) as the header
- Source URL as a small external link
- Index completeness: three colored dots labeled Text / Visual / Action (green = indexed, gray = not yet)
- "Re-index context" button → calls `POST /build_contexts` with `[video_id]`; shows a brief spinner

---

## 4. Data Flow Summary

```
User clicks "Add video"
  → IngestModal: POST /ingest → job_id (202)
  → poll GET /ingest/status/{job_id} every 2s
    → stage: "Downloading…" → "Transcribing…" → "Building context…"
  → status: done → title displayed, library refreshed

User types query + hits Search
  → POST /query → {hits, score_range}
  → ResultCard shows title (not video_id), normalized score %
  → query + mode saved to localStorage history

User focuses empty search box
  → history chips appear → click to restore query + re-search

User opens Library page
  → GET /videos → title, source_url, top_objects per video
  → object filter datalist populated with top_objects
```

---

## 5. What Is Not Changed

- Unified `/query` scoring and reranking logic
- SigLIP / Florence-2 / OWLv2 model code
- `visual_ingest.py`, `ingest.py`, `gdino.py`, `chunking.py`
- FAISS index formats and SQLite schema (additive only: new indexes, no column drops)
- CORS, static file mounts
- All existing API response shapes (new fields are additive)

---

## 6. Testing Approach

- Manually ingest one YouTube video end-to-end after the refactor; verify all four search modes return results identical to pre-refactor.
- Verify progress polling: open ingest modal, confirm stage labels advance and "done" state reached.
- Verify title appears on result cards and library after ingest.
- Verify search history persists across page reload and deduplicates.
- Verify score normalization: best result shows ~100%, weakest ~0%.
- Verify FAISS cache: second search on same video is noticeably faster.
