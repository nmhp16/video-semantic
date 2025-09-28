# Video Semantic Search System

A powerful multi-modal video search system that allows you to search through videos using text, visual content, and actions. This system can process YouTube videos or local video files and make them searchable using natural language queries.

## 🚀 Quick Start

### 1. Installation
```bash
cd backend && pip install -r requirements.txt
```

### 2. Start the API Server
```bash
uvicorn app:app --reload --port 8000
```

### 3. Ingest a Video
```bash
curl -X POST "http://127.0.0.1:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"video_url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

### 4. Search the Video
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

## 📁 Project Structure

```
video-semantic/
├── backend/
│   ├── app.py              # Main FastAPI application
│   ├── models.py           # Pydantic data models
│   ├── ingest.py           # Text/audio ingestion pipeline
│   ├── visual_ingest.py    # Visual content ingestion
│   ├── gdino.py           # GroundingDINO object detection
│   ├── gdino_helper.py    # GroundingDINO utilities
│   ├── store.py           # Database and index management
│   ├── chunking.py        # Text chunking utilities
│   ├── utils_unified.py   # General utilities
│   └── requirements.txt   # Python dependencies
└── README.md
```

## 🔧 File-by-File Documentation

### `app.py` - Main API Server
The core FastAPI application with all search endpoints.

**Key Functions:**
- `search()` - Basic text search endpoint
- `vsearch()` - Visual search endpoint  
- `asearch()` - Action search endpoint
- `asearch_chain()` - Action chain search endpoint
- `unified_query()` - Main unified search endpoint supporting all modes
- `ingest_video()` - Video ingestion endpoint
- `list_videos()` - List all processed videos

**Search Modes:**
- `text` - Search through video transcripts
- `visual` - Search for objects/scenes in video frames
- `action` - Search for actions in video segments
- `action_chain` - Search for sequences of related actions

### `models.py` - Data Models
Pydantic models for API request/response validation.

**Key Models:**
- `UnifiedSearchRequest` - Main search request model
- `UnifiedSearchResponse` - Search results model
- `VideoIngestRequest` - Video ingestion request
- `SearchHit` - Individual search result
- `TranscriptSegment` - Audio transcript segment

### `ingest.py` - Text/Audio Processing
Handles video download, audio extraction, transcription, and text indexing.

**Key Functions:**
- `ingest(url_or_path)` - Main ingestion function
- `ytdlp(url, out)` - Download video from YouTube
- `extract_audio(infile, outfile)` - Extract audio using FFmpeg
- `transcribe(wav_path)` - Transcribe audio using Whisper
- `embed_texts(texts)` - Generate text embeddings
- `chunk_segments(segments)` - Split transcripts into searchable chunks

**Process:**
1. Download video or use local file
2. Extract audio track
3. Transcribe audio using Whisper
4. Chunk transcript into overlapping segments
5. Generate embeddings using SentenceTransformer
6. Store in FAISS index and SQLite database

### `visual_ingest.py` - Visual Content Processing
Handles frame extraction, visual embeddings, and object detection.

**Key Functions:**
- `ingest_visual(url_or_path)` - Main visual ingestion
- `sample_frames(video_path, out_dir)` - Extract frames at regular intervals
- `ClipEncoder` - CLIP model for visual embeddings
- `YoloDetector` - YOLO model for object detection
- `build_clip_windows()` - Create action clips from frames

**Process:**
1. Download video or use local file
2. Extract frames every second (configurable)
3. Generate CLIP embeddings for each frame
4. Detect objects using YOLO
5. Create action clips by averaging frame embeddings
6. Store visual indexes and metadata

### `gdino.py` - Advanced Object Detection
GroundingDINO integration for precise object detection with text prompts.

**Key Functions:**
- `detect_on_image(image_path, prompts)` - Detect objects in single image
- `_caption(prompts)` - Format prompts for GroundingDINO

**Features:**
- Text-prompted object detection
- Configurable confidence thresholds
- Returns bounding boxes and labels

### `gdino_helper.py` - GroundingDINO Utilities
Helper functions for integrating GroundingDINO with search results.

**Key Functions:**
- `rerank_with_gdino()` - Re-rank search results using GroundingDINO verification

### `store.py` - Data Storage Management
Handles FAISS indexes and SQLite database operations.

**Key Functions:**
- `save_index()` - Save text embeddings to FAISS
- `load_index()` - Load text search index
- `save_visual_index()` - Save visual embeddings
- `load_visual_index()` - Load visual search index
- `save_action_clips_index()` - Save action clip embeddings
- `load_action_clips_index()` - Load action search index
- `get_conn()` - Get SQLite database connection
- `clear_video()` - Remove all data for a video

**Database Tables:**
- `chunks` - Text transcript chunks
- `visual_chunks` - Individual video frames
- `visual_clips` - Action segments

### `chunking.py` - Text Processing
Utility for splitting transcripts into searchable chunks.

**Key Functions:**
- `chunk_segments(segments, max_sec, stride_sec)` - Split transcript into overlapping chunks

### `utils_unified.py` - General Utilities
Common utility functions.

**Key Functions:**
- `extract_video_id(video_url)` - Extract YouTube video ID from URL

## 🔍 Search Modes Explained

### 1. Text Search (`mode: "text"`)
Searches through video transcripts using semantic similarity.

**Example:**
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "video_url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "mode":"text",
    "query":"never gonna give you up",
    "k":5
  }'
```

### 2. Visual Search (`mode: "visual"`)
Searches for objects, scenes, or visual content in video frames.

**Example:**
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

**Parameters:**
- `filter_objects` - Only show results containing this object
- `verify_with_gdino` - Use GroundingDINO to verify results
- `verify_prompts` - Objects to look for in verification
- `verify_require_all` - Require all these objects to be present

### 3. Action Search (`mode: "action"`)
Searches for specific actions or activities in video segments.

**Example:**
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "video_url":"https://youtu.be/zPxQjuFoUBc",
    "mode":"action",
    "query":"chopping meat",
    "filter_objects":"person",
    "k":40
  }'
```

### 4. Action Chain Search (`mode: "action_chain"`)
Finds sequences of related actions in order.

**Example:**
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

**Parameters:**
- `steps` - Ordered list of actions to find
- `max_gap` - Maximum time gap between consecutive actions

## 🌐 Search Scopes

### Video Scope (`scope: "video"`)
Search within a specific video.

### Global Scope (`scope: "global"`)
Search across all processed videos.

**Example:**
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

## 🧪 Testing the System

### 1. Test Video Ingestion
```bash
# Ingest a short YouTube video
curl -X POST "http://127.0.0.1:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"video_url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'

# Check if video was processed
curl "http://127.0.0.1:8000/videos"
```

### 2. Test Text Search
```bash
# Search for specific words or phrases
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "video_url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "mode":"text",
    "query":"never gonna",
    "k":3
  }'
```

### 3. Test Visual Search
```bash
# Search for visual content (requires visual ingestion)
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "video_url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "mode":"visual",
    "query":"person singing",
    "k":5
  }'
```

### 4. Test Action Search
```bash
# Search for actions (requires visual ingestion)
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "video_url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "mode":"action",
    "query":"dancing",
    "k":10
  }'
```

### 5. Test GroundingDINO Verification
```bash
# Use advanced object detection to verify results
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "video_url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "mode":"visual",
    "query":"person with microphone",
    "verify_with_gdino": true,
    "verify_prompts": ["person", "microphone"],
    "verify_require_all": ["person"],
    "k":5
  }'
```

## 📊 Data Storage

The system stores data in the `backend/data/` directory:

```
data/
├── media/           # Downloaded video files
├── frames/          # Extracted video frames
├── transcripts/     # Audio transcripts (JSON)
└── indexes/         # FAISS search indexes
    ├── *.faiss      # Text search indexes
    ├── *.vfaiss     # Visual search indexes
    ├── *.aclip.faiss # Action search indexes
    └── meta.sqlite  # Metadata database
```

## 🔧 Configuration

### Environment Variables
- `GDINO_MODEL_ID` - GroundingDINO model to use (default: "IDEA-Research/grounding-dino-tiny")

### Model Parameters
- **Whisper Model**: "small" (can be changed in `ingest.py`)
- **CLIP Model**: "clip-ViT-B-32" (can be changed in `visual_ingest.py`)
- **YOLO Model**: "yolov8n.pt" (can be changed in `visual_ingest.py`)
- **Frame Sampling**: 1 second intervals (configurable in `visual_ingest.py`)

## 🚨 Troubleshooting

### Common Issues

1. **Video not found**: Ensure the video URL is valid and accessible
2. **No visual results**: Make sure visual ingestion completed successfully
3. **GroundingDINO errors**: Check if the model downloaded correctly
4. **Memory issues**: Reduce batch sizes or use smaller models

### Debugging

1. Check the API logs for error messages
2. Verify video files exist in `data/media/`
3. Check if indexes were created in `data/indexes/`
4. Use the `/videos` endpoint to see processing status

## 📈 Performance Tips

1. **For large videos**: Increase frame sampling interval
2. **For better accuracy**: Use GroundingDINO verification
3. **For speed**: Reduce the number of results (`k` parameter)
4. **For memory**: Use smaller models or process videos in batches

## 🔮 Advanced Usage

### Custom Video Processing
```python
# Process local video file
from ingest import ingest
from visual_ingest import ingest_visual

# Text processing only
ingest("/path/to/video.mp4")

# Visual processing only  
ingest_visual("/path/to/video.mp4")
```

### Direct Database Access
```python
from store import get_conn

conn = get_conn()
# Query transcripts
rows = conn.execute("SELECT * FROM chunks WHERE video_id=?", ("video_id",)).fetchall()
conn.close()
```

This system provides a complete solution for making videos searchable using natural language, with support for text, visual, and action-based queries across multiple search scopes.