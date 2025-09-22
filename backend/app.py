from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np, json
from sentence_transformers import SentenceTransformer
from store import load_index, get_conn, load_visual_index, load_action_clips_index
from models import SearchResponse, SearchHit, VideoIngestRequest, UnifiedSearchRequest, UnifiedSearchHit, UnifiedSearchResponse
from typing import List
from utils_unified import extract_video_id
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
CLIP_TXT = SentenceTransformer("clip-ViT-B-32")

def have_indexes(video_id: str, need_text=False, need_visual=False, need_action=False) -> bool:
    idx_dir = os.path.join(os.path.dirname(__file__), "data", "indexes")
    ok = True
    if need_text:
        ok &= os.path.exists(os.path.join(idx_dir, f"{video_id}.faiss"))
    if need_visual:
        ok &= os.path.exists(os.path.join(idx_dir, f"{video_id}.vfaiss"))
    if need_action:
        ok &= os.path.exists(os.path.join(idx_dir, f"{video_id}.aclip.faiss"))
    return ok

def ensure_ingested(video_url: str, video_id: str, need_text: bool, need_visual: bool, need_action: bool):
    from ingest import ingest as do_ingest
    from visual_ingest import ingest_visual as do_visual_ingest
    need_visual_or_action = need_visual or need_action
    idx_dir = os.path.join(os.path.dirname(__file__), "data", "indexes")
    if need_text and not os.path.exists(os.path.join(idx_dir, f"{video_id}.faiss")):
        do_ingest(video_url)
    if need_visual_or_action and not os.path.exists(os.path.join(idx_dir, f"{video_id}.vfaiss")):
        do_visual_ingest(video_url)

def expand_prompt(prompt: str) -> list[str]:
    return [
        prompt,
        f"a person {prompt}",
        f"someone {prompt}",
    ]

def encode_prompt_set(clip_txt_model, prompts: list[str]) -> np.ndarray:
    X = clip_txt_model.encode(prompts, normalize_embeddings=True).astype('float32')
    v = X.mean(axis=0)
    v /= (np.linalg.norm(v) + 1e-12)
    return v.reshape(1, -1)

def search_action_clips(video_id: str, q: str, k: int, filter_objects: str | None):
    index, rows = load_action_clips_index(video_id)
    qv = encode_prompt_set(CLIP_TXT, expand_prompt(q))
    D, I = index.search(qv, k)
    hits = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1: continue
        _, start, end, objects_json = rows[idx]
        objs = json.loads(objects_json) if objects_json else []
        if filter_objects and (filter_objects not in objs):
            continue
        hits.append({"start": float(start), "end": float(end), "score": float(score), "objects": objs})

    # Sort by time to help chaining
    hits.sort(key=lambda h: (h["start"], -h["score"]))
    return hits

def chain_actions(video_id: str, steps: list[str], k_per_step=40, max_gap=8.0, filter_objects: str | None = None):
    # Candidates per step
    cand = []
    for q in steps:
        cand.append(search_action_clips(video_id, q, k_per_step, filter_objects))

    # Simple DP: for each candidate in step t, find best predecessor in step t-1
    paths = []
    # Initialize with step 0 hits
    for h in cand[0]:
        paths.append([h])

    for t in range(1, len(steps)):
        new_paths = []
        for h in cand[t]:
            # Find best previous path that can connect
            best = None
            best_score = -1e9
            for p in paths:
                prev = p[-1]
                # Enforce order & gap constraint
                if h["start"] >= prev["end"] and (h["start"] - prev["end"] <= max_gap):
                    score = sum(x["score"] for x in p) + h["score"]
                    if score > best_score:
                        best = p
                        best_score = score
            if best:
                new_paths.append(best + [h])
        # Fallback: if no connections, allow restart to avoid empty
        if not new_paths and cand[t]:
            # Start anew with this step’s top few to keep going
            new_paths = [[h] for h in cand[t][:3]]
        paths = new_paths if new_paths else paths

    # Pick best path (max total score, must cover all steps if possible)
    def total_score(p): return sum(x["score"] for x in p)
    paths = sorted(paths, key=total_score, reverse=True)

    # Keep only paths that follow the step count if feasible
    full = [p for p in paths if len(p) == len(steps)]
    chosen = full[0] if full else (paths[0] if paths else [])
    return chosen, cand

@app.get("/search", response_model=SearchResponse)
async def search(video_id: str = Query(...), q: str = Query(...), k: int = 5):
    index, rows = load_index(video_id)
    qv = EMB.encode([q], normalize_embeddings=True).astype('float32')
    D, I = index.search(qv, k)
    hits = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        _, start, end, text = rows[idx]
        hits.append(SearchHit(start=start, end=end, text=text, score=score))
    return SearchResponse(video_id=video_id, hits=hits)

def fetch_clip_row(video_id: str, clip_idx: int):
    conn = get_conn()
    row = conn.execute("""
        SELECT idx, start, end, objects
        FROM visual_clips
        WHERE video_id=? AND idx=?
    """, (video_id, clip_idx)).fetchone()
    conn.close()
    return row

def search_action_global(q: str, k: int, filter_objects: str | None = None, restrict_videos: list[str] | None = None):
    """Search across all videos' action clips by searching individual video indexes."""
    try:
        all_hits = []
        
        # Get all videos that have action indexes
        index_dir = os.path.join(os.path.dirname(__file__), "data", "indexes")
        video_ids = []
        
        for filename in os.listdir(index_dir):
            if filename.endswith('.aclip.faiss'):
                video_id = filename.replace('.aclip.faiss', '')
                if not restrict_videos or video_id in restrict_videos:
                    video_ids.append(video_id)
        
        # Search each video individually
        for video_id in video_ids:
            try:
                hits = search_action_clips(video_id, q, k, filter_objects)
                
                # Add video_id to each hit and collect
                for hit in hits:
                    hit['video_id'] = video_id
                    all_hits.append(hit)
                    
            except Exception as e:
                print(f"Warning: Error searching video {video_id}: {e}")
                continue
        
        # Sort all hits by score (descending) and take top k
        all_hits.sort(key=lambda h: h["score"], reverse=True)
        return all_hits[:k]
        
    except Exception as e:
        print(f"Error in search_action_global: {e}")
        raise e

@app.get("/vsearch")
async def vsearch(video_id: str = Query(...), q: str = Query(...), k: int = 6, filter_objects: str | None = None):  
    try:
        index, rows = load_visual_index(video_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    qv = CLIP_TXT.encode([q], normalize_embeddings=True).astype('float32')
    D, I = index.search(qv, k)
    hits = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        _, start, end, frame, objects = rows[idx]

        # Object filter
        import json
        objs = json.loads(objects) if objects else []
        if filter_objects and filter_objects not in objs:
            continue
        hits.append({
            "start": float(start),
            "end": float(end),
            "frame": frame,
            "objects": objs,
            "score": float(score)
        })
    
    return {"video_id": video_id, "hits": hits}

@app.get("/asearch")
async def asearch(
    video_id: str = Query(...), 
    q: str = Query(...), 
    k: int = 40,
    filter_objects: str | None = None
):
    try:
        hits = search_action_clips(video_id, q, k, filter_objects)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"video_id": video_id, "hits": hits}

@app.get("/asearch_chain")
async def asearch_chain(
    video_id: str = Query(...),
    steps: List[str] = Query(..., description="Ordered list of action prompts"),
    k_per_step: int = 40,
    max_gap: float = 8.0,             # seconds allowed between steps
    filter_objects: str | None = None
):
    try:
        path, cand = chain_actions(video_id, steps, k_per_step=k_per_step, max_gap=max_gap, filter_objects=filter_objects)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Action clip index missing: {e}")

    return {
        "video_id": video_id,
        "steps": steps,
        "best_path": path,        # ordered segments picked
        "candidates_preview": [c[:5] for c in cand]  # top 5 per step for debugging
    }

@app.post("/ingest")
async def ingest_video(request: VideoIngestRequest):
    """Ingest a video from URL and create searchable indexes."""
    try:
        import os
        import re
        from urllib.parse import urlparse, parse_qs
        import subprocess
        
        video_url = request.video_url
        
        # Extract video ID from URL if not provided
        if request.video_id:
            video_id = request.video_id
        else:
            # Try to extract from YouTube URL
            if 'youtube.com' in video_url or 'youtu.be' in video_url:
                if 'youtu.be/' in video_url:
                    video_id = video_url.split('youtu.be/')[-1].split('?')[0]
                else:
                    parsed = urlparse(video_url)
                    video_id = parse_qs(parsed.query).get('v', [None])[0]
                
                if not video_id:
                    raise ValueError("Could not extract video ID from YouTube URL")
            else:
                # Generate ID from URL
                video_id = re.sub(r'[^a-zA-Z0-9_-]', '', video_url.split('/')[-1])[:11]
        
        print(f"Ingesting video: {video_id} from {video_url}")
        
        # Check if video already exists
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        media_path = os.path.join(data_dir, "media", f"{video_id}.mp4")
        
        if os.path.exists(media_path):
            return {
                "success": True, 
                "message": f"Video {video_id} already exists", 
                "video_id": video_id,
                "status": "already_exists"
            }
        
        # Run the ingestion process
        try:
            # Import the ingestion functions
            from ingest import ingest as do_ingest
            from visual_ingest import ingest_visual as do_visual_ingest
            
            # Download and process the video
            print(f"Starting ingestion for {video_id}...")
            
            # Text ingestion (ASR + embeddings)
            do_ingest(video_url)
            print(f"Text ingestion completed for {video_id}")
            
            # Visual ingestion (frames + action clips)
            do_visual_ingest(video_url)
            print(f"Visual ingestion completed for {video_id}")
            
            return {
                "success": True,
                "message": f"Video {video_id} ingested successfully",
                "video_id": video_id,
                "status": "completed"
            }
            
        except Exception as e:
            print(f"Ingestion error for {video_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
            
    except Exception as e:
        print(f"Error in video ingestion: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/videos")
async def list_videos():
    """List all ingested videos."""
    try:
        import os
        
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        media_dir = os.path.join(data_dir, "media")
        
        if not os.path.exists(media_dir):
            return {"videos": []}
        
        videos = []
        for filename in os.listdir(media_dir):
            if filename.endswith('.mp4'):
                video_id = filename.replace('.mp4', '')
                
                # Check what indexes exist
                indexes_dir = os.path.join(data_dir, "indexes")
                has_text = os.path.exists(os.path.join(indexes_dir, f"{video_id}.faiss"))
                has_visual = os.path.exists(os.path.join(indexes_dir, f"{video_id}.vfaiss"))
                has_actions = os.path.exists(os.path.join(indexes_dir, f"{video_id}.aclip.faiss"))
                
                videos.append({
                    "video_id": video_id,
                    "has_text_search": has_text,
                    "has_visual_search": has_visual,
                    "has_action_search": has_actions
                })
        
        return {"videos": sorted(videos, key=lambda x: x['video_id'])}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing videos: {e}")

@app.get("/asearch_all")
def asearch_all(
    q: str = Query(...),
    k: int = 50,
    filter_objects: str | None = None,
    videos: list[str] | None = Query(None, description="Optional list of video_ids to restrict to")
):
    try:
        hits = search_action_global(q, k, filter_objects=filter_objects, restrict_videos=videos)
        return {"query": q, "hits": hits[:k]}
    except Exception as e:
        print(f"Error in /asearch_all: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {e}")
    
@app.post("/query", response_model=UnifiedSearchResponse)
def unified_query(body: UnifiedSearchRequest):
    vid = extract_video_id(body.video_url)

    need_text   = body.mode == "text"
    need_visual = body.mode == "visual"
    need_action = body.mode in ("action", "action_chain")

    if body.ingest_if_needed and not have_indexes(vid, need_text, need_visual, need_action):
        ensure_ingested(body.video_url, vid, need_text, need_visual, need_action)

    if body.mode == "text":
        index, rows = load_index(vid)
        qv = EMB.encode([body.query or ""], normalize_embeddings=True).astype("float32")
        D, I = index.search(qv, body.k)
        hits = []
        for s, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx == -1: continue
            _, start, end, text = rows[idx]
            hits.append(UnifiedSearchHit(start=start, end=end, score=float(s), text=text, video_id=vid))
        return UnifiedSearchResponse(video_id=vid, mode="text", hits=hits)

    if body.mode == "visual":
        index, rows = load_visual_index(vid)
        qv = CLIP_TXT.encode([body.query or ""], normalize_embeddings=True).astype("float32")
        D, I = index.search(qv, body.k)
        hits = []
        import json as _json
        for s, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx == -1: continue
            _, start, end, frame, objects = rows[idx]
            objs = _json.loads(objects) if objects else []
            if body.filter_objects and body.filter_objects not in objs:
                continue
            hits.append(UnifiedSearchHit(start=float(start), end=float(end), score=float(s),
                                         frame=frame, objects=objs, video_id=vid))
        return UnifiedSearchResponse(video_id=vid, mode="visual", hits=hits)

    if body.mode == "action":
        hits = search_action_clips(vid, body.query or "", body.k, body.filter_objects)
        return UnifiedSearchResponse(
            video_id=vid, mode="action",
            hits=[UnifiedSearchHit(start=h["start"], end=h["end"], score=h["score"],
                                   objects=h.get("objects"), video_id=vid) for h in hits]
        )

    if body.mode == "action_chain":
        if not body.steps:
            raise HTTPException(400, "steps is required for mode=action_chain")
        path, cand = chain_actions(vid, body.steps, k_per_step=body.k,
                                   max_gap=body.max_gap, filter_objects=body.filter_objects)
        hits = [UnifiedSearchHit(start=h["start"], end=h["end"], score=h["score"],
                                 objects=h.get("objects"), video_id=vid) for h in path]
        return UnifiedSearchResponse(video_id=vid, mode="action_chain", hits=hits,
                                     info={"steps": body.steps, "preview_per_step": [c[:5] for c in cand]})

    raise HTTPException(400, f"Unknown mode {body.mode}")