from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np, json
from sentence_transformers import SentenceTransformer
import faiss
from store import load_index, load_global_action_index, unpack_gid, video_id_from_crc, get_conn, load_visual_index, load_action_clips_index
from models import SearchResponse, SearchHit
from typing import List

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
    # Encode prompt
    qv = encode_prompt_set(CLIP_TXT, expand_prompt(q)).astype("float32")

    # Search global FAISS
    index = load_global_action_index()
    D, I = index.search(qv, k)  # I are gids
    hits = []
    for score, gid in zip(D[0].tolist(), I[0].tolist()):
        if gid == -1: continue
        crc, clip_idx = unpack_gid(int(gid))
        video_id = video_id_from_crc(crc)
        if not video_id: continue
        if restrict_videos and video_id not in restrict_videos:
            continue
        row = fetch_clip_row(video_id, int(clip_idx))
        if not row: continue
        _, start, end, objects_json = row
        import json
        objs = json.loads(objects_json) if objects_json else []
        if filter_objects and (filter_objects not in objs):
            continue
        hits.append({
            "video_id": video_id,
            "start": float(start),
            "end": float(end),
            "objects": objs,
            "score": float(score),
        })
    # sort by score desc
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits

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

@app.get("/asearch_all")
def asearch_all(
    q: str = Query(...),
    k: int = 50,
    filter_objects: str | None = None,
    videos: list[str] | None = Query(None, description="Optional list of video_ids to restrict to")
):
    try:
        hits = search_action_global(q, k, filter_objects=filter_objects, restrict_videos=videos)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Global index missing or query error: {e}")
    return {"query": q, "hits": hits[:k]}