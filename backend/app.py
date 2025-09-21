from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from store import load_index, load_visual_index
from models import SearchResponse, SearchHit

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

@app.get("vsearch")
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