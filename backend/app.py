# backend/app.py
import os, logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from db import init_db
from routers import search, ingest, videos

logging.basicConfig(level=logging.INFO)
app = FastAPI()

_cors_raw = os.environ.get("CORS_ORIGINS", "").strip()
if _cors_raw:
    _cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
else:
    _cors_origins = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_BASE = os.path.dirname(__file__)
_data_dir   = os.path.join(_BASE, "data")
_frames_dir = os.path.join(_data_dir, "frames")
_media_dir  = os.path.join(_data_dir, "media")
os.makedirs(_frames_dir, exist_ok=True)
os.makedirs(_media_dir, exist_ok=True)
app.mount("/frames", StaticFiles(directory=_frames_dir), name="frames")
app.mount("/media",  StaticFiles(directory=_media_dir),  name="media")

app.include_router(search.router)
app.include_router(ingest.router)
app.include_router(videos.router)


@app.on_event("startup")
async def startup():
    init_db()
