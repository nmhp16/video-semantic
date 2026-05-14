from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class TranscriptSegment(BaseModel):
    text: str
    start: float
    end: float

class SearchHit(BaseModel):
    start: float
    end: float
    text: str
    score: float

class SearchResponse(BaseModel):
    video_id: str
    hits: List[SearchHit]

class VideoIngestRequest(BaseModel):
    video_url: str
    video_id: Optional[str] = None  # If not provided, will be extracted from URL

MODE = Literal["text", "visual", "action", "auto"]

# Keep search fan-out bounded so a single client can't request thousands
# of hits (every hit may trigger lazy Florence-2 captioning).
MAX_K = 200

class UnifiedSearchRequest(BaseModel):
    video_url: Optional[str] = None
    video_id: Optional[str] = None
    query: Optional[str] = None
    mode: MODE
    k: int = Field(default=50, ge=1, le=MAX_K)
    filter_objects: Optional[str] = None
    ingest_if_needed: bool = True
    scope: str = "video"
    videos: Optional[List[str]] = None

class UnifiedSearchHit(BaseModel):
    start: float
    end: float
    score: float
    text: Optional[str] = None                 # for text mode
    frame: Optional[str] = None                # for visual mode
    objects: Optional[List[str]] = None        # visual/action modes
    caption: Optional[str] = None
    video_id: str

class UnifiedSearchResponse(BaseModel):
    video_id: Optional[str] = None
    mode: MODE
    hits: List[UnifiedSearchHit] = Field(default_factory=list)
    info: dict = Field(default_factory=dict)
    score_range: Optional["ScoreRange"] = None

class ScoreRange(BaseModel):
    min: float
    max: float


class IngestJobResponse(BaseModel):
    job_id: Optional[str]
    video_id: str
    status: str   # "queued" | "already_exists"
    message: Optional[str] = None

class JobStatusResponse(BaseModel):
    job_id: str
    video_id: str
    status: str   # "queued" | "running" | "done" | "error"
    stage: str
    error: Optional[str] = None

# Resolve forward references now that all models are defined
UnifiedSearchResponse.model_rebuild()