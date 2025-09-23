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

MODE = Literal["text", "visual", "action", "action_chain"]
class UnifiedSearchRequest(BaseModel):
    video_url: Optional[str] = None
    video_id: Optional[str] = None
    query: Optional[str] = None
    mode: MODE
    k: int = 10
    filter_objects: Optional[str] = None
    # Action chain params
    steps: Optional[List[str]] = None
    max_gap: float = 8.0
    ingest_if_needed: bool = True
    scope: str = "video"
    videos: Optional[List[str]] = None
    verify_with_gdino: bool = False
    verify_prompts: Optional[List[str]] = None
    verify_require_all: Optional[List[str]] = None
    verify_box_threshold: float = 0.3
    verify_text_threshold: float = 0.35
    verify_topk: int = 30

class UnifiedSearchHit(BaseModel):
    start: float
    end: float
    score: float
    text: Optional[str] = None                 # for text mode
    frame: Optional[str] = None                # for visual mode
    objects: Optional[List[str]] = None        # visual/action modes
    video_id: str

class UnifiedSearchResponse(BaseModel):
    video_id: Optional[str] = None
    mode: MODE
    hits: List[UnifiedSearchHit] = Field(default_factory=list)
    info: dict = Field(default_factory=dict)   # extra info (e.g., chosen path for action_chain)

class OVVerifyRequest(BaseModel):
    frames: List[str]
    prompts: List[str]
    box_threshold: float = 0.25
    text_threshold: float = 0.25