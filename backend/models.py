from pydantic import BaseModel, Field
from typing import List, Optional, Literal

MODE = Literal["text", "visual", "action"]

class VideoIngestRequest(BaseModel):
    video_url: str
    video_id: Optional[str] = None

class UnifiedSearchRequest(BaseModel):
    video_url:        Optional[str]       = None
    video_id:         Optional[str]       = None
    query:            Optional[str]       = None
    mode:             MODE
    k:                int                 = 50
    scope:            str                 = "video"
    videos:           Optional[List[str]] = None   # restrict global search to these video_ids
    ingest_if_needed: bool                = True

class UnifiedSearchHit(BaseModel):
    video_id: str
    start:    float
    end:      float
    score:    float
    text:     Optional[str]       = None  # text mode: what was said
    frame:    Optional[str]       = None  # visual mode: thumbnail path
    objects:  Optional[List[str]] = None  # visual/action: detected objects
    caption:  Optional[str]       = None  # visual/action: frame description

class UnifiedSearchResponse(BaseModel):
    video_id: Optional[str]        = None
    mode:     MODE
    hits:     List[UnifiedSearchHit] = Field(default_factory=list)
    info:     dict                   = Field(default_factory=dict)
