from pydantic import BaseModel
from typing import List, Optional

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
