# 1. Make sure you already ingested audio/transcript
python ingest.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# 2. Then ingest visuals (frames + CLIP + YOLO)
python - <<'PY'
from visual_ingest import ingest_visual
ingest_visual("dQw4w9WgXcQ", "data/media/dQw4w9WgXcQ.wav", every_sec=1.0)
PY

# 3) Start API
uvicorn app:app --reload --port 8000

# 4) Query visual search
curl "http://localhost:8000/vsearch?video_id=dQw4w9WgXcQ&q=singing&filter_objects=person"