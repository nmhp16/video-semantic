# Install dependencies
cd backend && pip install -r requirements.txt

# Ingest audio/transcript
python ingest.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Then ingest visuals (frames + CLIP + YOLO)
python visual_ingest.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Start API
uvicorn app:app --reload --port 8000

# Query visual search
curl "http://localhost:8000/vsearch?video_id=dQw4w9WgXcQ&q=singing&filter_objects=person"

# Check meta.sqlite for saved frames
