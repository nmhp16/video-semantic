# Install dependencies
```
cd backend && pip install -r requirements.txt
```

# Start API
```
uvicorn app:app --reload --port 8000
```

# Ingest audio/transcript and visuals (frames + CLIP + YOLO)
```
curl -X POST "http://127.0.0.1:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"video_url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

# Text search
```
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "video_url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "mode":"text",
    "query":"chorus",
    "k":5
  }'
```

# Visual object search
```
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "video_url":"https://youtu.be/zPxQjuFoUBc",
    "scope":"video",
    "mode":"visual",
    "query":"chef chopping meat on a board",
    "filter_objects":"person",
    "verify_with_gdino": true,
    "verify_prompts": ["chef","knife","cutting board"],
    "verify_require_all": ["knife","cutting board"],
    "k":12
  }'
```

# Visual action search
```
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "video_url":"https://youtu.be/zPxQjuFoUBc",
    "scope":"video",
    "mode":"action",
    "query":"chopping meat",
    "filter_objects":"person",
    "verify_with_gdino": true,
    "verify_prompts": ["chef","knife","cutting board"],
    "verify_require_all": ["knife","cutting board"],
    "k":40
  }'
```

# Action chain search
```
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "video_url":"https://youtu.be/zPxQjuFoUBc",
    "scope":"video",
    "mode":"action_chain",
    "steps":["open fridge","take meat","chop meat"],
    "k":50,
    "max_gap":8.0,
    "filter_objects":"person",
    "verify_with_gdino": true,
    "verify_prompts": ["chef","knife","cutting board"],
    "verify_require_all": ["knife","cutting board"],
  }'
```

# Global action search
**Change mode like above**

```
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "scope":"global",
    "mode":"action",
    "query":"chopping meat",
    "filter_objects":"person",
    "verify_with_gdino": true,
    "verify_prompts": ["chef","knife","cutting board"],
    "verify_require_all": ["knife","cutting board"],
    "k":40
  }'
```

# Check meta.sqlite for saved frames
