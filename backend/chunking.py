from typing import List, Dict

def chunk_segments(segments: List[Dict], max_sec=20, stride_sec=5):
    chunks = []
    i = 0
    while i < len(segments):
        t0 = segments[i]["start"]
        t1 = t0
        texts = []
        j = i
        while j < len(segments) and (segments[j]["end"] - t0) <= max_sec:
            texts.append(segments[j]["text"])
            t1 = segments[j]["end"]; j += 1
        if texts:
            chunks.append({"start": t0, "end": t1, "text": " ".join(texts)})
        
        # Move by stride
        # Find segment whose start >= t0 + stride_sec
        advance_to = t0 + stride_sec
        while i < len(segments) and segments[i]["start"] < advance_to:
            i += 1
    return chunks