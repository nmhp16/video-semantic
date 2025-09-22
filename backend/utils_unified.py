import re
from urllib.parse import urlparse, parse_qs

YT_ID_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")

def extract_video_id(video_url: str) -> str:
    m = YT_ID_RE.search(video_url)
    if m:
        return m.group(1)
    if "youtu.be/" in video_url:
        return video_url.split("youtu.be/")[-1].split("?")[0]
    if "youtube.com" in video_url:
        parsed = urlparse(video_url)
        v = parse_qs(parsed.query).get("v", [None])[0]
        if v: return v
    # fallback: sanitize tail
    tail = urlparse(video_url).path.split("/")[-1]
    return re.sub(r"[^A-Za-z0-9_-]", "", tail)[:11]