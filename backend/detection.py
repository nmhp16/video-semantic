import os, logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def detect_on_image(
    image_path: str,
    prompts: List[str],
    box_threshold: float = 0.20,
    text_threshold: float = 0.20,
) -> Dict[str, Any]:
    """Detect objects using moondream2's built-in detect() — reuses the shared
    model instance so there is zero extra loading cost over captioning."""
    if not os.path.exists(image_path):
        return {"detections": [], "debug": {"error": "file_not_found", "path": image_path}}

    text_queries = [p.strip() for p in prompts if p and p.strip()]
    if not text_queries:
        return {"detections": [], "debug": {"error": "no_prompts"}}

    from caption_model import get_captioner
    from PIL import Image as _Pil

    captioner = get_captioner()
    image = _Pil.open(image_path).convert("RGB")
    w, h = image.size
    enc = captioner.model.encode_image(image)

    detections = []
    for prompt in text_queries:
        try:
            result = captioner.model.detect(enc, prompt, captioner.tokenizer)
            for obj in result.get("objects", []):
                x0 = obj["x_min"] * w
                y0 = obj["y_min"] * h
                x1 = obj["x_max"] * w
                y1 = obj["y_max"] * h
                detections.append({
                    "label": prompt,
                    "score": 1.0,
                    "box": [x0, y0, x1 - x0, y1 - y0],
                })
        except Exception:
            logger.exception("moondream2 detect() failed for %r on %s", prompt, image_path)

    return {
        "detections": detections,
        "debug": {"model": "moondream2", "queries": text_queries, "image_size": [w, h]},
    }
