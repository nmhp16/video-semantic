import os, torch
from PIL import Image
from typing import List, Dict, Any
from transformers import Owlv2Processor, Owlv2ForObjectDetection

_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
_model_id = os.environ.get("OWL_MODEL_ID", "google/owlv2-base-patch16-ensemble")

_processor = Owlv2Processor.from_pretrained(_model_id)
_model = Owlv2ForObjectDetection.from_pretrained(_model_id).to(_device).eval()

def detect_on_image(image_path: str,
                    prompts: List[str],
                    box_threshold: float = 0.20,
                    text_threshold: float = 0.20) -> Dict[str, Any]:
    if not os.path.exists(image_path):
        return {"detections": [], "debug": {"error": "file_not_found", "path": image_path}}

    image = Image.open(image_path).convert("RGB")
    # OWLv2 expects text as a list of lists (one list of queries per image)
    text_queries = [p.strip() for p in prompts if p and p.strip()]
    if not text_queries:
        return {"detections": [], "debug": {"error": "no_prompts"}}

    inputs = _processor(text=[text_queries], images=image, return_tensors="pt").to(_device)
    with torch.no_grad():
        outputs = _model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=_device)  # (H, W)
    results = _processor.post_process_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=box_threshold
    )[0]

    dets = []
    for box, score, label_idx in zip(results["boxes"], results["scores"], results["labels"]):
        x0, y0, x1, y1 = [float(v) for v in box]
        label = text_queries[int(label_idx)] if int(label_idx) < len(text_queries) else str(int(label_idx))
        dets.append({"label": label, "score": float(score), "box": [x0, y0, x1 - x0, y1 - y0]})

    return {
        "detections": dets,
        "debug": {
            "device": _device,
            "model_id": _model_id,
            "queries": text_queries,
            "image_size": list(image.size),
            "thresholds": {"box": box_threshold, "text": text_threshold}
        }
    }
