import os, torch
from PIL import Image
from typing import List, Dict, Any
from transformers import AutoProcessor, GroundingDinoForObjectDetection

_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
_model_id = os.environ.get("GDINO_MODEL_ID", "IDEA-Research/grounding-dino-tiny")

_processor = AutoProcessor.from_pretrained(_model_id)
_model = GroundingDinoForObjectDetection.from_pretrained(_model_id).to(_device).eval()

def _caption(prompts: List[str]) -> str:
    return " ".join([p.strip() + " ." for p in prompts if p and p.strip()])

def detect_on_image(image_path: str,
                    prompts: List[str],
                    box_threshold: float = 0.20,
                    text_threshold: float = 0.20) -> Dict[str, Any]:
    if not os.path.exists(image_path):
        return {"detections": [], "debug": {"error": "file_not_found", "path": image_path}}

    image = Image.open(image_path).convert("RGB")
    caption = _caption(prompts)

    inputs = _processor(images=image, text=caption, return_tensors="pt").to(_device)
    with torch.no_grad():
        outputs = _model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=_device)  # (H, W)

    try:
        # Newer API (kwargs)
        out = _processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes
        )[0]
    except TypeError:
        # Older API (positional-only): (outputs, input_ids, target_sizes, box_thr, text_thr)
        out = _processor.post_process_grounded_object_detection(
            outputs, 
            inputs.input_ids, 
            box_threshold, 
            text_threshold,
            target_sizes
        )[0]

    dets = []
    for box, score, label in zip(out["boxes"], out["scores"], out["labels"]):
        x0, y0, x1, y1 = [float(v) for v in box]
        dets.append({"label": label, "score": float(score), "box": [x0, y0, x1 - x0, y1 - y0]})

    return {
        "detections": dets,
        "debug": {
            "device": _device,
            "model_id": _model_id,
            "caption": caption,
            "image_size": list(image.size),
            "thresholds": {"box": box_threshold, "text": text_threshold}
        }
    }
