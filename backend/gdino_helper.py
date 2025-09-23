from gdino import detect_on_image
import os
from collections import defaultdict
from typing import List, Dict, Any, Optional, Callable


_GDINO_CACHE: Dict[tuple, Any] = {}

def _val(h, name, default=None):
    if isinstance(h, dict):
        return h.get(name, default)
    return getattr(h, name, default)

def _gdino_detect_cached(frame_path: str, prompts: List[str],
                         box_th: float, text_th: float) -> Any:
    key = (os.path.abspath(frame_path), tuple(prompts), round(box_th,3), round(text_th,3))
    if key in _GDINO_CACHE:
        return _GDINO_CACHE[key]
    res = detect_on_image(
        image_path=key[0],
        prompts=list(key[1]),
        box_threshold=key[2],
        text_threshold=key[3]
    )
    _GDINO_CACHE[key] = res
    return res

def _normalize_detections(det_out: Any) -> List[Dict[str, Any]]:
    """
    Normalize various possible outputs of detect_on_image into:
      [{"label": str, "score": float, ...}, ...]
    Acceptable inputs:
      - {"detections": [...]}
      - {"results": [...]}
      - list of dicts
      - list of tuples/lists like (label, score, *rest)
    """
    if det_out is None:
        return []

    # unwrap dict containers
    if isinstance(det_out, dict):
        for k in ("detections", "results", "objects", "items"):
            if k in det_out and isinstance(det_out[k], (list, tuple)):
                det_out = det_out[k]
                break
        # if still a dict (unexpected), return empty
        if isinstance(det_out, dict):
            return []

    # now det_out should be a list/tuple
    if not isinstance(det_out, (list, tuple)):
        return []

    norm: List[Dict[str, Any]] = []
    for d in det_out:
        if isinstance(d, dict):
            # possible field names for label
            label = d.get("label")
            if label is None:
                label = d.get("text", d.get("class", d.get("category", "")))
            # possible score fields
            score = d.get("score")
            if score is None:
                score = d.get("confidence", d.get("logit", 0.0))
            try:
                norm.append({"label": str(label).lower(), "score": float(score), **d})
            except Exception:
                # if score isn't castable, skip
                continue
        elif isinstance(d, (list, tuple)) and len(d) >= 2:
            label, score = d[0], d[1]
            try:
                norm.append({"label": str(label).lower(), "score": float(score)})
            except Exception:
                continue
        # else: skip unknown element type
    return norm

def gdino_verify_score(det_out: Any, require_all: Optional[List[str]] = None) -> float:
    """
    Score is:
      - if require_all provided: average over max score of each required label; if any missing -> 0
      - else: average of max score over all labels present
    """
    dets = _normalize_detections(det_out)
    if not dets:
        return 0.0
    by_label = defaultdict(list)
    for d in dets:
        lbl = str(d.get("label", "")).lower()
        try:
            sc = float(d.get("score", 0.0))
        except Exception:
            sc = 0.0
        if lbl:
            by_label[lbl].append(sc)

    if require_all:
        req = [r.lower() for r in require_all]
        if not all(r in by_label for r in req):
            return 0.0
        vals = [max(by_label[r]) for r in req]
    else:
        vals = [max(v) for v in by_label.values()]

    if not vals:
        return 0.0
    return sum(vals) / float(len(vals))

def rerank_with_gdino(
    hits: List[Dict[str, Any]],
    prompts: List[str],
    require_all: Optional[List[str]],
    box_th: float,
    text_th: float,
    w_clip: float = 0.6,
    w_gdino: float = 0.4,
    frame_resolver: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
) -> List[Dict[str, Any]]:
    """
    For each hit:
      - resolve frame path (hit['frame'] or via frame_resolver)
      - run GDINO, compute verify score
      - compute fused = w_clip * hit['score'] + w_gdino * verify
      - attach: hit['verify_score'], hit['score_fused']; ensure hit['frame'] is set if resolved
    Returns hits sorted by fused score (desc).
    """
    out: List[Dict[str, Any]] = []
    for h in hits:
        # 1) resolve frame
        frame = h.get("frame")
        if not frame and frame_resolver:
            try:
                frame = frame_resolver(h)
            except Exception:
                frame = None

        verify = 0.0
        if frame and prompts:
            try:
                dets = _gdino_detect_cached(frame, prompts, box_th, text_th)
                verify = gdino_verify_score(dets, require_all=require_all)
            except Exception:
                verify = 0.0

        try:
            clip_s = float(h.get("score", 0.0))
        except Exception:
            clip_s = 0.0

        fused = w_clip * clip_s + w_gdino * verify
        hh = dict(h)
        if frame and not hh.get("frame"):
            hh["frame"] = frame
        hh["verify_score"] = verify
        hh["score_fused"]  = fused
        out.append(hh)

    out.sort(key=lambda x: float(x.get("score_fused", x.get("score", 0.0))), reverse=True)
    return out