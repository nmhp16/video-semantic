# Shared moondream2 singleton — imported by both search.py and detection.py
# so the model is loaded once regardless of which path triggers it first.
_instance = None


def get_captioner():
    global _instance
    if _instance is None:
        from visual_ingest import Moondream2Captioner
        _instance = Moondream2Captioner()
    return _instance
