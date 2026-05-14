"""Standalone ingest worker script — run per-job as a subprocess."""
import sys, os, json, logging

logging.basicConfig(level=logging.INFO)

def main():
    if len(sys.argv) != 4:
        print("Usage: ingest_worker.py <status_file> <url> <video_id>", file=sys.stderr)
        sys.exit(1)

    status_file, url, video_id = sys.argv[1], sys.argv[2], sys.argv[3]

    backend = os.path.dirname(os.path.abspath(__file__))
    if backend not in sys.path:
        sys.path.insert(0, backend)

    from routers.ingest import _run_ingest_proc
    _run_ingest_proc(status_file, url, video_id)

if __name__ == "__main__":
    main()
