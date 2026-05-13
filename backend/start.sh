#!/bin/sh
set -a
. "$(dirname "$0")/.env"
set +a
exec uvicorn app:app --reload "$@"
