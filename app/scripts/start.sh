#!/usr/bin/env bash
set -e

mkdir -p /app/checkpoints/sam3.1

python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/sam3.1",
    local_dir="/app/checkpoints/sam3.1",
    token=os.environ["HF_TOKEN"],
)
PY

exec python -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}"