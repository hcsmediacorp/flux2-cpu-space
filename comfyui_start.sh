#!/usr/bin/env bash
set -euo pipefail

cd /app/ComfyUI
mkdir -p models/unet models/clip models/vae input output user/default/workflows

python - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path

items = [
    ("unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q4_K_M.gguf", "models/unet"),
    ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q8_0.gguf", "models/clip"),
]

for repo, filename, target_dir in items:
    target = Path(target_dir) / filename
    if target.exists():
        print(f"Already present: {target}")
        continue
    print(f"Downloading {repo}/{filename} -> {target}")
    path = hf_hub_download(repo_id=repo, filename=filename)
    target.symlink_to(path)
PY

exec python main.py --listen 0.0.0.0 --port 7860 --cpu
