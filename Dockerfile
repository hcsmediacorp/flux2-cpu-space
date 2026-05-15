FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    SDCPP_BIN=/usr/local/bin/sd \
    FLUX_GGUF_MODEL=/app/models/flux-2-klein-4b-Q4_K_M.gguf \
    FLUX_LLM_MODEL=/app/models/qwen_3_4b_q8_0.gguf \
    FLUX_VAE_MODEL=/app/models/flux2_ae.safetensors

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Prebuilt sd binary (CPU-friendly for HF Spaces)
ARG SDCPP_URL_BASE=https://github.com/leejet/stable-diffusion.cpp/releases/latest/download
RUN set -eux; \
    arch="$(uname -m)"; \
    case "$arch" in \
      x86_64) sd_artifact="sd-linux-x64" ;; \
      aarch64|arm64) sd_artifact="sd-linux-arm64" ;; \
      *) echo "Unsupported architecture: $arch" >&2; exit 1 ;; \
    esac; \
    curl -fL "$SDCPP_URL_BASE/$sd_artifact" -o /usr/local/bin/sd; \
    chmod +x /usr/local/bin/sd; \
    python - <<'PY'
from pathlib import Path
p = Path('/usr/local/bin/sd')
b = p.read_bytes()[:4]
if b != b'\x7fELF':
    raise SystemExit(f"/usr/local/bin/sd is not an ELF binary (magic={b!r})")
PY

WORKDIR /app
RUN pip install --upgrade pip && pip install gradio pillow numpy huggingface_hub

# Pre-download required FLUX.2 klein CPU assets and normalize filenames for workflow consistency
RUN mkdir -p /app/models && \
    curl -L "https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q4_K_M.gguf" -o /app/models/flux-2-klein-4b-Q4_K_M.gguf && \
    curl -L "https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf" -o /app/models/qwen_3_4b_q8_0.gguf && \
    curl -L "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/vae/diffusion_pytorch_model.safetensors" -o /app/models/flux2_ae.safetensors

COPY app_sdcpp.py /app/app_sdcpp.py
COPY SDCPP_NOTES.md /app/SDCPP_NOTES.md
COPY README.md /app/README.md

EXPOSE 7860
CMD ["python", "/app/app_sdcpp.py"]