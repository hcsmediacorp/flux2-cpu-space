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
ARG SDCPP_URL=https://github.com/leejet/stable-diffusion.cpp/releases/latest/download/sd-linux-x64
RUN curl -L "$SDCPP_URL" -o /usr/local/bin/sd && chmod +x /usr/local/bin/sd

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