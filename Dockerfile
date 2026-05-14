FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/.huggingface \
    COMFYUI_PATH=/app/ComfyUI

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs wget curl build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI \
    && git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git /app/ComfyUI/custom_nodes/ComfyUI-GGUF

WORKDIR /app/ComfyUI

RUN pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio \
    && pip install -r requirements.txt \
    && pip install -r custom_nodes/ComfyUI-GGUF/requirements.txt \
    && pip install gradio huggingface_hub gguf safetensors sentencepiece protobuf opencv-python-headless

COPY app.py /app/app.py
COPY comfyui_start.sh /app/comfyui_start.sh
COPY comfyui_flux2_gguf_workflow.json /app/ComfyUI/user/default/workflows/flux2_gguf_img2img_seed_workflow.json
COPY comfyui_flux2_gguf_api_workflow.json /app/ComfyUI/user/default/workflows/flux2_gguf_api_workflow.json
RUN chmod +x /app/comfyui_start.sh

EXPOSE 7860
CMD ["python", "/app/app.py"]
