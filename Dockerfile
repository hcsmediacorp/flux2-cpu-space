FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    SDCPP_BIN=/usr/local/bin/sd

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Prebuilt sd binary to avoid heavy C++ compilation on HF CPU Spaces
ARG SDCPP_URL=https://github.com/leejet/stable-diffusion.cpp/releases/latest/download/sd-linux-x64
RUN curl -L "$SDCPP_URL" -o /usr/local/bin/sd \
    && chmod +x /usr/local/bin/sd

WORKDIR /app
RUN pip install --upgrade pip && pip install gradio pillow numpy huggingface_hub

COPY app_sdcpp.py /app/app_sdcpp.py
COPY SDCPP_NOTES.md /app/SDCPP_NOTES.md

EXPOSE 7860
CMD ["python", "/app/app_sdcpp.py"]