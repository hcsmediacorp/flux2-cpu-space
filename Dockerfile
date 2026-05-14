FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/.huggingface \
    SDCPP_PATH=/app/stable-diffusion.cpp

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs wget curl build-essential cmake libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone --depth 1 --recursive https://github.com/leejet/stable-diffusion.cpp.git /app/stable-diffusion.cpp \
    && cd /app/stable-diffusion.cpp && git submodule update --init --recursive

WORKDIR /app/stable-diffusion.cpp
RUN cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target sd -j1

WORKDIR /app
RUN pip install --upgrade pip \
    && pip install gradio pillow numpy huggingface_hub

COPY app_sdcpp.py /app/app_sdcpp.py
COPY SDCPP_NOTES.md /app/SDCPP_NOTES.md

EXPOSE 7860
CMD ["python", "/app/app_sdcpp.py"]