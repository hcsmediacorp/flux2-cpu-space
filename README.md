---
title: Flux2 GGUF (SD.cpp only)
emoji: ⚡
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# FLUX.2 klein 4B GGUF • SD.cpp only

Dieses Projekt nutzt **ausschließlich stable-diffusion.cpp (ggml/GGUF)**.

## Enthalten
- Gradio UI (`app_sdcpp.py`)
- Inferenz über `sd` Binary (stable-diffusion.cpp)
- Reproduzierbare Seeds, Sampler-Auswahl, Steps/CFG/Resolution

## ENV
- `SDCPP_BIN` (Default: `/app/stable-diffusion.cpp/build/bin/sd`)
- `FLUX_GGUF_MODEL` (Pfad zur GGUF-Datei)
- optional: `FLUX_CLIP_L`, `FLUX_T5XXL`, `FLUX_VAE`

## Run
```bash
uv run python app_sdcpp.py
```
