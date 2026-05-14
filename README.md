---
title: Flux2 GGUF (stable-diffusion.cpp)
emoji: ⚡
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# FLUX.2 klein 4B GGUF

Primäres Backend ist jetzt **stable-diffusion.cpp (ggml/GGUF)**.

## Features
- Direkte GGUF-Inferenz via `sd` CLI
- Gradio UI mit Seed/Sampler/Steps/CFG/Resolution
- Reproduzierbare Runs (Fixed Seed)
- Docker-Start standardmäßig über `app_sdcpp.py`

## Wichtig
- Modellpfad per ENV: `FLUX_GGUF_MODEL`
- Binary per ENV: `SDCPP_BIN` (Default: `/app/stable-diffusion.cpp/build/bin/sd`)
- Optional: `FLUX_CLIP_L`, `FLUX_T5XXL`, `FLUX_VAE`

## Lokaler Check
```bash
uv run python -m py_compile app_sdcpp.py
uv run --with pytest pytest -q
```
