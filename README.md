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
- `SDCPP_BIN` (Default: `/usr/local/bin/sd`)
- `FLUX_GGUF_MODEL` (Zielpfad der GGUF-Datei)
- `FLUX_GGUF_REPO` (Default: `unsloth/FLUX.2-klein-4B-GGUF`)
- `FLUX_GGUF_FILE` (Default: `flux-2-klein-4b-Q4_K_M.gguf`)
- optional: `FLUX_CLIP_L`, `FLUX_T5XXL`, `FLUX_VAE`

Beim Start/ersten Run wird das Modell automatisch geladen, falls es lokal fehlt.
img2img ist integriert (Referenzbild + Denoise/Strength).

## Run
```bash
uv run python app_sdcpp.py
```
