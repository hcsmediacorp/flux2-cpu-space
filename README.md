---
title: Flux2 GGUF (SD.cpp only)
emoji: ⚡
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# FLUX.2 klein 4B • SD.cpp GUI

CPU-friendly Hugging Face Space mit **stable-diffusion.cpp** als einzigem Backend.

## Unterstützt
- txt2img
- img2img (Referenzbild + Strength)
- Presets: Fast / Balanced / Quality
- Seed-Reproduzierbarkeit (Random/Fixed)
- Auto-Download des GGUF-Modells

## Standard-Workflow
1. Prompt eingeben
2. Optional Referenzbild für img2img hochladen
3. Preset wählen (Fast/Balanced/Quality)
4. Generate

## ENV
- `SDCPP_BIN` (Default: `/usr/local/bin/sd`)
- `FLUX_GGUF_MODEL` (Default: `/app/models/flux-2-klein-4b-Q4_K_M.gguf`)
- `FLUX_GGUF_REPO` (Default: `unsloth/FLUX.2-klein-4B-GGUF`)
- `FLUX_GGUF_FILE` (Default: `flux-2-klein-4b-Q4_K_M.gguf`)
- optional: `FLUX_CLIP_L`, `FLUX_T5XXL`, `FLUX_VAE`

## Hinweise aus Doku
- FLUX/SD.cpp nutzt oft `cfg-scale ~1.0` als guten Startpunkt.
- Für höhere Qualität: mehr Steps + größere Auflösung.
- Bei CPU-Limits zuerst Auflösung/Steps reduzieren.
