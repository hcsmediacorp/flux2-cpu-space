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
- Minimal-UI: Prompt + optional img2img Referenzbild
- Seed in Advanced (default random)
- Auto-Download des GGUF-Modells

## Standard-Workflow
1. Prompt eingeben
2. Optional Referenzbild für img2img hochladen
3. Preset wählen (Fast/Balanced/Quality)
4. Generate

## ENV
- `SDCPP_BIN` (Default: `/usr/local/bin/sd`)
- `FLUX_GGUF_MODEL` (Default: `/app/models/flux-2-klein-4b-Q4_K_M.gguf`)
- `FLUX_LLM_MODEL` (Default: `/app/models/qwen_3_4b_q8_0.gguf`)
- `FLUX_VAE_MODEL` (Default: `/app/models/flux2_ae.safetensors`)

Diese 3 Dateien werden im Build vorab geladen und mit festen Dateinamen abgelegt, damit der Workflow stabil passt.

Hinweis zur Quantisierung:
- Diffusion: GGUF quantisiert (Q4_K_M)
- Text-Encoder/LLM: GGUF quantisiert (Q8_0)
- VAE: nur Safetensors (kein etabliertes GGUF-Äquivalent für diesen Pfad)

## Hinweise aus Doku
- FLUX/SD.cpp nutzt oft `cfg-scale ~1.0` als guten Startpunkt.
- Für höhere Qualität: mehr Steps + größere Auflösung.
- Bei CPU-Limits zuerst Auflösung/Steps reduzieren.
