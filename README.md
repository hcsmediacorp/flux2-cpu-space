---
title: Flux2 GGUF ComfyUI CPU
emoji: 🧩
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# Flux2 GGUF ComfyUI CPU Space

This Space runs **ComfyUI** with **ComfyUI-GGUF** so the workflow can use quantized GGUF models directly instead of forcing a fragile Diffusers startup path.

## Included

- ComfyUI web UI on port `7860`
- ComfyUI-GGUF custom node
- FLUX.2 klein GGUF download at startup:
  - `unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q4_K_M.gguf`
- Quantized text encoder download at startup:
  - `unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q8_0.gguf`
- Workflow notes: `comfyui_flux2_gguf_workflow.json`

## Usage

1. Open the live Space.
2. In ComfyUI, build/load a FLUX GGUF graph using the GGUF UNet loader.
3. Use KSampler seed controls for deterministic outputs.
4. For image-to-image: `Load Image → VAE Encode → KSampler latent_image`, then set denoise around `0.35–0.75`.

## Notes

FLUX.2 GGUF support is cutting-edge. If a specific ComfyUI node name differs after updates, select the downloaded files manually from `models/unet` and `models/clip`.
