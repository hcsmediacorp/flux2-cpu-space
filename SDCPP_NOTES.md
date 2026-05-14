# stable-diffusion.cpp Backend (FLUX.2 Klein GGUF)

## Warum
Für GGUF-Dateien ist `stable-diffusion.cpp` (auf `ggml`) ein direkter und robuster Inferenzpfad.

## Relevante Doku
- https://github.com/cmdr2/stable-diffusion.cpp
- https://github.com/leejet/stable-diffusion.cpp
- https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/flux.md
- https://huggingface.co/black-forest-labs/FLUX.2-klein-4B

## Neue App
- `app_sdcpp.py` nutzt `sd` CLI statt ComfyUI API.
- Erwartete ENV Variablen:
  - `SDCPP_BIN` (z. B. `/app/stable-diffusion.cpp/bin/sd`)
  - `FLUX_GGUF_MODEL` (Pfad zur FLUX.2 Klein GGUF Datei)
  - optional: `FLUX_CLIP_L`, `FLUX_T5XXL`, `FLUX_VAE`

## Beispielstart
```bash
python app_sdcpp.py
```

## Beispiel CLI (direkt)
```bash
$SDCPP_BIN -M txt2img \
  --diffusion-model $FLUX_GGUF_MODEL \
  --prompt "cinematic portrait" \
  --negative-prompt "blurry" \
  --steps 20 --cfg-scale 3.5 \
  --width 1024 --height 1024 \
  --seed 42 --output /tmp/out.png
```

## Hinweis
Je nach konkretem FLUX.2-Paket werden zusätzliche Encoder/VAE-Dateien benötigt. Dafür sind die optionalen ENV-Variablen vorgesehen.
