import os
import random
import shlex
import subprocess
from pathlib import Path

import gradio as gr
from huggingface_hub import hf_hub_download

SDCPP_BIN = os.getenv("SDCPP_BIN", "/usr/local/bin/sd")
DIFFUSION_MODEL = os.getenv("FLUX_GGUF_MODEL", "/app/models/flux-2-klein-4b-Q4_K_M.gguf")
MODEL_REPO = os.getenv("FLUX_GGUF_REPO", "unsloth/FLUX.2-klein-4B-GGUF")
MODEL_FILE = os.getenv("FLUX_GGUF_FILE", "flux-2-klein-4b-Q4_K_M.gguf")
CLIP_L = os.getenv("FLUX_CLIP_L", "")
T5XXL = os.getenv("FLUX_T5XXL", "")
VAE = os.getenv("FLUX_VAE", "")
OUT_DIR = Path(os.getenv("SDCPP_OUT_DIR", "/tmp/sdcpp-out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRESETS = {
    "Fast (CPU)": {"steps": 4, "cfg": 1.0, "width": 768, "height": 768},
    "Balanced": {"steps": 8, "cfg": 1.0, "width": 1024, "height": 1024},
    "Quality": {"steps": 16, "cfg": 1.0, "width": 1024, "height": 1024},
}


def ensure_model() -> Path:
    target = Path(DIFFUSION_MODEL)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    downloaded = Path(hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE))
    if not target.exists():
        try:
            target.symlink_to(downloaded)
        except OSError:
            target.write_bytes(downloaded.read_bytes())
    return target


def apply_preset(name: str):
    p = PRESETS.get(name, PRESETS["Balanced"])
    return p["steps"], p["cfg"], p["width"], p["height"]


def run_generate(prompt: str, negative: str, reference_image: str | None, denoise: float, seed_mode: str, seed: int, steps: int, cfg: float, width: int, height: int, sampling: str):
    if not prompt.strip():
        raise gr.Error("Bitte Prompt eingeben.")
    if not Path(SDCPP_BIN).exists():
        raise gr.Error(f"stable-diffusion.cpp Binary nicht gefunden: {SDCPP_BIN}")

    model_path = ensure_model()

    final_seed = random.randint(0, 2**32 - 1) if seed_mode == "Random" else int(seed)
    out_file = OUT_DIR / f"flux2_{final_seed}.png"

    cmd = [
        SDCPP_BIN,
        "-M", "img2img" if reference_image else "txt2img",
        "--diffusion-model", str(model_path),
        "--prompt", prompt,
        "--negative-prompt", negative or "",
        "--sampling-method", sampling,
        "--steps", str(int(steps)),
        "--cfg-scale", str(float(cfg)),
        "--width", str(int(width)),
        "--height", str(int(height)),
        "--seed", str(final_seed),
        "--output", str(out_file),
    ]

    if reference_image:
        cmd += ["--init-img", reference_image, "--strength", str(float(denoise))]
    if CLIP_L:
        cmd += ["--clip_l", CLIP_L]
    if T5XXL:
        cmd += ["--t5xxl", T5XXL]
    if VAE:
        cmd += ["--vae", VAE]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired as exc:
        raise gr.Error("sd.cpp Timeout nach 15 Minuten. Bitte Steps/Auflösung reduzieren.") from exc

    if proc.returncode != 0:
        details = (proc.stderr or proc.stdout or "Unbekannter Fehler")[-5000:]
        raise gr.Error("sd.cpp Fehler:\n" + details)
    if not out_file.exists():
        raise gr.Error("Kein Output-Bild erzeugt.")

    mode = "img2img" if reference_image else "txt2img"
    return str(out_file), final_seed, " ".join(shlex.quote(c) for c in cmd), f"✅ Fertig ({mode})"


with gr.Blocks(title="Flux2 GGUF via stable-diffusion.cpp", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# FLUX.2 klein 4B (GGUF) • stable-diffusion.cpp")
    gr.Markdown("Dokubasierte GUI: txt2img + img2img, Presets und Auto-Modell-Download.")

    status = gr.Markdown("🟡 Bereit")
    prompt = gr.Textbox(label="Prompt", lines=4, value="cinematic portrait, ultra detailed, soft light")
    negative = gr.Textbox(label="Negative Prompt", lines=2, value="low quality, blurry, distorted")

    with gr.Row():
        preset = gr.Dropdown(list(PRESETS.keys()), value="Balanced", label="Workflow Preset")
        seed_mode = gr.Radio(["Random", "Fixed"], value="Random", label="Seed Mode")
        seed = gr.Number(value=42, precision=0, label="Seed")

    reference = gr.Image(label="Referenzbild (optional für img2img)", type="filepath")
    denoise = gr.Slider(0.1, 1.0, value=0.55, step=0.05, label="Denoise/Strength (img2img)")

    with gr.Row():
        steps = gr.Slider(1, 40, value=8, step=1, label="Steps")
        cfg = gr.Slider(0.0, 12.0, value=1.0, step=0.1, label="CFG")
        sampling = gr.Dropdown(["euler", "heun", "dpm2", "dpm++2m"], value="euler", label="Sampler")

    with gr.Row():
        width = gr.Slider(256, 1536, value=1024, step=64, label="Width")
        height = gr.Slider(256, 1536, value=1024, step=64, label="Height")

    preset.change(apply_preset, inputs=[preset], outputs=[steps, cfg, width, height])

    run = gr.Button("Generate", variant="primary")
    image = gr.Image(label="Output")
    used_seed = gr.Number(label="Used Seed", precision=0)
    cmdline = gr.Textbox(label="Ausgeführter sd.cpp Command", lines=3)

    run.click(
        run_generate,
        inputs=[prompt, negative, reference, denoise, seed_mode, seed, steps, cfg, width, height, sampling],
        outputs=[image, used_seed, cmdline, status],
    )


demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860)
