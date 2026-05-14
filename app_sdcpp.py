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
OUT_DIR = Path(os.getenv("SDCPP_OUT_DIR", "/tmp/sdcpp-out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULTS = {
    "sampling": "euler",
    "steps": 8,
    "cfg": 1.0,
    "width": 1024,
    "height": 1024,
    "denoise": 0.55,
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


def run_generate(prompt: str, negative: str, reference_image: str | None, seed: int):
    if not prompt.strip():
        raise gr.Error("Bitte Prompt eingeben.")
    if not Path(SDCPP_BIN).exists():
        raise gr.Error(f"stable-diffusion.cpp Binary nicht gefunden: {SDCPP_BIN}")

    model_path = ensure_model()
    final_seed = random.randint(0, 2**32 - 1) if seed == -1 else int(seed)
    out_file = OUT_DIR / f"flux2_{final_seed}.png"

    cmd = [
        SDCPP_BIN,
        "-M", "img2img" if reference_image else "txt2img",
        "--diffusion-model", str(model_path),
        "--prompt", prompt,
        "--negative-prompt", negative or "",
        "--sampling-method", DEFAULTS["sampling"],
        "--steps", str(DEFAULTS["steps"]),
        "--cfg-scale", str(DEFAULTS["cfg"]),
        "--width", str(DEFAULTS["width"]),
        "--height", str(DEFAULTS["height"]),
        "--seed", str(final_seed),
        "--output", str(out_file),
    ]

    if reference_image:
        cmd += ["--init-img", reference_image, "--strength", str(DEFAULTS["denoise"])]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired as exc:
        raise gr.Error("sd.cpp Timeout nach 15 Minuten. Bitte später erneut versuchen.") from exc

    if proc.returncode != 0:
        details = (proc.stderr or proc.stdout or "Unbekannter Fehler")[-5000:]
        raise gr.Error("sd.cpp Fehler:\n" + details)
    if not out_file.exists():
        raise gr.Error("Kein Output-Bild erzeugt.")

    mode = "img2img" if reference_image else "txt2img"
    return str(out_file), final_seed, " ".join(shlex.quote(c) for c in cmd), f"✅ Fertig ({mode})"


with gr.Blocks(title="Flux2 GGUF via stable-diffusion.cpp", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# FLUX.2 klein 4B (GGUF) • SD.cpp")
    gr.Markdown("Einfacher Flow: Prompt + optional Referenzbild (img2img).")

    status = gr.Markdown("🟡 Bereit")
    prompt = gr.Textbox(label="Prompt", lines=4, value="cinematic portrait, ultra detailed, soft light")
    negative = gr.Textbox(label="Negative Prompt", lines=2, value="low quality, blurry, distorted")
    reference = gr.Image(label="Referenzbild (optional für img2img)", type="filepath")

    with gr.Accordion("Advanced", open=False):
        gr.Markdown("Seed: -1 = Random")
        seed = gr.Number(value=-1, precision=0, label="Seed (default random)")

    run = gr.Button("Generate", variant="primary")
    image = gr.Image(label="Output")
    used_seed = gr.Number(label="Used Seed", precision=0)
    cmdline = gr.Textbox(label="Ausgeführter sd.cpp Command", lines=3)

    run.click(
        run_generate,
        inputs=[prompt, negative, reference, seed],
        outputs=[image, used_seed, cmdline, status],
    )


demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860)
