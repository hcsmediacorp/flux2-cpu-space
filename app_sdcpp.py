import os
import random
import shlex
import stat
import subprocess
from pathlib import Path

import gradio as gr

SDCPP_BIN = os.getenv("SDCPP_BIN", "/usr/local/bin/sd")
DIFFUSION_MODEL = Path(os.getenv("FLUX_GGUF_MODEL", "/app/models/flux-2-klein-4b-Q4_K_M.gguf"))
LLM_MODEL = Path(os.getenv("FLUX_LLM_MODEL", "/app/models/qwen_3_4b_q8_0.gguf"))
VAE_MODEL = Path(os.getenv("FLUX_VAE_MODEL", "/app/models/flux2_ae.safetensors"))
OUT_DIR = Path(os.getenv("SDCPP_OUT_DIR", "/tmp/sdcpp-out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# HF Spaces CPU basic defaults: 2 vCPU / 16 GB RAM => conservative generation settings
DEFAULTS = {"sampling": "euler", "steps": 3, "cfg": 1.0, "width": 768, "height": 768, "denoise": 0.55}


def resolve_sd_bin() -> Path:
    candidates = [Path(SDCPP_BIN), Path("/usr/local/bin/sd-cli"), Path("/usr/local/bin/sd")]
    for c in candidates:
        if c.exists():
            mode = c.stat().st_mode
            if not (mode & stat.S_IXUSR):
                raise gr.Error(f"sd.cpp Binary ist nicht ausführbar: {c}")
            # Quick ELF sanity-check to surface Exec format issues with clear message
            with c.open("rb") as fh:
                magic = fh.read(4)
            if magic != b"\x7fELF":
                raise gr.Error(
                    f"Ungültiges sd.cpp Binary-Format ({c}). Erwartet ELF, bekam magic={magic!r}. "
                    "Vermutlich falsche Architektur oder fehlerhafter Download im Docker-Build."
                )
            return c
    raise gr.Error("sd.cpp Binary nicht gefunden (erwartet z.B. /usr/local/bin/sd-cli)")


def validate_runtime_assets() -> None:
    missing = []
    if not DIFFUSION_MODEL.exists():
        missing.append(f"diffusion model: {DIFFUSION_MODEL}")
    if not LLM_MODEL.exists():
        missing.append(f"llm/text encoder: {LLM_MODEL}")
    if not VAE_MODEL.exists():
        missing.append(f"vae: {VAE_MODEL}")
    if missing:
        raise gr.Error("Fehlende Build-Artefakte:\n- " + "\n- ".join(missing))


def _run_cmd(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return proc.returncode, (proc.stderr or proc.stdout or "")


def run_generate(prompt: str, negative: str, reference_image: str | None, seed: int):
    if not prompt.strip():
        raise gr.Error("Bitte Prompt eingeben.")

    sd_bin = resolve_sd_bin()
    validate_runtime_assets()
    final_seed = random.randint(0, 2**32 - 1) if seed == -1 else int(seed)
    out_file = OUT_DIR / f"flux2_{final_seed}.png"

    # Primary syntax
    cmd_a = [
        str(sd_bin),
        "-M", "img2img" if reference_image else "txt2img",
        "--diffusion-model", str(DIFFUSION_MODEL),
        "--llm", str(LLM_MODEL),
        "--vae", str(VAE_MODEL),
        "--prompt", prompt,
        "--negative-prompt", negative or "",
        "--sampling-method", DEFAULTS["sampling"],
        "--steps", str(DEFAULTS["steps"]),
        "--cfg-scale", str(DEFAULTS["cfg"]),
        "--width", str(DEFAULTS["width"]),
        "--height", str(DEFAULTS["height"]),
        "--seed", str(final_seed),
        "--output", str(out_file),
        "--offload-to-cpu",
    ]
    if reference_image:
        cmd_a += ["--init-img", reference_image, "--strength", str(DEFAULTS["denoise"])]

    # Fallback syntax seen in sd.cpp docs
    cmd_b = [
        str(sd_bin),
        "-M", "img2img" if reference_image else "txt2img",
        "--diffusion-model", str(DIFFUSION_MODEL),
        "--llm", str(LLM_MODEL),
        "--vae", str(VAE_MODEL),
        "-p", prompt,
        "-n", negative or "",
        "--sampling-method", DEFAULTS["sampling"],
        "--steps", str(DEFAULTS["steps"]),
        "--cfg-scale", str(DEFAULTS["cfg"]),
        "--width", str(DEFAULTS["width"]),
        "--height", str(DEFAULTS["height"]),
        "--seed", str(final_seed),
        "-o", str(out_file),
        "--offload-to-cpu",
    ]
    if reference_image:
        cmd_b += ["-r", reference_image, "--strength", str(DEFAULTS["denoise"])]

    try:
        rc, log = _run_cmd(cmd_a)
        used = cmd_a
        if rc != 0 or not out_file.exists():
            rc, log2 = _run_cmd(cmd_b)
            used = cmd_b
            log = log + "\n--- fallback ---\n" + log2
        if rc != 0 or not out_file.exists():
            raise gr.Error("sd.cpp Fehler:\n" + log[-6000:])
    except subprocess.TimeoutExpired as exc:
        raise gr.Error("sd.cpp Timeout nach 10 Minuten (CPU Space Limit). Bitte niedrigere Auflösung/Steps versuchen.") from exc

    mode = "img2img" if reference_image else "txt2img"
    return str(out_file), final_seed, " ".join(shlex.quote(c) for c in used), f"✅ Fertig ({mode})"


with gr.Blocks(title="Flux2 GGUF via stable-diffusion.cpp") as demo:
    gr.Markdown("# FLUX.2 klein 4B (GGUF) • SD.cpp CPU-only")
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

    run.click(run_generate, inputs=[prompt, negative, reference, seed], outputs=[image, used_seed, cmdline, status])


demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
