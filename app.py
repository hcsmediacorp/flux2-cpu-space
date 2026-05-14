import json
import os
import random
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any
from urllib import request, parse

import gradio as gr

COMFY_DIR = Path(os.getenv("COMFYUI_PATH", "/app/ComfyUI"))
COMFY_HOST = "127.0.0.1"
COMFY_PORT = 8188
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"
CLIENT_ID = str(uuid.uuid4())
WORKFLOW_PATH = COMFY_DIR / "user/default/workflows/flux2_gguf_api_workflow.json"


def link_or_copy(source: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    try:
        target.symlink_to(source)
    except OSError:
        target.write_bytes(Path(source).read_bytes())


def ensure_models() -> None:
    from huggingface_hub import hf_hub_download

    downloads = [
        (
            "unsloth/FLUX.2-klein-4B-GGUF",
            "flux-2-klein-4b-Q4_K_M.gguf",
            [COMFY_DIR / "models/unet/flux-2-klein-4b-Q4_K_M.gguf", COMFY_DIR / "models/diffusion_models/flux-2-klein-4b-Q4_K_M.gguf"],
        ),
        (
            "unsloth/Qwen3-4B-GGUF",
            "Qwen3-4B-Q8_0.gguf",
            [COMFY_DIR / "models/clip/Qwen3-4B-Q8_0.gguf", COMFY_DIR / "models/text_encoders/Qwen3-4B-Q8_0.gguf"],
        ),
        (
            "black-forest-labs/FLUX.2-klein-4B",
            "vae/diffusion_pytorch_model.safetensors",
            [COMFY_DIR / "models/vae/ae.safetensors"],
        ),
    ]
    for repo_id, filename, targets in downloads:
        local_path = hf_hub_download(repo_id=repo_id, filename=filename)
        for target in targets:
            link_or_copy(local_path, target)


def start_comfyui() -> subprocess.Popen:
    COMFY_DIR.mkdir(parents=True, exist_ok=True)
    ensure_models()
    return subprocess.Popen(
        ["python", "main.py", "--listen", COMFY_HOST, "--port", str(COMFY_PORT), "--cpu"],
        cwd=str(COMFY_DIR),
    )


def wait_for_comfyui(timeout: int = 300) -> None:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            with request.urlopen(f"{COMFY_URL}/system_stats", timeout=5) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"ComfyUI did not start in time: {last_error}")


def api_get(path: str) -> Any:
    with request.urlopen(f"{COMFY_URL}{path}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def api_post(path: str, payload: dict[str, Any]) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{COMFY_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def upload_image(image_path: str) -> str:
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    filename = Path(image_path).name
    body = []
    body.append(f"--{boundary}\r\n".encode())
    body.append(
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        b"Content-Type: image/png\r\n\r\n"
    )
    body.append(Path(image_path).read_bytes())
    body.append(f"\r\n--{boundary}--\r\n".encode())
    req = request.Request(
        f"{COMFY_URL}/upload/image",
        data=b"".join(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with request.urlopen(req, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["name"]


def load_workflow() -> dict[str, Any]:
    if WORKFLOW_PATH.exists():
        return json.loads(WORKFLOW_PATH.read_text())
    raise FileNotFoundError(f"Workflow missing: {WORKFLOW_PATH}")


def set_node_input(workflow: dict[str, Any], node_id: str, key: str, value: Any) -> None:
    if node_id in workflow and "inputs" in workflow[node_id]:
        workflow[node_id]["inputs"][key] = value


def patch_workflow(
    workflow: dict[str, Any],
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    width: int,
    height: int,
    denoise: float,
    image_name: str | None,
) -> dict[str, Any]:
    # Node IDs match comfyui_flux2_gguf_api_workflow.json. If ComfyUI-GGUF changes node names,
    # users can still paste a corrected workflow into the advanced box and keep these IDs.
    set_node_input(workflow, "6", "text", prompt)
    set_node_input(workflow, "7", "text", negative_prompt or "")
    set_node_input(workflow, "3", "seed", seed)
    set_node_input(workflow, "3", "steps", steps)
    set_node_input(workflow, "3", "cfg", cfg)
    set_node_input(workflow, "3", "denoise", denoise)
    set_node_input(workflow, "5", "width", width)
    set_node_input(workflow, "5", "height", height)
    if image_name:
        set_node_input(workflow, "12", "image", image_name)
        # Switch KSampler latent input to VAEEncode output for img2img.
        if "3" in workflow:
            workflow["3"]["inputs"]["latent_image"] = ["13", 0]
    return workflow


def queue_prompt(workflow: dict[str, Any]) -> str:
    result = api_post("/prompt", {"prompt": workflow, "client_id": CLIENT_ID})
    return result["prompt_id"]


def fetch_outputs(prompt_id: str) -> list[str]:
    history = api_get(f"/history/{prompt_id}")
    outputs = []
    prompt_history = history.get(prompt_id, {})
    for node in prompt_history.get("outputs", {}).values():
        for image in node.get("images", []):
            query = parse.urlencode(
                {
                    "filename": image["filename"],
                    "subfolder": image.get("subfolder", ""),
                    "type": image.get("type", "output"),
                }
            )
            outputs.append(f"{COMFY_URL}/view?{query}")
    return outputs


def generate(
    prompt: str,
    negative_prompt: str,
    reference_image: str | None,
    seed_mode: str,
    seed: int,
    steps: int,
    cfg: float,
    width: int,
    height: int,
    denoise: float,
    workflow_json: str,
    progress=gr.Progress(track_tqdm=True),
):
    if not prompt.strip():
        raise gr.Error("Bitte Prompt eingeben.")

    final_seed = random.randint(0, 2**32 - 1) if seed_mode == "Random" else int(seed)
    progress(0.05, desc="Workflow vorbereiten")

    image_name = upload_image(reference_image) if reference_image else None
    workflow = json.loads(workflow_json) if workflow_json.strip() else load_workflow()
    workflow = patch_workflow(
        workflow,
        prompt,
        negative_prompt,
        final_seed,
        int(steps),
        float(cfg),
        int(width),
        int(height),
        float(denoise),
        image_name,
    )

    progress(0.15, desc="ComfyUI Queue senden")
    prompt_id = queue_prompt(workflow)

    for i in range(240):
        progress(min(0.95, 0.15 + i / 260), desc=f"Generiere… Prompt ID {prompt_id}")
        outputs = fetch_outputs(prompt_id)
        if outputs:
            progress(1.0, desc="Fertig")
            return outputs, final_seed, json.dumps(workflow, indent=2)
        time.sleep(2)

    raise gr.Error("Timeout: Keine Ausgabe von ComfyUI erhalten. Prüfe Logs/Workflow-Nodes.")


comfy_process = start_comfyui()
wait_for_comfyui()
DEFAULT_WORKFLOW = WORKFLOW_PATH.read_text() if WORKFLOW_PATH.exists() else "{}"

with gr.Blocks(title="Flux2 GGUF Workflow UI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Flux2 GGUF ComfyUI Workflow\nSeed, Random Seed, Image-to-Image, Progress und direkte ComfyUI API-Queue.")
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt", lines=4, value="cinematic portrait, ultra detailed, soft light")
            negative = gr.Textbox(label="Negative Prompt", lines=2, value="low quality, blurry, distorted")
            reference = gr.Image(label="Image-to-Image Referenz (optional)", type="filepath")
            with gr.Row():
                seed_mode = gr.Radio(["Random", "Fixed"], value="Random", label="Seed Mode")
                seed = gr.Number(value=42, precision=0, label="Seed")
            with gr.Row():
                steps = gr.Slider(1, 12, value=4, step=1, label="Steps")
                cfg = gr.Slider(0.0, 8.0, value=1.0, step=0.1, label="CFG")
            with gr.Row():
                width = gr.Slider(512, 1280, value=1024, step=64, label="Width")
                height = gr.Slider(512, 1280, value=1024, step=64, label="Height")
            denoise = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Denoise (1.0 txt2img, 0.35–0.75 img2img)")
            run = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            gallery = gr.Gallery(label="Outputs", columns=2, height=520)
            used_seed = gr.Number(label="Used Seed", precision=0)
    with gr.Accordion("Advanced: ComfyUI API Workflow JSON", open=False):
        workflow_box = gr.Code(label="Workflow JSON", language="json", value=DEFAULT_WORKFLOW, lines=24)
        patched = gr.Code(label="Last submitted workflow", language="json", lines=24)

    run.click(
        generate,
        inputs=[prompt, negative, reference, seed_mode, seed, steps, cfg, width, height, denoise, workflow_box],
        outputs=[gallery, used_seed, patched],
    )


demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860)
