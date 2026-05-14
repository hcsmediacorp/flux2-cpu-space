import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from diffusers import Flux2KleinPipeline, FluxTransformer2DModel, GGUFQuantizationConfig


device = "cpu"
dtype = torch.bfloat16

GGUF_PATH = hf_hub_download(
    repo_id="unsloth/FLUX.2-klein-4B-GGUF",
    filename="flux-2-klein-4b-Q6_K.gguf",
)

transformer = FluxTransformer2DModel.from_single_file(
    GGUF_PATH,
    quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
    torch_dtype=dtype,
)

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    transformer=transformer,
    torch_dtype=dtype,
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()


def generate(prompt, steps=4, guidance=4.0, seed=42):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = pipe(
        prompt,
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]
    return image


with gr.Blocks(title="Flux 2 CPU Space") as demo:
    gr.Markdown("# Flux 2 Klein 4B – CPU Optimized")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", lines=3)
        btn = gr.Button("Generate")
        output = gr.Image(label="Generated Image")
    btn.click(generate, inputs=prompt, outputs=output)


demo.launch()