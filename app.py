import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, GGUFQuantizationConfig


device = "cpu"
dtype = torch.bfloat16

# Quantized FLUX2 transformer (prevents full-size transformer download)
transformer_path = hf_hub_download(
    repo_id="unsloth/FLUX.2-klein-4B-GGUF",
    filename="flux-2-klein-4b-Q4_K_M.gguf",
)
transformer = Flux2Transformer2DModel.from_single_file(
    transformer_path,
    config="black-forest-labs/FLUX.2-klein-4B",
    subfolder="transformer",
    quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
    torch_dtype=dtype,
)

# Quantized text encoder 2 (Qwen3)
text_encoder_2 = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen3-4B-GGUF",
    gguf_file="Qwen3-4B-Q8_0.gguf",
    torch_dtype=dtype,
)
tokenizer_2 = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    transformer=transformer,
    text_encoder=text_encoder_2,
    tokenizer=tokenizer_2,
    torch_dtype=dtype,
)
try:
    pipe.enable_model_cpu_offload()
except Exception:
    # Fallback for environments where accelerate hooks are unavailable.
    pass
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