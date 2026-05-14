import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import Flux2KleinPipeline


device = "cpu"
dtype = torch.bfloat16

text_encoder_2 = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen3-4B-GGUF",
    gguf_file="Qwen3-4B-Q8_0.gguf",
    torch_dtype=dtype,
)
tokenizer_2 = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    gguf_repo_id="unsloth/FLUX.2-klein-4B-GGUF",
    gguf_file="flux-2-klein-4b-Q4_K_M.gguf",
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
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