# https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
# https://huggingface.co/stabilityai/stable-diffusion-3.5-large
# https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo

from operator import contains
from transformers import T5EncoderModel
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch

model_id = "stabilityai/stable-diffusion-3.5-medium"

if "medium" in model_id:
    load_in_8bit = True
    load_in_4bit = False
else:
    load_in_8bit = False
    load_in_4bit = True

nf4_config = BitsAndBytesConfig(
    load_in_4bit=load_in_4bit,
    load_in_8bit=load_in_8bit,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16,
)

t5_nf4 = T5EncoderModel.from_pretrained("diffusers/t5-nf4", torch_dtype=torch.bfloat16)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id, transformer=model_nf4, text_encoder_3=t5_nf4, torch_dtype=torch.bfloat16
)
pipeline.enable_model_cpu_offload()

prompt = "A detailed and realistic photo of a group of beautiful and perfect daffodils on a meadow. Do not make it artistic or paint like it should look like a sharp realistic photo."

image = pipeline(
    prompt=prompt,
    num_inference_steps=28,
    guidance_scale=12.0,
    max_sequence_length=512,
).images[0]
image.save("daffodil-medium3.png")
