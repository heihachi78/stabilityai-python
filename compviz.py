#https://huggingface.co/CompVis/stable-diffusion-v1-4

import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")

image = pipe(
    "A detailed and realistic photo of a group of beautiful and perfect daffodils on a meadow. Do not make it artistic or paint like it should look like a sharp realistic photo.",
    num_inference_steps=64,
    guidance_scale=5.0,
).images[0]
image.save("daffodil-compviz.png")
