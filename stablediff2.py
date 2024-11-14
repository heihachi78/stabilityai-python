from diffusers import DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import make_image_grid
from diffusers import AutoencoderKL
import torch


def get_inputs(batch_size=1):
    generator = [torch.Generator("cuda").manual_seed(i + 19780428) for i in range(batch_size)]
    prompts = batch_size * [prompt]
    num_inference_steps = 50
    return {
        "prompt": prompts,
        "generator": generator,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": 7.5
    }


prompt = "portrait photo of a old warrior chief"
prompt += ", tribal panther make up, light grey on dark grey, side profile, looking away, serious eyes"
prompt += " 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta"

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32
).to("cuda")

pipeline = DiffusionPipeline.from_pretrained(
    model_id, use_safetensors=True, torch_dtype=torch.float32
)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_attention_slicing()
pipeline.vae = vae
pipeline = pipeline.to("cuda")

images = pipeline(**get_inputs(batch_size=4)).images
image = make_image_grid(images, 2, 2)
image.save("chief.png")
