from flask import Flask, request, send_file
from PIL import Image
import io
from flask_cors import CORS
import logging
from io import BytesIO
from diffusers import DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import make_image_grid
from diffusers import AutoencoderKL
import torch

def get_inputs(prompt, guidance, inf_steps, seed, batch_size=1):
    generator = [torch.Generator("cuda").manual_seed(i+seed) for i in range(batch_size)]
    prompts = batch_size * [prompt]
    num_inference_steps = inf_steps
    return {
        "prompt": prompts,
        "generator": generator,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance
    }

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16
).to("cuda")

pipeline = DiffusionPipeline.from_pretrained(
    model_id, use_safetensors=True, torch_dtype=torch.float16
)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_attention_slicing()
pipeline.vae = vae
pipeline = pipeline.to("cuda")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('werkzeug')

@app.before_request
def log_request_info():
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request path: {request.path}")
    logger.info(f"Request headers: {request.headers}")
    if request.is_json:
        logger.info(f"Request JSON data: {request.get_json()}")
    else:
        logger.info(f"Request form data: {request.form}")

def process_request(prompt, infnum, guidance, seed):
    # Placeholder function to handle the logic with prompt, infnum, and guidance
    # For now, just print them
    print(f"Prompt: {prompt}")
    print(f"Inference Number: {infnum}")
    print(f"Guidance: {guidance}")
    print(f"Seed: {seed}")

    images = pipeline(**get_inputs(prompt, guidance, infnum, seed,batch_size=4)).images
    img = make_image_grid(images, 2, 2)

    return img

@app.route('/api', methods=['POST'])
def api():
    data = request.get_json()

    if not data or 'prompt' not in data or 'infnum' not in data or 'guidance' not in data:
        return "Invalid JSON", 400

    prompt = data['prompt']
    infnum = int(data['infnum'])
    guidance = float(data['guidance'])
    seed = int(data['seed'])

    # Return an empty PNG image
    image = process_request(prompt, infnum, guidance, seed)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    return send_file(buffered, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)