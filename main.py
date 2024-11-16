from flask import Flask, request, send_file, jsonify
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
import os

def get_inputs(prompt, nprompt, guidance, inf_steps, seed, batch_size=1):
    generator = [torch.Generator("cuda").manual_seed(i+seed) for i in range(batch_size)]
    prompts = batch_size * [prompt]
    nprompts = batch_size * [nprompt]
    num_inference_steps = inf_steps
    return {
        "prompt": prompts,
        "negative_prompt": nprompts,
        "generator": generator,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance
    }

def load_model():
    model_id = os.getenv("MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5")
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
    return pipeline

pipeline = load_model()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('image-gen')

@app.before_request
def log_request_info():
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request path: {request.path}")

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        logger.info(f"Request path: {request.json}")
        data = request.json
        prompt = data.get('positive_prompt', 'an abstract image about an ai generating an abstract image')
        nprompt = data.get('negative_prompt', 'nsfw')
        guidance = data.get('guidance', 7.5)
        inf_steps = data.get('infnum', 50)
        seed = data.get('seed', 42)
        batch_size = data.get('batch_size', 1)

        inputs = get_inputs(prompt, nprompt, guidance, inf_steps, seed, batch_size)
        images = pipeline(**inputs).images

        # Convert images to bytes and send as response
        img_byte_arr = BytesIO()
        images[0].save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return send_file(io.BytesIO(img_byte_arr), mimetype='image/png')

    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)