import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import DiffusionPipeline
from modelscope.pipelines import pipeline
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import torch
import cv2
import io
import base64
from PIL import Image

app = FastAPI(title="🎨🎬🔊 Unified AI Suite")

# 🌟 GLOBAL MODELS (lazy load)
models = {"image": None, "video": None}
device = "cpu"

# Remove preload_audio from startup
# @app.on_event("startup")
# async def preload_audio():
#     """Audio loads fast - preload immediately"""
#     preload_models()
#     print("✅ Audio models preloaded")

@app.post("/generate/image")
async def generate_image(prompt: str, steps: int = 20, width: int = 1024, height: int = 1024):
    if models["image"] is None:
        models["image"] = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large", 
            torch_dtype=torch.float32  # Changed to float32 for CPU
        ).to(device)
        print("✅ Image model loaded")
    
    try:
        pipe = models["image"]
        image = pipe(prompt, num_inference_steps=steps, width=width, height=height).images[0]
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return {"image_b64": base64.b64encode(buffered.getvalue()).decode()}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/generate/video")
async def generate_video(prompt: str, steps: int = 25):
    if models["video"] is None:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"🚀 Loading video model attempt {attempt + 1}")
                models["video"] = pipeline(
                    "text-to-video-synthesis", 
                    model="Wan-AI/Wan2.2-TI2V-5B", 
                    model_revision="bf16",  # Keep bf16, but monitor CPU performance
                    cache_dir="/app/model_cache"
                )
                print("✅ Video model loaded!")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise HTTPException(500, f"Video model failed: {e}")
                time.sleep(5)
    
    try:
        output = models["video"]({"text": prompt, "num_inference_steps": steps})
        frame = output["videos"][0][0]
        ret, buffer = cv2.imencode('.png', frame)
        return {"video_frame_b64": base64.b64encode(buffer).decode()}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/generate/audio")
async def generate_audio(text: str):
    # Load models only when needed
    preload_models()
    print("✅ Audio models preloaded")
    
    try:
        audio_array = generate_audio(text)
        buffer = io.BytesIO()
        write_wav(buffer, SAMPLE_RATE, audio_array)
        return {"audio_b64": base64.b64encode(buffer.getvalue()).decode()}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/health")
async def health():
    return {"status": "🚀 Unified AI Suite LIVE", "services": ["image", "video", "audio"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)