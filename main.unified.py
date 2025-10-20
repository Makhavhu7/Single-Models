import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🌟 GLOBAL MODELS (lazy load)
models = {"image": None, "video": None}
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

# Add root route
@app.get("/")
async def root():
    return {"message": "Welcome to Unified AI Suite. Use /health or /generate endpoints."}

@app.post("/generate/image")
async def generate_image(prompt: str, steps: int = 20, width: int = 1024, height: int = 1024):
    if models["image"] is None:
        models["image"] = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        print("✅ Image model loaded")
    
    try:
        pipe = models["image"]
        image = pipe(prompt, num_inference_steps=steps, width=width, height=height).images[0]
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        return {"image_b64": img_b64, "message": f"Generated image for prompt: {prompt}"}
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
                    model_revision="bf16",
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
        video_frame_b64 = base64.b64encode(buffer).decode()
        return {"video_frame_b64": video_frame_b64, "message": f"Generated video frame for prompt: {prompt}"}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/generate/audio")
async def generate_audio(text: str):
    preload_models()
    print("✅ Audio models preloaded")
    
    try:
        audio_array = generate_audio(text)
        buffer = io.BytesIO()
        write_wav(buffer, SAMPLE_RATE, audio_array)
        audio_b64 = base64.b64encode(buffer.getvalue()).decode()
        return {"audio_b64": audio_b64, "message": f"Generated audio for text: {text}"}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/health")
async def health():
    return {"status": "🚀 Unified AI Suite LIVE", "services": ["image", "video", "audio"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)  # Changed to 8080 to match Runpod default