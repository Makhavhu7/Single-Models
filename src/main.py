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
from typing import Optional

app = FastAPI(title="AI API SDXL")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {"image": None, "video": None, "audio": None}
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

def validate_prompt(prompt: str) -> str:
    """Basic prompt validation"""
    if not prompt or len(prompt.strip()) == 0:
        raise ValueError("Prompt cannot be empty")
    if len(prompt) > 1000:
        raise ValueError("Prompt too long (max 1000 characters)")
    return prompt.strip()

@app.get("/")
async def root():
    return {"message": "Welcome to AI API SDXL. Use /health or /generate endpoints."}

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 25
    refiner_inference_steps: int = 50
    guidance_scale: float = 7.5
    strength: float = 0.3
    high_noise_frac: float = 0.8
    seed: int = 1337
    scheduler: str = "K_EULER"
    num_images: int = 1
    image_url: Optional[str] = None

class VideoRequest(BaseModel):
    prompt: str
    steps: int = 25

class AudioRequest(BaseModel):
    text: str

@app.post("/generate/image")
async def generate_image(request: ImageRequest):
    try:
        prompt = validate_prompt(request.prompt)
        print(f"🖼️ Generating image for prompt: {prompt}")
        
        if models["image"] is None:
            print("Loading image model...")
            models["image"] = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-3.5-large",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                cache_dir="/app/model_cache"
            ).to(device)
            print("✅ Image model loaded")
        
        pipe = models["image"]
        
        # Generate image
        image = pipe(
            prompt,
            negative_prompt=request.negative_prompt,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            strength=request.strength,
            high_noise_frac=request.high_noise_frac,
            generator=torch.Generator(device=device).manual_seed(request.seed),
            scheduler=request.scheduler
        ).images[0]
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {"image_b64": img_b64, "message": f"Generated image for prompt: {prompt}"}
        
    except Exception as e:
        print(f"❌ Error generating image: {str(e)}")
        raise HTTPException(500, str(e))

@app.post("/generate/video")
async def generate_video(request: VideoRequest):
    try:
        prompt = validate_prompt(request.prompt)
        print(f"🎥 Generating video for prompt: {prompt}")
        
        if models["video"] is None:
            print("Loading video model...")
            max_retries = 3
            for attempt in range(max_retries):
                try:
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
                    print(f"⚠️ Video model load attempt {attempt + 1} failed, retrying...")
                    time.sleep(5)
        
        output = models["video"]({"text": prompt, "num_inference_steps": request.steps})
        frame = output["videos"][0][0]
        ret, buffer = cv2.imencode('.png', frame)
        video_frame_b64 = base64.b64encode(buffer).decode()
        
        return {"video_frame_b64": video_frame_b64, "message": f"Generated video frame for prompt: {prompt}"}
        
    except Exception as e:
        print(f"❌ Error generating video: {str(e)}")
        raise HTTPException(500, str(e))

@app.post("/generate/audio")
async def generate_audio_endpoint(request: AudioRequest):
    try:
        text = validate_prompt(request.text)
        print(f"🔊 Generating audio for text: {text}")
        
        if models["audio"] is None:
            print("Loading audio models...")
            preload_models()
            print("✅ Audio models preloaded")
            models["audio"] = True
        
        audio_array = generate_audio(text)
        buffer = io.BytesIO()
        write_wav(buffer, SAMPLE_RATE, audio_array)
        audio_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {"audio_b64": audio_b64, "message": f"Generated audio for text: {text}"}
        
    except Exception as e:
        print(f"❌ Error generating audio: {str(e)}")
        raise HTTPException(500, str(e))

@app.get("/health")
async def health():
    model_status = {
        "image": "loaded" if models["image"] is not None else "not loaded",
        "video": "loaded" if models["video"] is not None else "not loaded", 
        "audio": "loaded" if models["audio"] is not None else "not loaded"
    }
    
    return {
        "status": "🚀 AI API SDXL LIVE", 
        "device": device,
        "models": model_status,
        "services": ["image", "video", "audio"]
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI API SDXL Server...")
    print(f"📊 Device: {device}")
    print(f"🔌 HF Token: {'Set' if os.getenv('HF_TOKEN') else 'Not Set'}")
    uvicorn.run(app, host="0.0.0.0", port=8080)