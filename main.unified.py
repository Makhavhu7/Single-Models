# main.gpu.py -> put into container as /app/main.py (rename to main.py)
import os
import io
import time
import base64
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import cv2
import torch
from scipy.io.wavfile import write as write_wav

# Third-party model libs (diffusers, modelscope, bark)
from diffusers import DiffusionPipeline
from modelscope.pipelines import pipeline as modelscope_pipeline
from bark import SAMPLE_RATE, generate_audio, preload_models

app = FastAPI(title="Unified AI Suite (GPU)")

# device selection
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# lazy loaded model handles
models = {"image": None, "video": None, "audio_preloaded": False}

@app.on_event("startup")
async def startup_load_audio():
    # Preload Bark resources (fast)
    preload_models()
    models["audio_preloaded"] = True
    print("✅ Audio preloaded")

@app.post("/generate/image")
async def generate_image(prompt: str, steps: int = 20, width: int = 1024, height: int = 1024):
    """
    GPU-ready image endpoint. Uses FP16 when on GPU.
    """
    try:
        if models["image"] is None:
            dtype = torch.float16 if device == "cuda" else torch.float32
            print("Loading image model with dtype", dtype)
            models["image"] = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-3.5-large",
                torch_dtype=dtype,
                safety_checker=None
            ).to(device)
            # If GPU, enable attention slicing to reduce VRAM use
            try:
                models["image"].enable_attention_slicing()
            except Exception:
                pass
            print("✅ Image model loaded")

        pipe = models["image"]
        result = pipe(prompt, num_inference_steps=steps, width=width, height=height)
        image = result.images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return {"image_b64": base64.b64encode(buf.getvalue()).decode()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")

@app.post("/generate/video")
async def generate_video(prompt: str, steps: int = 25):
    """
    Text-to-video using modelscope pipeline. Expects GPU and BF16 support.
    Returns a single video frame (as PNG base64) for quick demo.
    """
    try:
        if models["video"] is None:
            # modelscope expects a CUDA environment for large models
            print("Loading video model (modelscope)...")
            models["video"] = modelscope_pipeline(
                "text-to-video-synthesis",
                model="Wan-AI/Wan2.2-TI2V-5B",
                model_revision="bf16",
                cache_dir="/app/model_cache",
                device=device  # modelscope tries to use torch device
            )
            print("✅ Video model loaded")

        output = models["video"]({"text": prompt, "num_inference_steps": steps})
        # Many models return a dict with 'videos' -> list of frames arrays
        frame = output["videos"][0][0]  # take first frame
        ret, buf = cv2.imencode('.png', frame)
        return {"video_frame_b64": base64.b64encode(buf.tobytes()).decode()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video generation failed: {e}")

@app.post("/generate/audio")
async def generate_audio_endpoint(text: str):
    """
    Bark audio generation (CPU/GPU both work). Returns WAV base64.
    """
    try:
        # generate_audio from bark returns numpy array int16 or float32 depending on implementation
        audio_array = generate_audio(text)
        buf = io.BytesIO()
        write_wav(buf, SAMPLE_RATE, audio_array)
        return {"audio_b64": base64.b64encode(buf.getvalue()).decode()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {e}")

@app.get("/health")
async def health():
    return {"status": "Unified AI Suite (GPU) LIVE", "device": device, "services": ["image", "video", "audio"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
