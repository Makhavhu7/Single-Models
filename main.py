from fastapi import FastAPI, HTTPException
from app.video_model import generate_video
from app.image_model import generate_image
from app.audio_model import generate_audio
from pydantic import BaseModel

app = FastAPI(title="Unified AI Service", version="1.0")

# Shared request schema
class GenerateRequest(BaseModel):
    prompt: str = None
    text: str = None
    num_inference_steps: int = 25
    width: int = 1024
    height: int = 1024

@app.post("/generate/video")
async def video(req: GenerateRequest):
    try:
        return await generate_video(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/image")
async def image(req: GenerateRequest):
    try:
        return await generate_image(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/audio")
async def audio(req: GenerateRequest):
    try:
        return await generate_audio(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"status": "ok", "message": "Unified Model API online"}
