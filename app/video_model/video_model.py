from modelscope.pipelines import pipeline
import cv2, base64, torch, time
import numpy as np

pipe = None

async def generate_video(req):
    global pipe
    if pipe is None:
        print("Loading video model...")
        for attempt in range(3):
            try:
                pipe = pipeline(
                    "text-to-video-synthesis",
                    model="Wan-AI/Wan2.2-TI2V-5B",
                    model_revision="bf16",
                    cache_dir="/app/model_cache"
                )
                print("✅ Video model loaded!")
                break
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
                time.sleep(10)
    output = pipe({"text": req.prompt, "num_inference_steps": req.num_inference_steps})
    frame = output["videos"][0][0]
    _, buffer = cv2.imencode('.png', frame)
    return {"video_frame_b64": base64.b64encode(buffer).decode()}
