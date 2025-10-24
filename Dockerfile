# Use NVIDIA CUDA 12.1 base image (matches RTX 4090)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install PyTorch and related packages with CUDA 12.1
RUN pip3 install --no-cache-dir \
    torch==2.1.2+cu121 \
    torchvision==0.16.2+cu121 \
    torchaudio==2.1.2+cu121 \
    -f https://download.pytorch.org/whl/cu121

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    numpy<2.0 \
    Pillow==10.4.0 \
    opencv-python==4.8.1.78 \
    diffusers==0.27.2 \
    transformers==4.44.0 \
    huggingface_hub==0.24.7 \
    accelerate==0.33.0 \
    safetensors==0.4.5 \
    modelscope==1.11.0 \
    scipy==1.10.1 \
    librosa==0.10.1 \
    soundfile==0.12.1 \
    runpod~=1.7.0

# Install bark from GitHub
RUN pip3 install --no-cache-dir git+https://github.com/suno-ai/bark.git

# Copy your application code (adjust as needed)
COPY . .

# Expose port for FastAPI/uvicorn (RunPod default)
EXPOSE 8000

# Command to run your application (modify based on your app)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]