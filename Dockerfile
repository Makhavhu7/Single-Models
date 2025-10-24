# Base image with NVIDIA CUDA support
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
COPY sources.list /etc/apt/sources.list
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    git \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
RUN python3.11 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install Python dependencies from official PyTorch source
COPY builder/requirements.txt .
RUN pip install --no-cache-dir \
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
    git+https://github.com/suno-ai/bark.git \
    runpod~=1.7.0 \
    && pip install --no-cache-dir torch==2.5.0+cu121 torchvision==0.20.0+cu121 torchaudio==2.5.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html

# Copy application code
COPY src/ ./src/
COPY tests/ ./tests/
COPY config/ ./config/

# Install NVIDIA drivers and utilities
RUN apt-get update && apt-get install -y \
    nvidia-driver-535 \
    nvidia-utils-535 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN
ENV HF_HOME=/app/model_cache
ENV HUGGINGFACE_HUB_CACHE=/app/model_cache
ENV MODELSCOPE_CACHE=/app/model_cache

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "src/main.py"]