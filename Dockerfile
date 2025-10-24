# Base image with NVIDIA CUDA support
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set non-interactive frontend to avoid keyboard layout prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
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

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA 12.1
RUN pip install --no-cache-dir \
    torch==2.1.2+cu121 \
    torchvision==0.16.2+cu121 \
    torchaudio==2.1.2+cu121 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Set environment variables
ENV HF_HOME=/app/model_cache
ENV HUGGINGFACE_HUB_CACHE=/app/model_cache
ENV MODELSCOPE_CACHE=/app/model_cache

# Create model cache directory
RUN mkdir -p /app/model_cache

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "src/main.py"]