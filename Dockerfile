FROM ubuntu:22.04

# Copy APT sources
COPY sources.list /etc/apt/sources.list

# Install base dependencies
RUN echo 'tzdata tzdata/Areas select Etc' | debconf-set-selections && \
    echo 'tzdata tzdata/Zones/Etc select UTC' | debconf-set-selections && \
    DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip git libsndfile1 ffmpeg build-essential rustc cargo \
    nvidia-driver-535 nvidia-utils-535 && \
    rm -rf /var/lib/apt/lists/*

# Create appuser
RUN useradd -m appuser

# Install PyTorch with CUDA 12.1
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

WORKDIR /app

# Install dependencies
COPY builder/requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/ src/
COPY src/main.py .

# Create model cache
RUN mkdir -p /app/model_cache && chmod 777 /app/model_cache

# Cleanup
RUN apt-get purge -y build-essential rustc cargo && apt-get autoremove -y

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/model_cache \
    HUGGINGFACE_HUB_CACHE=/app/model_cache \
    MODELSCOPE_CACHE=/app/model_cache \
    HF_TOKEN=${HF_TOKEN} \
    PORT=8080

# Switch to appuser
USER appuser

CMD ["python3", "-u", "main.py"]
