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

# Install Python dependencies
COPY builder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
