# Base image
FROM ubuntu:22.04

# Copy apt sources
COPY sources.list /etc/apt/sources.list

# Timezone + core setup
RUN echo 'tzdata tzdata/Areas select Etc' | debconf-set-selections && \
    echo 'tzdata tzdata/Zones/Etc select UTC' | debconf-set-selections && \
    DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y python3-pip git libsndfile1 ffmpeg wget curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python deps
WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# Copy code
COPY app/ app/
COPY main.py main.py

# Model cache
RUN mkdir -p /app/model_cache && chmod 777 /app/model_cache

ENV PYTHONUNBUFFERED=1 \
    HUGGINGFACE_HUB_CACHE=/app/model_cache \
    TRANSFORMERS_CACHE=/app/model_cache \
    MODELSCOPE_CACHE=/app/model_cache

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
