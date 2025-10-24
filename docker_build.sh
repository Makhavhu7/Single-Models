#!/bin/bash
echo "🚀 Building AI API SDXL..."

# Build
docker build -t dorfnew/ai-api-sdxl:dev .

# Push
docker push dorfnew/ai-api-sdxl:dev

echo "✅ COMPLETE! Run: docker run -p 8080:8080 -e HF_TOKEN=$HF_TOKEN dorfnew/ai-api-sdxl:dev"