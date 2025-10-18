#!/bin/bash
echo "🚀 Building Unified AI Suite..."

# Build
docker build -f Dockerfile.unified -t dorfnew/unified-ai-suite:latest .

# Push
docker push dorfnew/unified-ai-suite:latest

echo "✅ COMPLETE! Run: docker run -p 8000:8000 dorfnew/unified-ai-suite:latest"