#!/bin/bash
echo "🚀 Building Unified AI Suite..."

# Build
docker build -f Dockerfile.unified -t dorfnew/unified-ai-suite:latest .

# Push
docker push dorfnew/unified-ai-suite:latest

echo "✅ COMPLETE! Run: docker run -p 8080:8080 dorfnew/unified-ai-suite:latest"