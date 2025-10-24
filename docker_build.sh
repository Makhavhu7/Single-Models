#!/bin/bash
echo "🚀 Building Unified AI Suite..."

# Build
docker build -t dorfnew/unified-ai-suite:dev .

# Push
docker push dorfnew/unified-ai-suite:dev

echo "✅ COMPLETE! Run: docker run -p 8080:8080 -e HF_TOKEN=$HF_TOKEN dorfnew/unified-ai-suite:dev"
