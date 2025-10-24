#!/bin/bash
# Manually deploy to RunPod pod after getting POD_ID
# Set POD_ID and RUNPOD_API_KEY as environment variables before running

if [ -z "$POD_ID" ] || [ -z "$RUNPOD_API_KEY" ]; then
  echo "Error: POD_ID or RUNPOD_API_KEY not set. Export them first, e.g.:"
  echo "export POD_ID=your-pod-id"
  echo "export RUNPOD_API_KEY=your-api-key"
  exit 1
fi

if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN not set. Export it first, e.g.: export HF_TOKEN=your-hf-token"
  exit 1
fi

curl -X POST "https://api.runpod.io/pod/${POD_ID}/start" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "containerDiskInGb": 80,
    "ports": "8080/http",
    "imageName": "dorfnew/ai-api-sdxl:dev",
    "env": [{"key": "HF_TOKEN", "value": "'"$HF_TOKEN"'"}]
  }'

echo "Pod deployment initiated. Check RunPod dashboard for status."
