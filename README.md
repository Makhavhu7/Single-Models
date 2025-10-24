# AI API SDXL

A secure cloud deployment for Stable Diffusion XL (SDXL) image generation, with unified support for video and audio, running on RunPod pods with RTX 4090.

## Setup

1. Set GitHub Secrets: `RUNPOD_API_KEY`, `HF_TOKEN`, `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`.
2. Set GitHub Variables: `POD_ID` (from RunPod console).
3. Push to `main` to trigger CI/CD.
4. Deploy: Run `./deploy/deploy_pod.sh` or trigger `deploy_pod.yml`.

## Hardware
- 1x RTX 4090 (24GB VRAM)
- 36GB RAM, 6 vCPUs, 80GB disk
- Costs: $0.59/hr GPU, $0.011/hr running disk, $0.014/hr stopped disk

## Endpoints
- `POST /generate/image`: SDXL image generation (use test config for params).
- `POST /generate/video`: Video generation.
- `POST /generate/audio`: Audio generation.
- `GET /health`: Status check.

## Testing
Use `tests/test_config.json` for SDXL tests.