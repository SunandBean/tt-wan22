#!/usr/bin/env bash
# Render one P100a video job: ./run.sh <job dir> <images dir> [container name]
# Runs render.py in the tt-inference-server image (tt-metal with tt_dit's Wan VAE). The container's memory
# is capped: the renderer needs no more, and the host it ran on shared its RAM. Converted weights are cached
# in ~/.cache/wan-p100a (9.4 GB per expert, built on the first run), and so are the compiled kernels
# (TT_METAL_CACHE): without that every container compiles them again (~75 s on the first segment).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
JOB="$(cd "$1" && pwd)"
IMAGES="$(cd "$2" && pwd)"
NAME="${3:-wan22-p100a}"
IMAGE="${WAN_P100A_IMAGE:-ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.18.0-c49bb76-6b4a3a7}"
MODELS="${COMFY_MODELS:-$HOME/ComfyUI/models}"
mkdir -p "$HOME/.cache/wan-p100a"

RUN=(docker run --rm --name "$NAME" --ipc host --device /dev/tenstorrent
  --memory "${WAN_P100A_MEMORY:-24g}" --memory-swap "${WAN_P100A_MEMORY:-24g}"
  --user "$(id -u):$(id -g)"
  -v /dev/hugepages-1G:/dev/hugepages-1G -v "$MODELS:/comfy:ro" -v "$(cd "$HERE/.." && pwd):/pkg:ro"
  -v "$JOB:/job" -v "$IMAGES:/images:ro" -v "$HOME/.cache/wan-p100a:/cache"
  -e PYTHONPATH=/pkg -e WAN_CACHE=/cache -e TT_METAL_CACHE=/cache -e COMFY_MODELS=/comfy -e JOB_DIR=/job -e ARCH_NAME=blackhole -e TT_METAL_VISIBLE_DEVICES=0
  -e THREADS="${WAN_P100A_THREADS:-12}"
  --entrypoint /home/container_app_user/tt-metal/python_env/bin/python "$IMAGE" -m tt_wan22.render)

exec "${RUN[@]}"
