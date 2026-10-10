#!/usr/bin/env bash
# Run one of these scripts (SCRIPT, default segment_tt.py) on the P100a, in the tt-inference-server image
# (its tt-metal python_env has tt_dit's Wan VAE, diffusers and the conv3d kernels). The container's memory is
# capped: on the host this was developed on, most RAM was a ComfyUI model cache, and an uncapped run once
# pushed the host into a global OOM that killed ComfyUI instead of this container.
#   STAGES=blk2 SCRIPT=wan_check.py ./run_tt.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${TT_IMAGE:-ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.18.0-c49bb76-6b4a3a7}"
SCRIPT="${SCRIPT:-segment_tt.py}"
mkdir -p "$HOME/.cache/wan-p100a"
exec docker run --rm --name wan-dev --ipc host --device /dev/tenstorrent \
    --memory "${MEMORY:-24g}" --memory-swap "${MEMORY:-24g}" \
    -v /dev/hugepages-1G:/dev/hugepages-1G -v "$HOME/ComfyUI/models:/comfy:ro" -v "$HERE:/work" \
    -v "$HOME/.cache/wan-p100a:/cache" -e WAN_CACHE=/cache -e COMFY_MODELS=/comfy \
    -e ARCH_NAME=blackhole -e TT_METAL_VISIBLE_DEVICES=0 \
    -e STAGES -e REPEAT -e ENC_FRAMES -e ENC_CHUNK -e T_CHUNK -e L1_SMALL -e THREADS -e FFN_BFP4 -e SEED \
    -e EXPERT -e WAN_SIZES -e NB \
    --workdir /work --entrypoint /home/container_app_user/tt-metal/python_env/bin/python "$IMAGE" "/work/$SCRIPT"
