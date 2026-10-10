# SPDX-License-Identifier: Apache-2.0
"""Where this port's six weight files come from.

The port runs the ComfyUI repackaged single files, the same ones the reference GPU graph runs,
so a result compares one to one against it. All six live in one Apache-2.0 repo; `weights:` in
the container manifest names that repo and this module resolves the individual files, so the
package works both inside the image (the files are already in the HF cache) and against a
ComfyUI checkout on a workstation (COMFY_MODELS pointing at models/).

Order per file: an explicit WAN_<NAME> path, then the file under $COMFY_MODELS in ComfyUI's
own layout, then a download from the Hub into the HF cache.
"""
from __future__ import annotations

import os

HF_REPO = "Wan-AI/Wan2.2-I2V-A14B"  # the model these files repackage
COMFY_REPO = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
LORA_REPO = "lightx2v/Wan2.2-Lightning"
LICENSE = "apache-2.0"

# key -> (hub repo, path in that repo, path under a ComfyUI models/ directory)
COMFY_FILES = {
    "high": (COMFY_REPO,
             "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
             "diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"),
    "low": (COMFY_REPO,
            "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
            "diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"),
    "high_lora": (COMFY_REPO,
                  "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
                  "loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"),
    "low_lora": (COMFY_REPO,
                 "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
                 "loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"),
    "vae": (COMFY_REPO,
            "split_files/vae/wan_2.1_vae.safetensors",
            "vae/wan_2.1_vae.safetensors"),
    "te": (COMFY_REPO,
           "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
           "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
}


def comfy_models_dir() -> str:
    return os.environ.get("COMFY_MODELS", "/comfy")


def path(key: str, download: bool = True) -> str:
    repo, remote, local = COMFY_FILES[key]
    override = os.environ.get(f"WAN_{key.upper()}")
    if override:
        return override
    candidate = os.path.join(comfy_models_dir(), local)
    if os.path.exists(candidate) or not download:
        return candidate
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=remote)


def paths(download: bool = True) -> dict:
    return {key: path(key, download) for key in COMFY_FILES}
