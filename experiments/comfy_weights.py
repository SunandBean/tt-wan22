# SPDX-License-Identifier: Apache-2.0
"""Lazy reader for ComfyUI single-file checkpoints: dequantizes fp8 weights and merges a LoRA.

Two fp8 layouts occur: "comfy_quant" (<m>.weight float8_e4m3fn with a scalar <m>.weight_scale) and "scaled_fp8"
(<m>.weight with <m>.scale_weight, the Wan 2.2 files). Everything else is read as stored. The LoRA
(lora_up [out, r], lora_down [r, in], alpha) is merged like ComfyUI's LoRA patch, W += strength * alpha / r *
up @ down, in float32; lightx2v's Wan LoRAs prefix their keys with "diffusion_model.".
(Same reader as the sibling Qwen-Image-Edit port, github.com/SunandBean/tt-qwen-image-edit.)"""
from __future__ import annotations

from typing import Optional

import torch
from safetensors import safe_open


class ComfyCheckpoint:
    def __init__(self, path: str, lora_path: Optional[str] = None, lora_strength: float = 1.0, lora_prefix: str = ""):
        self.path, self.lora_path = path, lora_path
        self._f = safe_open(path, framework="pt", device="cpu")
        self._keys = set(self._f.keys())
        self._lora = safe_open(lora_path, framework="pt", device="cpu") if lora_path else None
        self._lora_keys = set(self._lora.keys()) if self._lora is not None else set()
        self.lora_strength = float(lora_strength)
        self.lora_prefix = lora_prefix

    def keys(self):
        return self._keys

    def _scale(self, module: str) -> Optional[torch.Tensor]:
        for suffix in ("weight_scale", "scale_weight"):
            k = f"{module}.{suffix}"
            if k in self._keys:
                return self._f.get_tensor(k).float()
        return None

    def get(self, key: str, dtype: Optional[torch.dtype] = torch.bfloat16) -> torch.Tensor:
        """Tensor `key` dequantized (and LoRA-merged for a module weight), cast to dtype."""
        t = self._f.get_tensor(key)
        merged = False
        if key.endswith(".weight"):
            module = key[: -len(".weight")]
            scale = self._scale(module) if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else None
            if scale is not None:
                t = t.float() * scale
            delta = self.lora_delta(module)
            if delta is not None:
                t = t.float() + delta
                merged = True
        if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            t = t.float()
        if dtype is not None and t.dtype != dtype:
            return t.to(dtype)
        return t if merged else t.clone()

    def lora_delta(self, module: str) -> Optional[torch.Tensor]:
        if self._lora is None or self.lora_strength == 0.0:
            return None
        module = self.lora_prefix + module
        up_k, down_k = f"{module}.lora_up.weight", f"{module}.lora_down.weight"
        if up_k not in self._lora_keys:
            return None
        up = self._lora.get_tensor(up_k).float()
        down = self._lora.get_tensor(down_k).float()
        rank = down.shape[0]
        alpha_k = f"{module}.alpha"
        alpha = float(self._lora.get_tensor(alpha_k).float()) if alpha_k in self._lora_keys else float(rank)
        return (self.lora_strength * alpha / rank) * (up @ down)


def wan_expert(models_dir: str, expert: str) -> ComfyCheckpoint:
    """The Wan 2.2 I2V A14B expert ("high" | "low") with its lightx2v 4-step LoRA."""
    return ComfyCheckpoint(f"{models_dir}/diffusion_models/wan2.2_i2v_{expert}_noise_14B_fp8_scaled.safetensors",
                           f"{models_dir}/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_{expert}_noise.safetensors", 1.0,
                           lora_prefix="diffusion_model.")
