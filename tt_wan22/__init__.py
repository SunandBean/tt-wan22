# SPDX-License-Identifier: Apache-2.0
"""Wan 2.2 I2V A14B with the lightx2v 4-step LoRA on one Tenstorrent Blackhole p100a.

Both experts (high-noise for steps 0-1, low-noise for 2-3), the Wan 2.1 VAE encoder and
decoder all run on the card. The UMT5 text encoder runs on the host.

The reference is the ComfyUI `wan_segment` graph: shift 5, simple, Euler, cfg 1, 4 steps.
"""
from .render import Experts, Renderer
from .vae import close_dev, open_dev

__all__ = ["Renderer", "Experts", "open_dev", "close_dev"]
__version__ = "0.1.0"
