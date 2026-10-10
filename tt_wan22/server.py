# SPDX-License-Identifier: Apache-2.0
"""HTTP server for Wan 2.2 I2V A14B on one Blackhole p100a.

    uvicorn tt_wan22.server:app --host 0.0.0.0 --port 20000

One segment at a time, one card, one process. A segment is an image plus a prompt in and a
short clip out, which is what the renderer has always produced -- `render.py` chains segments
for a whole video, and this server exposes one of them. Chaining is the caller's job: feed
`last_frame` of one response back as `start_image` of the next, exactly as `render.py` does.

Both experts live on the card at the sizes this port was validated at. Whether they stay
resident or swap is decided once at startup from WAN_MAX_* -- the renderer cannot change
that per request, so the server fixes the envelope rather than pretending otherwise.
"""
from __future__ import annotations

import base64
import io
import os
import time
from contextlib import asynccontextmanager
from threading import Lock
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .render import SWAP_TOKENS, Renderer, lanczos_center
from .vae import close_dev, open_dev

LICENSE = "apache-2.0"
MODEL_NAME = "Wan-AI/Wan2.2-I2V-A14B"
STEPS = 4  # the lightx2v 4-step LoRA; other step counts are not this port

# The validated envelope. The expert-residency decision is made once from these, so a request
# outside them is refused rather than silently served by a pipeline sized for something else.
MAX_W = int(os.environ.get("WAN_MAX_W", "832"))
MAX_H = int(os.environ.get("WAN_MAX_H", "480"))
MAX_FRAMES = int(os.environ.get("WAN_MAX_FRAMES", "81"))
MIN_DIM, DIM_STEP = 256, 16
TURN_WAIT_S = float(os.environ.get("WAN_TURN_WAIT_S", "1800"))

STATE: dict = {"status": "loading", "error": None, "generating": False, "load_s": None, "swap": None}
MODEL: Optional[Renderer] = None
DEVICE = None
LOCK = Lock()


def _tokens(w: int, h: int, frames: int) -> int:
    """The renderer's own token count for a segment -- what decides expert residency."""
    return ((frames - 1) // 4 + 1) * (h // 16) * (w // 16)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global MODEL, DEVICE
    try:
        DEVICE = open_dev()
        swap = _tokens(MAX_W, MAX_H, MAX_FRAMES) > SWAP_TOKENS
        MODEL = Renderer(DEVICE, lambda **kw: None, swap=swap)
        STATE.update(status="ok", load_s=MODEL.load_s, swap=MODEL.experts.swap)
    except Exception as exc:
        STATE.update(status="error", error=str(exc))
    try:
        yield
    finally:
        MODEL = None
        if DEVICE is not None:
            close_dev(DEVICE)


app = FastAPI(title="Wan 2.2 I2V A14B on p100a", lifespan=lifespan, docs_url=None, redoc_url=None)


class Request(BaseModel):
    model_config = ConfigDict(extra="ignore")
    prompt: str = Field(min_length=1, max_length=4000)
    start_image: str = Field(description="base64 PNG/JPEG; the first frame of the segment")
    end_image: Optional[str] = Field(default=None, description="base64; optional last frame")
    width: int = 832
    height: int = 480
    length: int = 81
    seed: int = Field(default=42, ge=0, le=2147483647)
    num_steps: Optional[int] = None
    # Frames come back as raw uint8 RGB, which is what this port was measured on -- re-encoding
    # them here would make the response no longer the thing the PSNR numbers describe. That is
    # ~97 MB for an 81-frame 832x480 segment before base64, so a caller that only needs to chain
    # segments (or only wants the timings) turns it off.
    return_frames: bool = True


def _check(request: Request) -> None:
    if request.num_steps is not None and request.num_steps != STEPS:
        raise ValueError(f"this port runs {STEPS} steps (the lightx2v 4-step LoRA)")
    for axis, value, cap in (("width", request.width, MAX_W), ("height", request.height, MAX_H)):
        if not MIN_DIM <= value <= cap:
            raise ValueError(f"{axis} must be between {MIN_DIM} and {cap}")
        if value % DIM_STEP:
            raise ValueError(f"{axis} must be a multiple of {DIM_STEP}")
    if not 1 <= request.length <= MAX_FRAMES:
        raise ValueError(f"length must be between 1 and {MAX_FRAMES}")
    if request.length % 4 != 1:
        raise ValueError("length must be 4n+1 (the VAE's temporal stride)")
    if _tokens(request.width, request.height, request.length) > _tokens(MAX_W, MAX_H, MAX_FRAMES):
        raise ValueError(
            f"segment is larger than the envelope this server was started with "
            f"({MAX_W}x{MAX_H}, {MAX_FRAMES} frames); restart it with WAN_MAX_W/H/FRAMES to widen"
        )


def _decode_image(b64: str, w: int, h: int) -> torch.Tensor:
    """base64 image -> [h, w, 3] float in [0, 1], cropped and scaled exactly as the renderer does."""
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception as exc:
        raise ValueError(f"image is not valid base64: {exc}") from exc
    return lanczos_center(io.BytesIO(raw), w, h)


def _png(frame: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/health")
def health():
    return dict(STATE)


@app.get("/info")
def info():
    return dict(STATE) | {
        "model": MODEL_NAME,
        "license": LICENSE,
        "device": "Tenstorrent Blackhole p100a",
        "num_steps": STEPS,
        "task_modes": ["image-to-video"],
        "envelope": {"max_width": MAX_W, "max_height": MAX_H, "max_frames": MAX_FRAMES,
                     "dim_step": DIM_STEP, "length_rule": "4n+1"},
        "chaining": "feed last_frame back as start_image to continue the video",
    }


@app.post("/predict")
def predict(request: Request):
    if not request.prompt.strip():
        raise HTTPException(400, "Empty prompt")
    try:
        _check(request)
        start = _decode_image(request.start_image, request.width, request.height)
        end = _decode_image(request.end_image, request.width, request.height) if request.end_image else None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if STATE["status"] != "ok" or MODEL is None:
        raise HTTPException(503, STATE["error"] or "Model is not ready")
    if not LOCK.acquire(timeout=TURN_WAIT_S):
        raise HTTPException(409, "Another segment is in progress")
    try:
        STATE["generating"] = True
        t0 = time.perf_counter()
        frames, timing = MODEL.segment(request.prompt, start, end, request.width, request.height,
                                       request.length, request.seed, lambda **kw: None)
        body = {
            "shape": list(int(v) for v in frames.shape),  # [frames, height, width, 3], uint8 RGB
            "dtype": "uint8",
            "last_frame": _png(frames[-1]),
            "model": MODEL_NAME,
            "license": LICENSE,
            "seed": request.seed,
            "num_steps": STEPS,
            "timing_s": timing | {"total_s": round(time.perf_counter() - t0, 2)},
        }
        if request.return_frames:
            body["frames_b64"] = base64.b64encode(np.ascontiguousarray(frames).tobytes()).decode()
        return body
    except Exception as exc:
        STATE.update(status="error", error=str(exc))
        raise HTTPException(503, str(exc)) from exc
    finally:
        STATE["generating"] = False
        LOCK.release()
