# SPDX-License-Identifier: Apache-2.0
"""Render a video's Wan 2.2 I2V segments on the P100a (run in the tt-inference-server image by run.sh).

The worker writes <job>/job.json and reads back, per segment i: i.frames.rgb (raw uint8 RGB frames), i_last.png
and i.json (frame count / size and timings); <job>/progress.json follows the run and <job>/result.json ends it. Same graph as
workflows.wan_segment on the GPU: UMT5 prompt, WanImageToVideo / WanFirstLastFrameToVideo conditioning, high
noise expert for steps 0-1 and low noise expert for 2-3 (lightx2v LoRAs, shift 5, simple, Euler, cfg 1). Each
segment after the first starts from the previous segment's last frame, as on the GPU.

job.json: {"gen_w", "gen_h", "fps", "segments": [{"prompt", "length", "seed", "start" (path or null = previous
last frame), "end" (path or null)}]}. Paths are inside the container (/job, /images)."""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
import ttnn  # noqa: E402
from . import vae as V
from . import wan_dit as W
from . import config
from .comfy_weights import wan_expert
from .umt5 import UMT5Encoder

JOB = Path(os.environ.get("JOB_DIR", "/job"))


def write(name, data):
    tmp = JOB / (name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, JOB / name)


def lanczos_center(path, w, h):
    """ImageScale(lanczos, crop=center) on a PNG -> [H, W, 3] float in [0, 1]."""
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    ow, oh = img.size
    oa, na = ow / oh, w / h
    x = y = 0
    if oa > na:
        x = round((ow - ow * (na / oa)) / 2)
    elif oa < na:
        y = round((oh - oh * (oa / na)) / 2)
    img = img.crop((x, y, ow - x, oh - y))
    if img.size != (w, h):
        img = img.resize((w, h), resample=Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)


def conditioning(vae, start, end, w, h, length):
    """WanImageToVideo / WanFirstLastFrameToVideo + WAN21.concat_cond -> [1, 20, T, h, w] (mask 4 + latent 16)."""
    frames = torch.full((length, h, w, 3), 0.5)
    frames[0] = start
    if end is not None:
        frames[-1] = end
    lat = vae.encode(frames.permute(3, 0, 1, 2)[None] * 2 - 1)
    T = lat.shape[2]
    mask = torch.zeros(1, 4, T, lat.shape[-2], lat.shape[-1])
    mask[:, :, 0] = 1.0  # the latent frame holding the start image (4 frame slots)
    if end is not None:
        mask[:, 3, -1] = 1.0  # the end image is the last frame slot of the last latent frame
    return torch.cat([mask, (lat - V.MEAN) / V.STD], dim=1)


# Above this many video tokens (480p x 81 frames is 32,760; 720p is 75,600) only one expert stays on the card at a
# time: the VAE and the attention of a 720p segment need the room both experts would take. The weight cache
# brings an expert back in ~12 s, so swapping costs ~25 s a segment.
SWAP_TOKENS = 40_000


class Experts:
    """Both Wan experts on the card (resident), or one at a time (swap), loaded from the weight cache."""

    def __init__(self, dev, swap, progress):
        import ttnn as _t
        self.dev, self.swap, self.progress = dev, swap, progress
        self.cache = os.environ.get("WAN_CACHE") or None
        self.prec = W.Prec(ffn_dtype=_t.bfloat4_b, host_text_kv=True)
        self.ckpt = {e: wan_expert(e) for e in ("high", "low")}
        self.tc = {e: W.TimeConditioning(self.ckpt[e]) for e in self.ckpt}
        self.models = {}
        self.swaps, self.swap_s = 0, 0.0
        for e in (("high",) if swap else ("high", "low")):
            self._load(e)

    def _load(self, e):
        self.progress(detail=f"expert {e}")
        self.models[e] = W.TTWanExpert(self.dev, self.ckpt[e], prec=self.prec, cache_dir=self.cache)

    def get(self, e):
        """The expert on the card, swapping the other one out first in swap mode."""
        if e not in self.models:
            t0 = time.time()
            if self.swap:
                for other in list(self.models):
                    self.models.pop(other).release()
            self._load(e)
            self.swaps += 1
            self.swap_s += time.time() - t0
        return self.models[e]


class Renderer:
    def __init__(self, dev, progress, swap=False):
        t0 = time.time()
        progress(state="loading")
        self.experts = Experts(dev, swap, progress)
        self.dev = dev
        progress(state="loading", detail="vae")
        self.vae = V.WanVAE(dev, config.path("vae"))
        self.text = UMT5Encoder(config.path("te"),
                                threads=int(os.environ.get("THREADS", "12")))
        self.kv_prompt, self.kv = None, None
        self.load_s = round(time.time() - t0, 1)

    def text_kv(self, prompt):
        """Cross-attention K/V of both experts for this prompt, on the card (kept while the prompt repeats)."""
        if prompt != self.kv_prompt:
            if self.kv is not None:
                for pairs in self.kv.values():
                    for k, v in pairs:
                        ttnn.deallocate(k)
                        ttnn.deallocate(v)
            context = self.text(prompt)
            up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                           device=self.dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self.kv = {e: [(up(k), up(v)) for k, v in W.host_text_kv_heads(ck, context)]
                       for e, ck in self.experts.ckpt.items()}
            self.kv_prompt = prompt
        return self.kv

    def segment(self, prompt, start, end, w, h, length, seed, progress):
        timing = {}
        t0 = time.time()
        progress(phase="text")
        kv = self.text_kv(prompt)
        timing["text_s"] = round(time.time() - t0, 1)
        t0 = time.time()
        progress(phase="encode")
        concat = conditioning(self.vae, start, end, w, h, length)
        timing["encode_s"] = round(time.time() - t0, 1)
        _, _, T, lh, lw = concat.shape
        gh, gw = lh // 2, lw // 2
        n = T * gh * gw
        S = W.round_up(n, 256)
        cos, sin = W.cos_sin(W.grid_ids(T, gh, gw), S)
        tab = lambda v: ttnn.from_torch(v.reshape(1, 1, S, -1).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                        layout=ttnn.TILE_LAYOUT, device=self.dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cos_t, sin_t = tab(cos), tab(sin)
        sig = W.sigmas_sd3()
        g = torch.Generator("cpu").manual_seed(int(seed))
        x = torch.randn((1, 16, T, lh, lw), generator=g, dtype=torch.float32) * sig[0]
        t0 = time.time()
        for i in range(4):
            progress(phase="sample", step=i + 1)
            e = "high" if i < 2 else "low"
            m, tc = self.experts.get(e), self.experts.tc[e]
            blocks, head = tc.rows(sig[i] * 1000.0)
            rows, head_rows = m.rows_to_dev(blocks, head)
            p = W.patchify(torch.cat([x.to(torch.bfloat16).float(), concat], dim=1))
            p = torch.cat([p, torch.zeros(S - n, p.shape[1])])
            v = W.unpatchify(m.forward(p, rows, head_rows, kv[e], cos_t, sin_t, n), T, gh, gw)
            for r in rows:
                for t in r.values():
                    ttnn.deallocate(t)
            for t in head_rows.values():
                ttnn.deallocate(t)
            s = torch.tensor(sig[i])
            x = x + (x - (x - v * s)) / s * (sig[i + 1] - sig[i])  # Euler on the CONST denoised
        ttnn.deallocate(cos_t)
        ttnn.deallocate(sin_t)
        timing["sample_s"] = round(time.time() - t0, 1)
        t0 = time.time()
        progress(phase="decode")
        y = self.vae.decode(x * V.STD + V.MEAN)
        timing["decode_s"] = round(time.time() - t0, 1)
        frames = ((y[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) / 2 * 255).numpy().astype(np.uint8)  # SaveImage rounding
        return frames, timing


def main():
    job = json.loads((JOB / "job.json").read_text())
    state = {"state": "starting", "started": time.time()}

    def progress(**kw):
        state.update(kw, updated=time.time())
        write("progress.json", state)

    progress()
    dev = V.open_dev()
    try:
        w, h = job["gen_w"], job["gen_h"]
        tokens = ((max(s["length"] for s in job["segments"]) - 1) // 4 + 1) * (h // 16) * (w // 16)
        r = Renderer(dev, progress, swap=tokens > SWAP_TOKENS)
        progress(state="rendering", load_s=r.load_s, swap=r.experts.swap)
        last = None
        for i, seg in enumerate(job["segments"]):
            from PIL import Image
            t0 = time.time()
            progress(segment=i, phase="start", step=0)
            start = lanczos_center(seg["start"], w, h) if seg.get("start") else last
            end = lanczos_center(seg["end"], w, h) if seg.get("end") else None
            frames, timing = r.segment(seg["prompt"], start, end, w, h, seg["length"], seg["seed"], progress)
            (JOB / f"{i}.frames.tmp").write_bytes(np.ascontiguousarray(frames).tobytes())
            os.replace(JOB / f"{i}.frames.tmp", JOB / f"{i}.frames.rgb")
            Image.fromarray(frames[-1]).save(JOB / f"{i}_last.png")
            last = torch.from_numpy(frames[-1].astype(np.float32) / 255.0)  # what the GPU path reloads from PNG
            timing["segment_s"] = round(time.time() - t0, 1)
            timing["expert_swaps"], timing["swap_s"] = r.experts.swaps, round(r.experts.swap_s, 1)
            timing["frames"], timing["height"], timing["width"] = (int(v) for v in frames.shape[:3])
            write(f"{i}.json", timing)
            progress(done=i + 1)
        write("result.json", {"status": "ok", "load_s": r.load_s, "segments": len(job["segments"])})
        progress(state="done")
    except Exception as exc:
        write("result.json", {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:2000],
                              "trace": traceback.format_exc()[-4000:]})
        progress(state="error")
        raise
    finally:
        V.close_dev(dev)


if __name__ == "__main__":
    main()
