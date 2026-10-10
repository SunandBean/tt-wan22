"""Wan 2.2 I2V A14B feasibility on one P100a (SCRIPT=wan_check.py ./run_tt.sh).
STAGES: blk2 (2 blocks vs CPU reference, small video), full (40 blocks: load time, DRAM, step time at 480p x 81
frames and smaller). Writes wan-check/report.json."""
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import wan_tt as W  # noqa: E402
from tt_device import close_device, dram_stats, open_device  # noqa: E402
from comfy_weights import ComfyCheckpoint, wan_expert  # noqa: E402

OUT = ROOT / "wan-check"
OUT.mkdir(exist_ok=True)
report = {"stages": {}}
torch.set_num_threads(int(os.environ.get("THREADS", "16")))
M = os.environ.get("COMFY_MODELS", "/comfy")
DIT = f"{M}/diffusion_models/wan2.2_i2v_{os.environ.get('EXPERT', 'high')}_noise_14B_fp8_scaled.safetensors"
LORA = f"{M}/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_{os.environ.get('EXPERT', 'high')}_noise.safetensors"


def save():
    (OUT / "report.json").write_text(json.dumps(report, indent=2))


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def ckpt(expert=None):
    e = expert or os.environ.get("EXPERT", "high")
    return ComfyCheckpoint(f"{M}/diffusion_models/wan2.2_i2v_{e}_noise_14B_fp8_scaled.safetensors",
                           f"{M}/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_{e}_noise.safetensors", 1.0,
                           lora_prefix="diffusion_model.")


def prec():
    import ttnn
    return W.Prec(ffn_dtype=ttnn.bfloat4_b) if os.environ.get("FFN_BFP4") else None


def stage(fn):
    def run(dev):
        t0 = time.time()
        try:
            res = fn(dev)
            report["stages"][fn.__name__] = {"status": "ok", "s": round(time.time() - t0, 1), **res}
        except Exception:
            report["stages"][fn.__name__] = {"status": "error", "error": traceback.format_exc()}
        save()
        print(fn.__name__, json.dumps(report["stages"][fn.__name__])[:3000], flush=True)
    return run


def inputs(t, gh, gw, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, W.IN_DIM, t, gh * 2, gw * 2, generator=g)
    ctx = torch.randn(512, W.TEXT_DIM, generator=g) * 0.1
    return W.patchify(x), ctx


@stage
def blk2(dev):
    ck = ckpt()
    m = W.TTWanExpert(dev, ck, n_blocks=2, prec=prec())
    tc = W.TimeConditioning(ck)
    blocks, head = tc.rows(1000.0 * 0.9375)
    rows, head_rows = m.rows_to_dev(blocks[:2], head)
    t, gh, gw = 4, 16, 32  # S = 2048, no padding
    patches, ctx = inputs(t, gh, gw)
    S = patches.shape[0]
    cos, sin = W.cos_sin(W.grid_ids(t, gh, gw), S)
    tab = lambda v: m.to_dev(v.reshape(1, 1, S, -1))
    kv = m.text_kv(ctx)
    taps = []
    m.forward(patches, rows, head_rows, kv, tab(cos), tab(sin), S, taps=taps)
    # CPU reference from the same embeddings
    pw = ck.get("patch_embedding.weight", torch.float32).reshape(W.DIM, -1)
    x = patches.to(torch.bfloat16).float() @ pw.t() + ck.get("patch_embedding.bias", torch.float32)
    t0 = ck.get("text_embedding.0.weight", torch.float32), ck.get("text_embedding.0.bias", torch.float32)
    t2 = ck.get("text_embedding.2.weight", torch.float32), ck.get("text_embedding.2.bias", torch.float32)
    c = torch.nn.functional.linear(torch.nn.functional.gelu(torch.nn.functional.linear(ctx.to(torch.bfloat16).float(), *t0),
                                                            approximate="tanh"), *t2)
    angles = W.rope_angles(W.grid_ids(t, gh, gw)).float()
    out = {}
    for i in range(2):
        x = W.RefBlock(ck, i)(x, blocks[i], c, angles)
        out[f"block{i}_pcc"] = pcc(taps[i], x)
    return out


@stage
def full(dev):
    ck = ckpt()
    t0 = time.time()
    m = W.TTWanExpert(dev, ck)
    load_s = time.time() - t0
    tc = W.TimeConditioning(ck)
    blocks, head = tc.rows(1000.0 * 0.9375)
    rows, head_rows = m.rows_to_dev(blocks, head)
    out = {"load_s": round(load_s, 1), "dram_after_load": dram_stats(dev)}
    ctx = torch.randn(512, W.TEXT_DIM) * 0.1
    t1 = time.time()
    kv = m.text_kv(ctx)
    out["text_kv_s"] = round(time.time() - t1, 2)
    report["stages"]["full"] = {"status": "running", **out}
    save()
    sizes = [(s.split(":")[0], tuple(int(v) for v in s.split(":")[1].split("x")))
             for s in os.environ.get("WAN_SIZES", "480p33f:9x30x52,480p81f:21x30x52").split(",")]
    for name, (t, gh, gw) in sizes:
        n = t * gh * gw
        S = W.round_up(n, 256)
        patches, _ = inputs(t, gh, gw)
        patches = torch.cat([patches, torch.zeros(S - n, patches.shape[1])])
        cos, sin = W.cos_sin(W.grid_ids(t, gh, gw), S)
        tab = lambda v: m.to_dev(v.reshape(1, 1, S, -1))
        ct, st = tab(cos), tab(sin)
        times = []
        for _ in range(2):
            t1 = time.time()
            v = m.forward(patches, rows, head_rows, kv, ct, st, n)
            times.append(round(time.time() - t1, 2))
        out[name] = {"tokens": n, "padded": S, "step_s": times, "finite": bool(torch.isfinite(v).all()),
                     "dram": dram_stats(dev)}
        report["stages"]["full"] = {"status": "running", **out}
        save()
    return out


@stage
def both(dev):
    """High and low experts resident together (FFN bfloat4_b): DRAM and one 81-frame step on each."""
    import ttnn
    out = {}
    experts = {}
    for e in ("high", "low"):
        t0 = time.time()
        ck = ckpt(e)
        experts[e] = (W.TTWanExpert(dev, ck, prec=W.Prec(ffn_dtype=ttnn.bfloat4_b)), W.TimeConditioning(ck))
        out[f"{e}_load_s"] = round(time.time() - t0, 1)
        out[f"dram_after_{e}"] = dram_stats(dev)
        report["stages"]["both"] = {"status": "running", **out}
        save()
    ctx = torch.randn(512, W.TEXT_DIM) * 0.1
    t, gh, gw = 21, 30, 52
    n = t * gh * gw
    S = W.round_up(n, 256)
    patches, _ = inputs(t, gh, gw)
    patches = torch.cat([patches, torch.zeros(S - n, patches.shape[1])])
    cos, sin = W.cos_sin(W.grid_ids(t, gh, gw), S)
    for e, sigma in (("high", 0.9375), ("low", 0.6)):
        m, tc = experts[e]
        blocks, head = tc.rows(1000.0 * sigma)
        rows, head_rows = m.rows_to_dev(blocks, head)
        kv = m.text_kv(ctx)
        tab = lambda v: m.to_dev(v.reshape(1, 1, S, -1))
        times = []
        for _ in range(2):
            t1 = time.time()
            v = m.forward(patches, rows, head_rows, kv, tab(cos), tab(sin), n)
            times.append(round(time.time() - t1, 2))
        out[f"{e}_step_s"] = times
        out[f"{e}_finite"] = bool(torch.isfinite(v).all())
        out[f"dram_{e}_run"] = dram_stats(dev)
        report["stages"]["both"] = {"status": "running", **out}
        save()
    return out


def main():
    dev = open_device()
    try:
        for s in os.environ.get("STAGES", "blk2,full").split(","):
            globals()[s](dev)
            import ttnn
            ttnn.synchronize_device(dev)
            dev.clear_program_cache()
    finally:
        save()
        close_device(dev)


if __name__ == "__main__":
    main()
