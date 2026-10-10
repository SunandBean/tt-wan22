"""Where a Wan DiT step spends its time on the P100a: 2 blocks of the high expert at 480p x 81 frames, every
ttnn op wrapped with a device sync and timed (per op name and per matmul shape). The synced total is a little
above the real step; the shares are what matter. Run with SCRIPT=profile_step.py ./run_tt.sh."""
import collections
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import ttnn  # noqa: E402
import wan_tt as W  # noqa: E402
from comfy_weights import ComfyCheckpoint, wan_expert  # noqa: E402

M = os.environ.get("COMFY_MODELS", "/comfy")
NB = int(os.environ.get("NB", "2"))
stats = collections.defaultdict(lambda: [0, 0.0])
dev = None


def wrap(owner, name, label=None):
    fn = getattr(owner, name)

    def timed(*a, **k):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn(*a, **k)
        ttnn.synchronize_device(dev)
        key = label or name
        if name == "minimal_matmul":
            x, w = a[0], a[1]
            key = f"mm {x.shape[-2]}x{x.shape[-1]}x{w.shape[-1]} {w.dtype}".replace("DataType.", "")
        stats[key][0] += 1
        stats[key][1] += time.perf_counter() - t0
        return r
    setattr(owner, name, timed)


def main():
    global dev
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=32768)
    try:
        ck = ComfyCheckpoint(f"{M}/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
                             f"{M}/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", 1.0,
                             lora_prefix="diffusion_model.")
        m = W.TTWanExpert(dev, ck, n_blocks=NB, prec=W.Prec(ffn_dtype=ttnn.bfloat4_b, host_text_kv=True))
        blocks, head = W.TimeConditioning(ck).rows(937.5)
        rows, head_rows = m.rows_to_dev(blocks[:NB], head)
        kv = m.text_kv(torch.randn(512, W.TEXT_DIM) * 0.1)
        t, gh, gw = 21, 30, 52
        n = t * gh * gw
        S = W.round_up(n, 256)
        p = torch.cat([torch.randn(n, W.IN_DIM * 4), torch.zeros(S - n, W.IN_DIM * 4)])
        cos, sin = W.cos_sin(W.grid_ids(t, gh, gw), S)
        tab = lambda v: m.to_dev(v.reshape(1, 1, S, -1))
        ct, st = tab(cos), tab(sin)
        m.forward(p, rows, head_rows, kv, ct, st, n)  # compile
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        m.forward(p, rows, head_rows, kv, ct, st, n)
        ttnn.synchronize_device(dev)
        plain = time.perf_counter() - t0
        for owner, names in ((ttnn, ["layer_norm", "rms_norm", "reshape", "permute", "slice", "add", "multiply"]),
                             (ttnn.experimental, ["minimal_matmul", "rotary_embedding_llama"]),
                             (ttnn.transformer, ["scaled_dot_product_attention", "concatenate_heads"])):
            for nme in names:
                wrap(owner, nme)
        t0 = time.perf_counter()
        m.forward(p, rows, head_rows, kv, ct, st, n)
        synced = time.perf_counter() - t0
        total = sum(v[1] for v in stats.values())
        out = {"blocks": NB, "tokens": n, "plain_forward_s": round(plain, 3), "synced_forward_s": round(synced, 3),
               "per_block_s": round(plain / NB, 3), "est_40_blocks_s": round(plain / NB * 40, 1),
               "ops": {k: {"calls": v[0], "s": round(v[1], 3), "share": round(v[1] / total, 3)}
                       for k, v in sorted(stats.items(), key=lambda kv_: -kv_[1][1])}}
        (ROOT / "profile_step.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
