"""Micro-benchmarks for the two biggest costs of a Wan step on the P100a (profile_step.py): the head split
(reshape + permute was 31%) and the self-attention SDPA (42%), at 480p x 81 frames (32760 tokens, 40 heads).
Run with SCRIPT=bench_attn.py ./run_tt.sh. Writes bench_attn.json."""
import json
import time
from pathlib import Path

import torch
import ttnn

ROOT = Path(__file__).resolve().parent
H, D, DIM = 40, 128, 5120
n, S = 32760, 32768
out = {}


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def timeit(fn, reps=3):
    fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        r = fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / reps, r


dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=32768)
grid = dev.compute_with_storage_grid_size()
T = lambda t: ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
try:
    torch.manual_seed(0)
    # ---------------------------------------------------------------- head split
    q, k, v = (T(torch.randn(1, 1, S, DIM)) for _ in range(3))

    def split_reshape():
        r = []
        for x in (q, k, v):
            r.append(ttnn.permute(ttnn.reshape(x, [1, S, H, D]), (0, 2, 1, 3)))
        return r

    def split_fused():
        qkv = ttnn.concat([q, k, v], dim=-1)
        r = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=H, num_kv_heads=H, transpose_k_heads=False,
                                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(qkv)
        return r

    ta, ra = timeit(split_reshape)
    tb, rb = timeit(split_fused)
    out["split_reshape_permute_s"] = round(ta, 4)
    out["split_concat_nlp_create_qkv_heads_s"] = round(tb, 4)
    out["split_equal"] = all(pcc(ttnn.to_torch(x)[:, :, :64], ttnn.to_torch(y)[:, :, :64]) > 0.99999 for x, y in zip(ra, rb))
    for t_ in (*ra, *rb):
        ttnn.deallocate(t_)
    # ---------------------------------------------------------------- self-attention SDPA
    qh = T(torch.randn(1, H, S, D))
    kh = T(torch.randn(1, H, n, D) * 1.5)
    vh = T(torch.randn(1, H, n, D))
    results = {}
    ref = None
    for fid_name, fid in (("HiFi4", ttnn.MathFidelity.HiFi4), ("HiFi2", ttnn.MathFidelity.HiFi2), ("LoFi", ttnn.MathFidelity.LoFi)):
        for fp32 in (True, False):
            for qc, kc in ((256, 256), (128, 256), (256, 512), (512, 256), (128, 512), (64, 512)):
                for approx in (False, True):
                    if fid_name != "HiFi4" and (qc, kc) not in ((256, 256), (128, 512), (256, 512)):
                        continue
                    name = f"{fid_name}{'_fp32' if fp32 else ''}_q{qc}_k{kc}{'_approx' if approx else ''}"
                    ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=fid, math_approx_mode=False,
                                                                fp32_dest_acc_en=fp32, packer_l1_acc=False)
                    cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=qc, k_chunk_size=kc,
                                                 exp_approx_mode=approx)
                    try:
                        t_, r = timeit(lambda: ttnn.transformer.scaled_dot_product_attention(
                            qh, kh, vh, is_causal=False, scale=D ** -0.5, program_config=cfg, compute_kernel_config=ck), reps=2)
                        rt = ttnn.to_torch(r)[:, :, :2048].float()
                        ttnn.deallocate(r)
                        if ref is None:
                            ref = rt
                        results[name] = {"s": round(t_, 3), "pcc_vs_first": round(pcc(rt, ref), 6)}
                    except Exception as e:
                        results[name] = {"error": str(e)[:160]}
                    out["sdpa"] = results
                    (ROOT / "bench_attn.json").write_text(json.dumps(out, indent=1))
finally:
    (ROOT / "bench_attn.json").write_text(json.dumps(out, indent=1))
    ttnn.close_mesh_device(dev)
print(json.dumps(out, indent=1))
