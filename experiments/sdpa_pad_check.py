"""Which way of hiding 8 pad keys works for the fused SDPA on one P100a: logical-shape padding or a bfp mask."""
import json, os, sys, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tt_device import close_device, open_device
import ttnn

def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double(); a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))

dev = open_device()
out = {}
try:
    H, D, n, S = 8, 128, 2040, 2048
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, H, S, D) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(q[:, :, :n], k[:, :, :n], v[:, :, :n])
    grid = dev.compute_with_storage_grid_size()
    ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    T = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
    for chunk in (256, 128):
        cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=chunk, k_chunk_size=chunk, exp_approx_mode=False)
        # (a) logical length n
        try:
            a = ttnn.transformer.scaled_dot_product_attention(T(q[:, :, :n]), T(k[:, :, :n]), T(v[:, :, :n]), is_causal=False, program_config=cfg, compute_kernel_config=ck)
            out[f"logical_c{chunk}"] = pcc(ttnn.to_torch(a)[:, :, :n], ref)
        except Exception as e:
            out[f"logical_c{chunk}"] = str(e)[:200]
        # (b) masks in bfp formats, (c) unmasked for reference
        m = torch.zeros(S); m[n:] = -1e9
        m = m.expand(S, S).reshape(1, 1, S, S)
        for name, dt in (("mask_bf16", ttnn.bfloat16), ("mask_bfp8", ttnn.bfloat8_b), ("mask_bfp4", ttnn.bfloat4_b), ("nomask", None)):
            try:
                a = ttnn.transformer.scaled_dot_product_attention(T(q), T(k), T(v), attn_mask=None if dt is None else T(m, dt), is_causal=False, program_config=cfg, compute_kernel_config=ck)
                out[f"{name}_c{chunk}"] = pcc(ttnn.to_torch(a)[:, :, :n], ref)
            except Exception as e:
                out[f"{name}_c{chunk}"] = str(e)[:200]
finally:
    close_device(dev)
print(json.dumps(out, indent=1))
