"""UMT5 host encoder vs ComfyUI's conditioning for the same prompt (segment/text.pt written by
experiments/host_cond.py text). Usage: python check_umt5.py <text.pt> <umt5 file>"""
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from umt5 import UMT5Encoder

ref = torch.load(sys.argv[1])
enc = UMT5Encoder(sys.argv[2], threads=16)
t0 = time.time()
out = enc(ref["prompt"])
dt = time.time() - t0
r = ref["context"].float()
valid = int((r.abs().sum(-1) > 0).sum())
a, b = out[:valid].flatten().double(), r[:valid].flatten().double()
a, b = a - a.mean(), b - b.mean()
print({"tokens": len(enc.tokens(ref["prompt"])), "comfy_valid_rows": valid, "pcc": float(a @ b / (a.norm() * b.norm())),
       "maxdiff": float((out - r).abs().max()), "pad_rows_zero": bool((out[valid:] == 0).all()), "s": round(dt, 1)})
