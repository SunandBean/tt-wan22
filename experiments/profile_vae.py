"""Per-op time of the tt_dit Wan VAE encoder / decoder on the P100a (480p, ENC_FRAMES frames streamed), every
ttnn op synced and timed. Run with SCRIPT=profile_vae.py ./run_tt.sh. Writes profile_vae.json."""
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
import vae_tt as V  # noqa: E402

stats = collections.defaultdict(lambda: [0, 0.0])


def wrap_all(dev):
    names = []
    for owner, prefix in ((ttnn, ""), (ttnn.experimental, "experimental.")):
        for n in dir(owner):
            f = getattr(owner, n)
            if n.startswith("_") or not callable(f) or isinstance(f, type) or n in (
                    "synchronize_device", "get_memory_view", "to_torch", "from_torch", "Tensor", "deallocate"):
                continue
            if type(f).__name__ not in ("FastOperation", "Operation", "function", "builtin_function_or_method"):
                continue

            def make(fn, key):
                def timed(*a, **k):
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    r = fn(*a, **k)
                    ttnn.synchronize_device(dev)
                    stats[key][0] += 1
                    stats[key][1] += time.perf_counter() - t0
                    return r
                return timed
            try:
                setattr(owner, n, make(f, prefix + n))
                names.append(prefix + n)
            except (AttributeError, TypeError):
                pass
    return names


def main():
    dev = V.open_dev()
    tv = V.torch_vae()
    out = {}
    try:
        F = int(os.environ.get("ENC_FRAMES", "21"))
        x = torch.rand(1, 3, F, 480, 832) * 2 - 1
        enc = V.build(V.WanEncoder, dev, tv.config, in_channels=3, dtype=ttnn.bfloat16)
        enc.load_torch_state_dict(tv.state_dict())
        o, _, _ = V.encode_streamed(enc, x[:, :, :5], dev)  # compile
        ttnn.deallocate(o)
        t0 = time.perf_counter()
        o, _, _ = V.encode_streamed(enc, x, dev)
        ttnn.synchronize_device(dev)
        out["encode_plain_s"] = round(time.perf_counter() - t0, 2)
        ttnn.deallocate(o)
        wrap_all(dev)
        t0 = time.perf_counter()
        o, _, _ = V.encode_streamed(enc, x, dev)
        out["encode_synced_s"] = round(time.perf_counter() - t0, 2)
        total = sum(v[1] for v in stats.values())
        out["frames"] = F
        out["ops"] = {k: {"calls": v[0], "s": round(v[1], 3), "share": round(v[1] / total, 3)}
                      for k, v in sorted(stats.items(), key=lambda kv: -kv[1][1])[:25]}
    finally:
        (ROOT / "profile_vae.json").write_text(json.dumps(out, indent=1))
        ttnn.close_mesh_device(dev)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
