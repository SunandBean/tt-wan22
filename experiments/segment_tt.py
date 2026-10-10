"""A whole Wan 2.2 I2V segment on the P100a in one process (tt-inference-server image): tt_dit VAE encoder for
the I2V conditioning, both 14B experts resident (FFN bfloat4_b) for the 4 steps, tt_dit VAE decoder. The prompt
embedding (segment/text.pt, UMT5 on the host) is the only host input besides the start frame.
Writes segment/segment_tt.json and segment/p100a_full.mp4.npy. Run with SCRIPT=segment_tt.py ./run_tt.sh."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SEG = ROOT / "segment"
sys.path.insert(0, str(ROOT))
import ttnn  # noqa: E402
import vae_tt as V  # noqa: E402
import wan_tt as W  # noqa: E402
from comfy_weights import ComfyCheckpoint, wan_expert  # noqa: E402

M = os.environ.get("COMFY_MODELS", "/comfy")
SEED = int(os.environ.get("SEED", "42"))
report = {"seed": SEED}


def dram(dev):
    ttnn.synchronize_device(dev)
    v = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
    return round(int(v.total_bytes_allocated_per_bank) * int(v.num_banks) / 2**30, 2)


def save():
    (SEG / "segment_tt.json").write_text(json.dumps(report, indent=2))


def main():
    from vae_wan import lanczos_center
    torch.set_num_threads(int(os.environ.get("THREADS", "16")))
    context = torch.load(SEG / "text.pt")["context"]
    w, h, length = 832, 480, 81
    dev = V.open_dev()
    tv = V.torch_vae()
    try:
        t0 = time.time()
        experts = {}
        load_times = {}
        for e in ("high", "low"):
            t_e = time.time()
            ck = ComfyCheckpoint(f"{M}/diffusion_models/wan2.2_i2v_{e}_noise_14B_fp8_scaled.safetensors",
                                 f"{M}/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_{e}_noise.safetensors", 1.0,
                                 lora_prefix="diffusion_model.")
            experts[e] = (W.TTWanExpert(dev, ck, prec=W.Prec(ffn_dtype=ttnn.bfloat4_b, host_text_kv=True),
                                          cache_dir=os.environ.get("WAN_CACHE") or None), W.TimeConditioning(ck))
            load_times[e] = round(time.time() - t_e, 1)
            c = experts[e][0].cache
            report.setdefault("cache", {})[e] = None if c is None else {"hits": c.hits, "built": c.built, "dir": str(c.dir)}
        report["expert_load_s"] = load_times
        enc = V.build(V.WanEncoder, dev, tv.config, in_channels=3, dtype=ttnn.bfloat16)
        enc.load_torch_state_dict(tv.state_dict())
        dec = V.build(V.WanDecoder, dev, tv.config, out_channels=3, dtype=ttnn.bfloat16)
        dec.load_torch_state_dict(tv.state_dict())
        report["load_s"] = round(time.time() - t0, 1)
        report["dram_after_load_gb"] = dram(dev)
        save()
        base = dict(report)
        kv_cache = {}
        runs = []
        for run in range(int(os.environ.get("REPEAT", "1"))):
            report.clear(); report.update(base)
            seg0 = time.time()
            # 1. conditioning video on the card
            frames = torch.full((length, h, w, 3), 0.5)
            frames[0] = lanczos_center(ROOT / "start.png", w, h)
            t0 = time.time()
            out, nh, nw = V.encode_streamed(enc, frames.permute(3, 0, 1, 2)[None] * 2 - 1, dev,
                                            chunk=int(os.environ.get("ENC_CHUNK", "4")))
            lat = V.to_host(out, dev)[:, :16, :, :nh, :nw].float()
            ttnn.deallocate(out)
            report["encode_s"] = round(time.time() - t0, 1)
            report["dram_after_encode_gb"] = dram(dev)
            T = lat.shape[2]
            mask = torch.zeros(1, 4, T, lat.shape[-2], lat.shape[-1])
            mask[:, :, 0] = 1.0
            concat = torch.cat([mask, (lat - V.MEAN) / V.STD], dim=1)
            ref = torch.load(SEG / "concat.pt")["concat"]
            report["concat_pcc_vs_cpu"] = V.pcc(concat, ref)
            save()
            # 2. sampling
            _, _, T, lh_, lw_ = concat.shape
            gh, gw = lh_ // 2, lw_ // 2
            n = T * gh * gw
            S = W.round_up(n, 256)
            sig = W.sigmas_sd3()
            t0 = time.time()
            t_kv = time.time()
            if kv_cache.get("key") != id(context):
                kv_cache.update(key=id(context), kv={e: m.text_kv(context) for e, (m, _) in experts.items()})
            kv = kv_cache["kv"]
            report["text_kv_host_s"] = round(time.time() - t_kv, 1)
            m0 = experts["high"][0]
            cos, sin = W.cos_sin(W.grid_ids(T, gh, gw), S)
            tab = lambda v: m0.to_dev(v.reshape(1, 1, S, -1))
            cos_t, sin_t = tab(cos), tab(sin)
            g = torch.Generator("cpu").manual_seed(SEED)
            x = torch.randn((1, 16, T, lh_, lw_), generator=g, dtype=torch.float32) * sig[0]
            steps = []
            for i in range(4):
                e = "high" if i < 2 else "low"
                m, tc = experts[e]
                t1 = time.time()
                blocks, head = tc.rows(sig[i] * 1000.0)
                rows, head_rows = m.rows_to_dev(blocks, head)
                p = W.patchify(torch.cat([x.to(torch.bfloat16).float(), concat], dim=1))
                p = torch.cat([p, torch.zeros(S - n, p.shape[1])])
                v = W.unpatchify(m.forward(p, rows, head_rows, kv[e], cos_t, sin_t, n), T, gh, gw)
                for r in rows:
                    for t in r.values():
                        ttnn.deallocate(t)
                s = torch.tensor(sig[i])
                x = x + (x - (x - v * s)) / s * (sig[i + 1] - sig[i])
                steps.append({"expert": e, "s": round(time.time() - t1, 1)})
            report["steps"] = steps
            report["sampling_s"] = round(time.time() - t0, 1)
            ref_lat = torch.load(SEG / "latent_p100a.pt")["latent"]
            report["latent_pcc_vs_stage1"] = V.pcc(x, ref_lat)
            prev = SEG / "latent_full_prev.pt"
            if prev.exists():
                report["latent_pcc_vs_previous_full_run"] = V.pcc(x, torch.load(prev)["latent"])
            torch.save({"latent": x}, SEG / "latent_full.pt")
            report["dram_after_sampling_gb"] = dram(dev)
            save()
            # 3. decode on the card
            z = x * V.STD + V.MEAN
            t0 = time.time()
            y = V.decode_streamed(dec, z, dev, chunk=int(os.environ.get("T_CHUNK", "1")))
            report["decode_s"] = round(time.time() - t0, 1)
            arr = ((y[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) / 2 * 255).numpy().astype(np.uint8)
            np.save(SEG / "p100a_full.mp4.npy", arr)
            gpu = np.load(SEG / "gpu.mp4.npy")
            report["psnr_vs_gpu_frames_0_40_80"] = [round(float(10 * np.log10(255 ** 2 / ((arr[i].astype(float) - gpu[i].astype(float)) ** 2).mean())), 1) for i in (0, 40, 80)]
            report["segment_s"] = round(time.time() - seg0, 1)
            report["frames"] = int(arr.shape[0])
            runs.append(dict(report))
        report.clear(); report.update(base); report["runs"] = runs
    except Exception:
        import traceback
        report["error"] = traceback.format_exc()[-4000:]
    finally:
        save()
        ttnn.close_mesh_device(dev)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    print(json.dumps(report, indent=1)[:4000])


if __name__ == "__main__":
    main()
