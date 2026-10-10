"""One Wan 2.2 I2V A14B segment sampled on the P100a (both experts resident, FFN bfloat4_b), from the host
conditioning in segment/ (text.pt from host_cond.py, concat.pt from vae_wan.py). Same schedule as
the reference `wan_segment` graph: shift 5, simple, 4 steps, high noise expert for steps 0-1, low for 2-3, Euler,
cfg 1. Writes segment/latent_p100a.pt (sampler space) and segment/report.json."""
import json, os, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import wan_tt as W  # noqa: E402
from tt_device import close_device, dram_stats, open_device  # noqa: E402
from comfy_weights import ComfyCheckpoint, wan_expert  # noqa: E402

SEG = ROOT / "segment"
M = os.environ.get("COMFY_MODELS", "/comfy")
SEED = int(os.environ.get("SEED", "42"))
report = {}


def save():
    (SEG / "report.json").write_text(json.dumps(report, indent=2))


def main():
    import ttnn
    torch.set_num_threads(int(os.environ.get("THREADS", "16")))
    context = torch.load(SEG / "text.pt")["context"]
    concat = torch.load(SEG / "concat.pt")["concat"]
    _, _, T, h, w = concat.shape
    gh, gw = h // 2, w // 2
    n = T * gh * gw
    S = W.round_up(n, 256)
    sig = W.sigmas_sd3()
    report.update({"sigmas": sig, "tokens": n, "padded": S, "seed": SEED})
    dev = open_device()
    try:
        experts = {}
        t0 = time.time()
        for e in ("high", "low"):
            ck = ComfyCheckpoint(f"{M}/diffusion_models/wan2.2_i2v_{e}_noise_14B_fp8_scaled.safetensors",
                                 f"{M}/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_{e}_noise.safetensors", 1.0,
                                 lora_prefix="diffusion_model.")
            m = W.TTWanExpert(dev, ck, prec=W.Prec(ffn_dtype=ttnn.bfloat4_b))
            experts[e] = (m, W.TimeConditioning(ck), None)
        report["load_s"] = round(time.time() - t0, 1)
        report["dram_after_load"] = dram_stats(dev)
        save()
        t0 = time.time()
        experts = {e: (m, tc, m.text_kv(context)) for e, (m, tc, _) in experts.items()}
        report["text_kv_s"] = round(time.time() - t0, 2)
        m0 = experts["high"][0]
        cos, sin = W.cos_sin(W.grid_ids(T, gh, gw), S)
        tab = lambda v: m0.to_dev(v.reshape(1, 1, S, -1))
        cos_t, sin_t = tab(cos), tab(sin)
        mask = None  # pad keys are hidden through the logical K/V length (sdpa_pad_check.py)
        g = torch.Generator("cpu").manual_seed(SEED)
        x = torch.randn((1, 16, T, h, w), generator=g, dtype=torch.float32) * sig[0]
        steps = []
        for i in range(4):
            e = "high" if i < 2 else "low"
            m, tc, kv = experts[e]
            t1 = time.time()
            blocks, head = tc.rows(sig[i] * 1000.0)
            rows, head_rows = m.rows_to_dev(blocks, head)
            inp = torch.cat([x.to(torch.bfloat16).float(), concat], dim=1)
            patches = W.patchify(inp)
            patches = torch.cat([patches, torch.zeros(S - n, patches.shape[1])])
            v = m.forward(patches, rows, head_rows, kv, cos_t, sin_t, n, mask=mask)
            v = W.unpatchify(v, T, gh, gw)
            s, s1 = torch.tensor(sig[i]), sig[i + 1]
            denoised = x - v * s
            x = x + (x - denoised) / s * (s1 - sig[i])
            steps.append({"expert": e, "sigma": sig[i], "s": round(time.time() - t1, 1),
                          "v_absmean": float(v.abs().mean()), "finite": bool(torch.isfinite(v).all())})
            report["steps"] = steps
            save()
        torch.save({"latent": x, "seed": SEED}, SEG / "latent_p100a.pt")
        report["sampling_s"] = round(sum(s["s"] for s in steps), 1)
        report["dram_end"] = dram_stats(dev)
    finally:
        save()
        close_device(dev)
    print(json.dumps(report)[:3000])


if __name__ == "__main__":
    main()
