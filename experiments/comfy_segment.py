"""The same segment through the reference ComfyUI setup (GPU): the `wan_segment` graph (14B, lightx2v, shift 5,
high 0-2 / low 2-4) ending in SaveLatent, for a latent-level comparison with the P100a run. Host python, stdlib:
    COMFY_IN=<ComfyUI input dir> COMFY_OUT=<ComfyUI output dir> python3 comfy_segment.py ..."""
import json, os, shutil, sys, time, urllib.request, uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parent
URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8190")
IN = Path(os.path.expanduser(os.environ["COMFY_IN"]))
OUTDIR = Path(os.path.expanduser(os.environ["COMFY_OUT"]))
NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
       "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


def main(image, prompt, w, h, length, seed):
    nodes = {}

    def add(cls, **inputs):
        k = str(len(nodes) + 1)
        nodes[k] = {"class_type": cls, "inputs": inputs}
        return k
    o = lambda k, s=0: [k, s]
    experts = []
    for e in ("high", "low"):
        m = add("UNETLoader", unet_name=f"wan2.2_i2v_{e}_noise_14B_fp8_scaled.safetensors", weight_dtype="default")
        m = add("LoraLoaderModelOnly", model=o(m), lora_name=f"wan2.2_i2v_lightx2v_4steps_lora_v1_{e}_noise.safetensors", strength_model=1.0)
        experts.append(add("ModelSamplingSD3", model=o(m), shift=5.0))
    clip = add("CLIPLoader", clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors", type="wan", device="default")
    vae = add("VAELoader", vae_name="wan_2.1_vae.safetensors")
    pos = add("CLIPTextEncode", text=prompt, clip=o(clip))
    neg = add("CLIPTextEncode", text=NEG, clip=o(clip))
    name = f"wan_parity_{uuid.uuid4().hex[:6]}.png"
    shutil.copy(image, IN / name)
    img = add("ImageScale", image=o(add("LoadImage", image=name)), upscale_method="lanczos", width=w, height=h, crop="center")
    cond = add("WanImageToVideo", positive=o(pos), negative=o(neg), vae=o(vae), width=w, height=h, length=length,
               batch_size=1, start_image=o(img))
    s = dict(steps=4, cfg=1.0, sampler_name="euler", scheduler="simple", positive=o(cond, 0), negative=o(cond, 1))
    hi = add("KSamplerAdvanced", **s, model=o(experts[0]), add_noise="enable", noise_seed=seed, latent_image=o(cond, 2),
             start_at_step=0, end_at_step=2, return_with_leftover_noise="enable")
    lo = add("KSamplerAdvanced", **s, model=o(experts[1]), add_noise="disable", noise_seed=0, latent_image=o(hi),
             start_at_step=2, end_at_step=4, return_with_leftover_noise="disable")
    prefix = f"wan_parity_{uuid.uuid4().hex[:6]}"
    add("SaveLatent", samples=o(lo), filename_prefix=f"latents/{prefix}")
    t0 = time.time()
    req = urllib.request.Request(URL + "/prompt", data=json.dumps({"prompt": nodes}).encode(), headers={"Content-Type": "application/json"})
    pid = json.loads(urllib.request.urlopen(req).read())["prompt_id"]
    while True:
        hist = json.loads(urllib.request.urlopen(f"{URL}/history/{pid}").read())
        st = hist.get(pid, {}).get("status", {})
        if st.get("completed"):
            break
        if st.get("status_str") == "error":
            raise SystemExit(json.dumps(st)[:2000])
        time.sleep(2)
    (IN / name).unlink(missing_ok=True)
    f = next(OUTDIR.rglob(prefix + "*.latent"))
    shutil.copy(f, ROOT / "segment/latent_gpu.latent")
    print({"gpu_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], int(a[2]), int(a[3]), int(a[4]), int(a[5]))
