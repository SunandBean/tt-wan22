"""Host side of one P100a test segment, with ComfyUI's own code on the CPU (ComfyUI venv, CUDA hidden):
  cond   UMT5-XXL prompt embedding [512, 4096] and the I2V concat input [1, 20, T, h, w] (mask 4 + image latent 16),
         exactly as the reference `wan_segment` graph builds them (ImageScale lanczos -> WanImageToVideo)
  decode a sampled latent (.pt or ComfyUI .latent) -> mp4 at 16 fps
    CUDA_VISIBLE_DEVICES= ~/ComfyUI/venv/bin/python host_cond.py cond <image> <prompt> <width> <height> <length>
    CUDA_VISIBLE_DEVICES= ~/ComfyUI/venv/bin/python host_cond.py decode <latent> <out.mp4>"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
COMFY = Path(os.path.expanduser(os.environ.get("COMFY", "~/ComfyUI")))
sys.path.insert(0, str(COMFY))
sys.argv_saved, sys.argv = sys.argv, [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import comfy.model_management  # noqa: E402  (reads --cpu)
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
from comfy import latent_formats  # noqa: E402
sys.argv = sys.argv_saved
torch.set_num_threads(int(os.environ.get("THREADS", "16")))
MODELS = COMFY / "models"
OUT = ROOT / "segment"
OUT.mkdir(exist_ok=True)


def load_image(path):
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)[None]


def vae():
    return comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(MODELS / "vae/wan_2.1_vae.safetensors")))


def cond(image_path, prompt, width, height, length):
    t0 = time.time()
    clip = comfy.sd.load_clip(ckpt_paths=[str(MODELS / "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors")],
                              clip_type=comfy.sd.CLIPType.WAN)
    c = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    context = c[0][0][0].float()  # [512, 4096]
    te_s = time.time() - t0
    del clip
    img = load_image(image_path)
    img = comfy.utils.common_upscale(img.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)  # ImageScale
    start = comfy.utils.common_upscale(img[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
    frames = torch.ones((length, height, width, 3)) * 0.5
    frames[: start.shape[0]] = start
    t0 = time.time()
    lat = vae().encode(frames)  # [1, 16, T, h, w]
    enc_s = time.time() - t0
    T = (length - 1) // 4 + 1
    concat_mask = torch.ones((1, 1, T, lat.shape[-2], lat.shape[-1]))
    concat_mask[:, :, : ((start.shape[0] - 1) // 4) + 1] = 0.0
    mask = (1.0 - concat_mask).repeat(1, 4, 1, 1, 1)
    image = latent_formats.Wan21().process_in(lat.float())
    concat = torch.cat([mask, image], dim=1)  # model_base WAN21.concat_cond
    torch.save({"context": context, "concat": concat, "prompt": prompt, "width": width, "height": height,
                "length": length, "text_s": te_s, "vae_encode_s": enc_s}, OUT / "cond.pt")
    print({"context": list(context.shape), "concat": list(concat.shape), "text_s": round(te_s, 1),
           "vae_encode_s": round(enc_s, 1)})


def decode(latent_path, out_path):
    if latent_path.endswith(".latent"):  # ComfyUI SaveLatent: already in VAE space
        z = comfy.utils.load_torch_file(latent_path)["latent_tensor"].float()
    else:  # P100a sampler output, sampler space
        z = latent_formats.Wan21().process_out(torch.load(latent_path)["latent"].float())
    t0 = time.time()
    frames = vae().decode(z)  # [F, H, W, 3] in [0, 1]
    frames = frames.reshape(-1, *frames.shape[-3:])
    dec_s = time.time() - t0
    import av
    arr = (frames.clamp(0, 1).numpy() * 255).astype(np.uint8)
    with av.open(out_path, "w") as c:
        s = c.add_stream("libx264", rate=16)
        s.width, s.height, s.pix_fmt = arr.shape[2], arr.shape[1], "yuv420p"
        s.options = {"crf": "14"}
        for f in arr:
            for p in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    np.save(out_path + ".frames.npy", arr[:: 8])
    print({"frames": arr.shape[0], "decode_s": round(dec_s, 1), "out": out_path})


def text(prompt):
    t0 = time.time()
    clip = comfy.sd.load_clip(ckpt_paths=[str(MODELS / "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors")],
                              clip_type=comfy.sd.CLIPType.WAN)
    c = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    context = c[0][0][0].float()
    torch.save({"context": context, "prompt": prompt, "text_s": time.time() - t0}, OUT / "text.pt")
    print({"context": list(context.shape), "nonzero_rows": int((context.abs().sum(-1) > 0).sum()),
           "text_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "text":
        text(a[1])
    elif a[0] == "cond":
        cond(a[1], a[2], int(a[3]), int(a[4]), int(a[5]))
    else:
        decode(a[1], a[2])
