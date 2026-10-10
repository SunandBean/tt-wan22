"""Wan 2.1 VAE in the model image (diffusers AutoencoderKLWan, bf16, CPU) for the P100a test segment:
  encode <start.png> <w> <h> <length>  -> segment/concat.pt   (WanImageToVideo + WAN21.concat_cond)
  decode <latent.pt|.latent> <out.mp4>                       (frames + mp4 at 16 fps)"""
import os, sys, time
from pathlib import Path
import numpy as np
import torch
from diffusers import AutoencoderKLWan
from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "segment"
OUT.mkdir(exist_ok=True)
torch.set_num_threads(int(os.environ.get("THREADS", "16")))
MEAN = torch.tensor([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                     0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]).view(1, 16, 1, 1, 1)
STD = torch.tensor([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]).view(1, 16, 1, 1, 1)


def vae():
    v = AutoencoderKLWan()
    v.load_state_dict(convert_wan_vae_to_diffusers(load_file(os.environ.get("WAN_VAE", "/comfy/vae/wan_2.1_vae.safetensors"))))
    return v.eval().to(torch.bfloat16)


def lanczos_center(path, w, h):
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    ow, oh = img.size
    oa, na = ow / oh, w / h
    x = y = 0
    if oa > na:
        x = round((ow - ow * (na / oa)) / 2)
    elif oa < na:
        y = round((oh - oh * (oa / na)) / 2)
    img = img.crop((x, y, ow - x, oh - y)).resize((w, h), resample=Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)


@torch.no_grad()
def encode(path, w, h, length):
    start = lanczos_center(path, w, h)
    frames = torch.full((length, h, w, 3), 0.5)
    frames[0] = start
    x = (frames.permute(3, 0, 1, 2)[None] * 2 - 1).to(torch.bfloat16)  # [1, 3, F, H, W]
    t0 = time.time()
    lat = vae().encode(x).latent_dist.mode().float()
    enc_s = time.time() - t0
    T = lat.shape[2]
    mask = torch.zeros(1, 4, T, lat.shape[-2], lat.shape[-1])
    mask[:, :, 0] = 1.0  # 1 - concat_mask: the latent frame holding the start image
    concat = torch.cat([mask, (lat - MEAN) / STD], dim=1)
    torch.save({"concat": concat, "vae_encode_s": enc_s, "width": w, "height": h, "length": length}, OUT / "concat.pt")
    print({"concat": list(concat.shape), "vae_encode_s": round(enc_s, 1)})


@torch.no_grad()
def decode(path, out):
    if path.endswith(".latent"):
        z = load_file(path)["latent_tensor"].float()
    else:
        z = torch.load(path)["latent"].float() * STD + MEAN
    t0 = time.time()
    y = vae().decode(z.to(torch.bfloat16)).sample.float()  # [1, 3, F, H, W]
    dec_s = time.time() - t0
    arr = ((y[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) / 2 * 255).numpy().astype(np.uint8)
    np.save(out + ".npy", arr)
    print({"frames": arr.shape[0], "decode_s": round(dec_s, 1)})
    try:
        import av
    except ImportError:  # the model image has no PyAV: mux on the host (mux.py)
        return
    with av.open(out, "w") as c:
        s = c.add_stream("libx264", rate=16)
        s.width, s.height, s.pix_fmt = arr.shape[2], arr.shape[1], "yuv420p"
        s.options = {"crf": "14"}
        for f in arr:
            for p in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


if __name__ == "__main__":
    a = sys.argv[1:]
    encode(a[1], int(a[2]), int(a[3]), int(a[4])) if a[0] == "encode" else decode(a[1], a[2])
