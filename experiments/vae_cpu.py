"""CPU cost of the Wan 2.1 VAE for one 480p x 81-frame segment (encode of the I2V conditioning video, decode)."""
import os, sys, time, resource, torch
from pathlib import Path
from diffusers import AutoencoderKLWan
from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
from safetensors.torch import load_file
torch.set_num_threads(int(os.environ.get("THREADS", "16")))
vae = AutoencoderKLWan(); vae.load_state_dict(convert_wan_vae_to_diffusers(load_file("/comfy/vae/wan_2.1_vae.safetensors"))); vae.eval().to(torch.bfloat16)
F = int(os.environ.get("FRAMES", "81"))
x = torch.zeros(1, 3, F, 480, 832, dtype=torch.bfloat16)
with torch.no_grad():
    t = time.time(); z = vae.encode(x).latent_dist.mode(); te = time.time() - t
    print("encode", F, round(te, 1), z.shape, flush=True)
    t = time.time(); y = vae.decode(z).sample; td = time.time() - t
    print("decode", round(td, 1), y.shape, "maxrss_gb", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1), flush=True)
