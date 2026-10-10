"""Wan 2.1 VAE on the P100a with tt-metal's tt_dit WanEncoder / WanDecoder (single chip, mesh 1x1), in the
tt-inference-server image (its tt-metal python_env has tt_dit, diffusers and the conv3d kernels these need).
  encode   segment/concat_tt.pt from start.png (WanImageToVideo conditioning, compared with segment/concat.pt)
  decode   segment/latent_p100a.pt -> segment/p100a_ttvae.mp4.npy (compared with the CPU decode p100a.mp4.npy)
Run through run_tt.sh (memory cap)."""
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
from diffusers import AutoencoderKLWan  # noqa: E402
from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from models.tt_dit.models.vae.vae_wan2_1 import WanDecoder, WanEncoder  # noqa: E402
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig  # noqa: E402
from models.tt_dit.parallel.manager import CCLManager  # noqa: E402
from models.tt_dit.utils.conv3d import conv_pad_height, conv_pad_in_channels, conv_pad_width  # noqa: E402
from models.tt_dit.utils.tensor import typed_tensor_2dshard  # noqa: E402

MEAN = torch.tensor([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                     0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]).view(1, 16, 1, 1, 1)
STD = torch.tensor([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]).view(1, 16, 1, 1, 1)
report = {}


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def torch_vae():
    v = AutoencoderKLWan()
    v.load_state_dict(convert_wan_vae_to_diffusers(load_file(os.environ.get("WAN_VAE", "/comfy/vae/wan_2.1_vae.safetensors"))))
    return v.eval()


def open_dev():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    return ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=int(os.environ.get("L1_SMALL", "32768")))


def build(cls, dev, cfg, **extra):
    ccl = CCLManager(dev, topology=ttnn.Topology.Linear, num_links=1)
    pc = VaeHWParallelConfig(height_parallel=ParallelFactor(factor=1, mesh_axis=0),
                             width_parallel=ParallelFactor(factor=1, mesh_axis=1))
    kw = dict(base_dim=cfg.base_dim, z_dim=cfg.z_dim, dim_mult=cfg.dim_mult, num_res_blocks=cfg.num_res_blocks,
              attn_scales=cfg.attn_scales, temperal_downsample=cfg.temperal_downsample, is_residual=cfg.is_residual,
              mesh_device=dev, ccl_manager=ccl, parallel_config=pc, **extra)
    return cls(**kw)


def to_dev(x_BCTHW, dev, multiple, dtype=ttnn.bfloat16):
    x = conv_pad_in_channels(x_BCTHW.permute(0, 2, 3, 4, 1))
    x, lh = conv_pad_height(x, multiple)
    x, lw = conv_pad_width(x, multiple)
    return typed_tensor_2dshard(x, dev, layout=ttnn.ROW_MAJOR_LAYOUT, shard_mapping={0: 2, 1: 3}, dtype=dtype), lh, lw


def to_host(t, dev):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(dev, mesh_shape=(1, 1), dims=[3, 4]))


def encode(dev, tv):
    from vae_wan import lanczos_center  # same frame preparation as the CPU path
    c = torch.load(SEG / "concat.pt")
    w, h, length = c["width"], c["height"], c["length"]
    frames = torch.full((length, h, w, 3), 0.5)
    frames[0] = lanczos_center(ROOT / "start.png", w, h)
    x = frames.permute(3, 0, 1, 2)[None] * 2 - 1  # [1, 3, F, H, W]
    t0 = time.time()
    enc = build(WanEncoder, dev, tv.config, in_channels=3, dtype=ttnn.bfloat16)
    enc.load_torch_state_dict(tv.state_dict())
    report["encoder_load_s"] = round(time.time() - t0, 1)
    xt, lh, lw = to_dev(x, dev, 8)
    t0 = time.time()
    out, nh, nw = enc(xt, lh, logical_w=lw)
    lat = to_host(out, dev)
    report["encode_s"] = round(time.time() - t0, 1)
    lat = lat[:, :, :, :nh, :nw].float()  # BTHWC or BCTHW? normalise below
    if lat.shape[1] != 16 and lat.shape[-1] in (16, 32):
        lat = lat.permute(0, 4, 1, 2, 3)
    lat = lat[:, :16]  # mean of the distribution
    image = (lat - MEAN) / STD
    ref = c["concat"][:, 4:]
    report["encode_shape"] = list(lat.shape)
    report["encode_pcc_vs_cpu"] = pcc(image, ref) if image.shape == ref.shape else f"shape {list(image.shape)} vs {list(ref.shape)}"
    torch.save({"image": image}, SEG / "concat_tt_image.pt")


def decode_mem(dev, tv):
    """Decoder alone, streamed: DRAM in use after loading and after each latent frame (peaks are higher)."""
    def gb():
        ttnn.synchronize_device(dev)
        v = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
        return round(int(v.total_bytes_allocated_per_bank) * int(v.num_banks) / 2**30, 2), round(int(v.largest_contiguous_bytes_free_per_bank) * int(v.num_banks) / 2**30, 2)
    z = torch.load(SEG / "latent_p100a.pt")["latent"].float() * STD + MEAN
    dec = build(WanDecoder, dev, tv.config, out_channels=3, dtype=ttnn.bfloat16)
    dec.load_torch_state_dict(tv.state_dict())
    report["dram_decoder_loaded"] = gb()
    import vae_tt as me
    orig = dec.decoder
    seen = []

    def spy(*a, **k):
        r = orig(*a, **k)
        seen.append(gb())
        return r
    dec.decoder = spy
    t0 = time.time()
    y = decode_streamed(dec, z, dev)
    report["decode_streamed_s"] = round(time.time() - t0, 1)
    report["dram_after_chunk_max"] = max(s_[0] for s_ in seen)
    report["dram_after_chunks_first5"] = seen[:5]
    report["dram_end"] = gb()


def decode(dev, tv):
    z = torch.load(SEG / "latent_p100a.pt")["latent"].float() * STD + MEAN  # VAE space [1, 16, T, h, w]
    t0 = time.time()
    dec = build(WanDecoder, dev, tv.config, out_channels=3, dtype=ttnn.bfloat16)
    dec.load_torch_state_dict(tv.state_dict())
    report["decoder_load_s"] = round(time.time() - t0, 1)
    zt, lh, lw = to_dev(z, dev, 1)
    t0 = time.time()
    chunk = os.environ.get("T_CHUNK")
    out, nh, nw = dec(zt, lh, t_chunk_size=int(chunk) if chunk else None, logical_w=lw)
    y = to_host(out, dev)
    report["decode_s"] = round(time.time() - t0, 1)
    report["decode_raw_shape"] = list(y.shape)
    y = y[:, :, :, :nh, :nw] if y.shape[1] == 3 else y
    if y.shape[1] != 3:  # BTHWC -> BCTHW
        y = y[..., :3].permute(0, 4, 1, 2, 3)
    y = y[:, :, :, :nh, :nw].float()
    arr = ((y[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) / 2 * 255).numpy().astype(np.uint8)
    np.save(SEG / "p100a_ttvae.mp4.npy", arr)
    ref = np.load(SEG / "p100a.mp4.npy")
    if ref.shape == arr.shape:
        mse = ((ref.astype(np.float64) - arr.astype(np.float64)) ** 2).mean()
        report["decode_psnr_vs_cpu"] = round(float(10 * np.log10(255 ** 2 / mse)), 2)
        report["decode_pcc_vs_cpu"] = pcc(torch.from_numpy(arr).float(), torch.from_numpy(ref).float())
    else:
        report["decode_shape_mismatch"] = [list(arr.shape), list(ref.shape)]


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "16")))
    dev = open_dev()
    tv = torch_vae()
    try:
        for stage in os.environ.get("STAGES", "decode,encode").split(","):
            try:
                globals()[stage](dev, tv)
            except Exception as exc:
                import traceback
                report[stage + "_error"] = traceback.format_exc()[-3000:]
            (SEG / "vae_tt_report.json").write_text(json.dumps(report, indent=2))
    finally:
        ttnn.close_mesh_device(dev)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    print(json.dumps(report, indent=1)[:4000])



def encode_streamed(enc, x_BCTHW, dev, chunk=4):
    """WanEncoder.forward with the video uploaded one time chunk at a time (frame 0 alone, then `chunk` frames),
    so only a chunk of the input is ever on the card. Same caching and output as encoder_t_chunk_size=chunk."""
    T = x_BCTHW.shape[2]
    enc.clear_cache()
    out = None
    nh = nw = None
    for t0, t1 in [(0, 1)] + [(s, min(s + chunk, T)) for s in range(1, T, chunk)]:
        xt, lh, lw = to_dev(x_BCTHW[:, :, t0:t1], dev, 8)
        enc._conv_idx = [0]
        o, nh, nw = enc.encoder(xt, lh, feat_cache=enc._feat_cache, feat_idx=enc._conv_idx, logical_w=lw)
        ttnn.deallocate(xt)
        if out is None:
            out = o
        else:
            cat = ttnn.concat([out, o], dim=1)
            ttnn.deallocate(out)
            ttnn.deallocate(o)
            out = cat
    enc.clear_cache()
    tile = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
    tile = enc.quant_conv(tile)
    rm = ttnn.to_layout(tile, ttnn.ROW_MAJOR_LAYOUT)
    bcthw = ttnn.permute(rm, (0, 4, 1, 2, 3))[:, : enc.z_dim, :, :, :]
    return bcthw, nh, nw


def decode_streamed(dec, z_BCTHW, dev, chunk=1):
    """WanDecoder.forward (t_chunk_size=chunk) with every output chunk moved to the host as soon as it is made,
    so the decoded video never accumulates on the card. Returns [1, 3, F, H, W] float32 in [-1, 1]."""
    zt, lh, lw = to_dev(z_BCTHW, dev, 1)
    dec.clear_cache()
    x = ttnn.to_layout(dec.post_quant_conv(ttnn.to_layout(zt, ttnn.TILE_LAYOUT)), ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(zt)
    T = z_BCTHW.shape[2]
    frames = []
    for t0 in range(0, T, chunk):
        dec._conv_idx = [0]
        piece = ttnn.slice(x, [0, t0, 0, 0, 0], [x.shape[0], min(t0 + chunk, T), x.shape[2], x.shape[3], x.shape[4]])
        o, nh, nw = dec.decoder(piece, lh, feat_cache=dec._feat_cache, feat_idx=dec._conv_idx, logical_w=lw)
        ttnn.deallocate(piece)
        host = ttnn.to_torch(o, mesh_composer=ttnn.ConcatMesh2dToTensor(dev, mesh_shape=(1, 1), dims=[2, 3]))
        ttnn.deallocate(o)
        frames.append(host[:, :, :nh, :nw, :3].float().clamp(-1, 1))  # B T H W C
    dec.clear_cache()
    ttnn.deallocate(x)
    return torch.cat(frames, dim=1).permute(0, 4, 1, 2, 3)


if __name__ == "__main__":
    main()
