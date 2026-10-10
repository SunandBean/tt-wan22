# SPDX-License-Identifier: Apache-2.0
"""Wan 2.1 VAE on the P100a with tt-metal's tt_dit WanEncoder / WanDecoder (mesh 1x1), sized to share the card with
both Wan experts: the encoder gets the conditioning video one time chunk at a time and the decoder hands every
finished chunk to the host, so neither holds the 81 frames on the card (experiments/wan22 NOTES, stage 2)."""
import os

import torch
import ttnn
from diffusers import AutoencoderKLWan
from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
from safetensors.torch import load_file
from models.tt_dit.models.vae.vae_wan2_1 import WanDecoder, WanEncoder
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.conv3d import conv_pad_height, conv_pad_in_channels, conv_pad_width
from models.tt_dit.utils.tensor import typed_tensor_2dshard

MEAN = torch.tensor([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                     0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]).view(1, 16, 1, 1, 1)
STD = torch.tensor([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]).view(1, 16, 1, 1, 1)


def torch_vae(path):
    v = AutoencoderKLWan()
    v.load_state_dict(convert_wan_vae_to_diffusers(load_file(path)))
    return v.eval()


def open_dev():
    """The card with fabric set up (tt_dit's CCL manager expects it even on one chip)."""
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    return ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=int(os.environ.get("L1_SMALL", "32768")))


def close_dev(dev):
    ttnn.close_mesh_device(dev)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


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


class WanVAE:
    def __init__(self, dev, path):
        tv = torch_vae(path)
        self.dev = dev
        self.enc = build(WanEncoder, dev, tv.config, in_channels=3, dtype=ttnn.bfloat16)
        self.enc.load_torch_state_dict(tv.state_dict())
        self.dec = build(WanDecoder, dev, tv.config, out_channels=3, dtype=ttnn.bfloat16)
        self.dec.load_torch_state_dict(tv.state_dict())

    def encode(self, video_BCTHW: torch.Tensor) -> torch.Tensor:
        """[1, 3, F, H, W] in [-1, 1] -> latent mean [1, 16, T, H/8, W/8] float32 (VAE space)."""
        out, nh, nw = encode_streamed(self.enc, video_BCTHW, self.dev)
        lat = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMesh2dToTensor(self.dev, mesh_shape=(1, 1), dims=[3, 4]))
        ttnn.deallocate(out)
        return lat[:, :16, :, :nh, :nw].float()

    def decode(self, z_BCTHW: torch.Tensor) -> torch.Tensor:
        """VAE-space latent -> [1, 3, F, H, W] float32 in [-1, 1]."""
        return decode_streamed(self.dec, z_BCTHW, self.dev)
