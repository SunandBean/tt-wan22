# SPDX-License-Identifier: Apache-2.0
"""Feasibility port: one Wan 2.2 I2V A14B expert (40 blocks, dim 5120) on ONE P100a, TTNN, batch 1.

Semantics follow ComfyUI's WanModel / WanAttentionBlock (t2v cross-attention: the 2.2 I2V experts have no image
cross-attention; the start frame enters through the 36 input channels):
  block: x += gate1 * self_attn(LN(x) * (1 + scale1) + shift1)        q/k RMSNorm over the full 5120, 3D rope
         x += cross_attn(LN_affine(x), text)                            q/k RMSNorm, no rope, 512 text tokens
         x += gate2 * ffn(LN(x) * (1 + scale2) + shift2)               GELU(tanh)
  modulation rows = block.modulation + time_projection(time_embedding(sin(t))) -> host, once per sigma
Cross-attention K/V of the text depend only on the prompt: computed once per segment and kept per block.
Weights: ComfyUI scaled-fp8 file + lightx2v LoRA merged at load, bfloat8_b on the card (~15 GB per expert)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

DIM, HEADS, HEAD_DIM, FFN, LAYERS, FREQ, TEXT_DIM, IN_DIM, OUT_DIM = 5120, 40, 128, 13824, 40, 256, 4096, 36, 16
AXES = (HEAD_DIM - 4 * (HEAD_DIM // 6), 2 * (HEAD_DIM // 6), 2 * (HEAD_DIM // 6))  # (44, 42, 42)
EPS = 1e-6
TILE = 32


def round_up(n, m=TILE):
    return (n + m - 1) // m * m


# ----------------------------------------------------------------------------- host
def grid_ids(t: int, h: int, w: int) -> torch.Tensor:
    """rope ids of the (t, h/2, w/2) token grid, ComfyUI rope_encode (no rope options)."""
    tt = torch.arange(t, dtype=torch.float32)
    hh = torch.arange(h, dtype=torch.float32)
    ww = torch.arange(w, dtype=torch.float32)
    ids = torch.stack(torch.meshgrid(tt, hh, ww, indexing="ij"), dim=-1)
    return ids.reshape(-1, 3)


def rope_angles(ids: torch.Tensor) -> torch.Tensor:
    out = []
    for i, d in enumerate(AXES):
        scale = torch.linspace(0, (d - 2) / d, steps=d // 2, dtype=torch.float64)
        omega = 1.0 / (10000.0 ** scale)
        out.append(torch.outer(ids[:, i].to(torch.float64), omega))
    return torch.cat(out, dim=-1)  # [S, 64]


def cos_sin(ids, rows, dtype=torch.bfloat16):
    a = rope_angles(ids).float()
    a = torch.cat([a, torch.zeros(rows - a.shape[0], a.shape[1])]) if rows > a.shape[0] else a
    return a.cos().repeat_interleave(2, -1).to(dtype), a.sin().repeat_interleave(2, -1).to(dtype)


def sinusoidal(t: float, dim: int = FREQ) -> torch.Tensor:
    half = dim // 2
    pos = torch.tensor([t], dtype=torch.float32)
    s = torch.outer(pos, torch.pow(10000, -torch.arange(half).to(pos).div(half)))
    return torch.cat([torch.cos(s), torch.sin(s)], dim=1)


def patchify(x: torch.Tensor) -> torch.Tensor:
    """[1, 36, T, H, W] -> [T * H/2 * W/2, 144] (Conv3d (1,2,2) patch order: c, ph, pw)."""
    _, c, t, h, w = x.shape
    x = x.view(c, t, h // 2, 2, w // 2, 2).permute(1, 2, 4, 0, 3, 5)
    return x.reshape(t * (h // 2) * (w // 2), c * 4)


def unpatchify(y: torch.Tensor, t, gh, gw) -> torch.Tensor:
    """[S, 64] -> [1, 16, T, 2gh, 2gw] (ComfyUI unpatchify: feature = (p_t, p_h, p_w, c))."""
    u = y[: t * gh * gw].view(t, gh, gw, 1, 2, 2, OUT_DIM)
    u = torch.einsum("fhwpqrc->cfphqwr", u)
    return u.reshape(1, OUT_DIM, t, gh * 2, gw * 2)


class TimeConditioning:
    def __init__(self, ckpt):
        g = lambda k: ckpt.get(k, torch.float32)
        self.te = [(g("time_embedding.0.weight"), g("time_embedding.0.bias")), (g("time_embedding.2.weight"), g("time_embedding.2.bias"))]
        self.tp = (g("time_projection.1.weight"), g("time_projection.1.bias"))
        self.block_mod = [g(f"blocks.{i}.modulation")[0] for i in range(LAYERS)]  # [6, dim]
        self.head_mod = g("head.modulation")[0]  # [2, dim]

    def rows(self, timestep: float):
        """Per block {ln1_w, ln1_b, gate1, ln2_w, ln2_b, gate2} and head (w, b), bf16 [dim]."""
        e = F.linear(sinusoidal(timestep).to(torch.bfloat16).float(), *self.te[0])
        e = F.linear(F.silu(e), *self.te[1]).to(torch.bfloat16).float()  # [1, dim]
        e0 = F.linear(F.silu(e), *self.tp).to(torch.bfloat16).float().view(6, DIM)
        b = lambda v: v.to(torch.bfloat16)
        blocks = []
        for m in self.block_mod:
            m6 = (m + e0)  # shift1, scale1, gate1, shift2, scale2, gate2
            blocks.append({"ln1_w": b(1 + m6[1]), "ln1_b": b(m6[0]), "gate1": b(m6[2]),
                           "ln2_w": b(1 + m6[4]), "ln2_b": b(m6[3]), "gate2": b(m6[5])})
        hm = self.head_mod + e
        return blocks, {"w": b(1 + hm[1]), "b": b(hm[0])}


# ----------------------------------------------------------------------------- CPU reference
def _rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def _rope(x, angles):
    cos, sin = angles.cos(), angles.sin()
    x0, x1 = x[..., 0::2], x[..., 1::2]
    return torch.stack([cos * x0 - sin * x1, sin * x0 + cos * x1], dim=-1).flatten(-2)


class RefBlock:
    def __init__(self, ckpt, i):
        p = f"blocks.{i}."
        self.w = {k[len(p):]: ckpt.get(k, torch.float32) for k in ckpt.keys()
                  if k.startswith(p) and k.endswith((".weight", ".bias"))}

    def lin(self, x, n):
        return F.linear(x, self.w[n + ".weight"], self.w.get(n + ".bias"))

    @torch.no_grad()
    def __call__(self, x, rows, ctx, angles):
        f = lambda k: rows[k].float()
        S = x.shape[0]
        h = F.layer_norm(x, (DIM,), eps=EPS) * f("ln1_w") + f("ln1_b")
        q = _rms(self.lin(h, "self_attn.q"), self.w["self_attn.norm_q.weight"]).view(S, HEADS, HEAD_DIM).transpose(0, 1)
        k = _rms(self.lin(h, "self_attn.k"), self.w["self_attn.norm_k.weight"]).view(S, HEADS, HEAD_DIM).transpose(0, 1)
        v = self.lin(h, "self_attn.v").view(S, HEADS, HEAD_DIM).transpose(0, 1)
        o = F.scaled_dot_product_attention(_rope(q, angles)[None], _rope(k, angles)[None], v[None])[0]
        x = x + self.lin(o.transpose(0, 1).reshape(S, DIM), "self_attn.o") * f("gate1")
        h = F.layer_norm(x, (DIM,), self.w["norm3.weight"], self.w["norm3.bias"], eps=EPS)
        L = ctx.shape[0]
        q = _rms(self.lin(h, "cross_attn.q"), self.w["cross_attn.norm_q.weight"]).view(S, HEADS, HEAD_DIM).transpose(0, 1)
        k = _rms(self.lin(ctx, "cross_attn.k"), self.w["cross_attn.norm_k.weight"]).view(L, HEADS, HEAD_DIM).transpose(0, 1)
        v = self.lin(ctx, "cross_attn.v").view(L, HEADS, HEAD_DIM).transpose(0, 1)
        o = F.scaled_dot_product_attention(q[None], k[None], v[None])[0]
        x = x + self.lin(o.transpose(0, 1).reshape(S, DIM), "cross_attn.o")
        h = F.layer_norm(x, (DIM,), eps=EPS) * f("ln2_w") + f("ln2_b")
        y = self.lin(F.gelu(self.lin(h, "ffn.0"), approximate="tanh"), "ffn.2")
        return x + y * f("gate2")


# ----------------------------------------------------------------------------- device
def _tt():
    import ttnn
    return ttnn


@dataclass
class Prec:
    weight_dtype: object = None
    ffn_dtype: object = None  # bfloat4_b lets both experts share the card
    host_text_kv: bool = False  # cross-attention K/V of the prompt on the host: their weights never use the card
    mm_fidelity: object = None
    sdpa_fidelity: object = None


class WeightCache:
    """Device-format (tiled bfp8 / bfp4 / bf16) weights on disk: built once from the fp8 checkpoint + LoRA, then
    loaded with ttnn.load_tensor (no dequantize, merge or tile conversion on the host). A directory is only
    trusted once complete.json is written at the end of a full load; the key covers the checkpoint and LoRA files
    (size, mtime), the precision choices and the tt-metal build."""

    def __init__(self, root, key: str):
        import pathlib
        self.dir = pathlib.Path(root) / key
        self.ready = (self.dir / "complete.json").exists()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = self.built = 0

    def get(self, name, make, dtype, dev, mem):
        ttnn = _tt()
        path = self.dir / f"{name}.{str(dtype).split('.')[-1]}.tensorbin"
        if self.ready and path.exists():
            self.hits += 1
            t = ttnn.load_tensor(str(path))
            return ttnn.to_device(t, dev, memory_config=mem)
        host = ttnn.from_torch(make().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT)
        ttnn.dump_tensor(str(path), host)
        self.built += 1
        return ttnn.to_device(host, dev, memory_config=mem)

    def finish(self, info: dict):
        import json
        if not self.ready:
            (self.dir / "complete.json").write_text(json.dumps(info, indent=1))
            self.ready = True


def cache_key(files, prec_desc: str) -> str:
    import hashlib, os
    h = hashlib.sha256()
    for f in files:
        st = os.stat(f)
        h.update(f"{os.path.basename(f)}:{st.st_size}:{st.st_mtime_ns};".encode())
    h.update(prec_desc.encode())
    h.update(os.environ.get("TT_METAL_COMMIT_SHA_OR_TAG", "").encode())
    return h.hexdigest()[:20]


class TTWanExpert:
    def __init__(self, dev, ckpt, n_blocks: int = LAYERS, prec: Optional[Prec] = None, cache_dir: Optional[str] = None):
        ttnn = _tt()
        self.ttnn, self.dev = ttnn, dev
        p = prec or Prec()
        wd = p.weight_dtype or ttnn.bfloat8_b
        fd = p.ffn_dtype or wd
        self.MEM = ttnn.DRAM_MEMORY_CONFIG
        self.cache = None
        if cache_dir:
            desc = f"w={wd};ffn={fd};host_kv={p.host_text_kv};blocks={n_blocks}"
            files = [ckpt.path] + ([ckpt.lora_path] if ckpt.lora_path else [])
            self.cache = WeightCache(cache_dir, cache_key(files, desc))

        def dev_t(key, make, d):
            if self.cache is not None:
                return self.cache.get(key, make, d, dev, self.MEM)
            return ttnn.from_torch(make().contiguous(), dtype=d, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=self.MEM)

        g = lambda k: ckpt.get(k, torch.bfloat16)
        row = lambda key: dev_t(key, lambda: g(key).to(torch.bfloat16).reshape(1, 1, 1, -1), ttnn.bfloat16)
        mm = lambda n, d=wd: (dev_t(n + ".weight", lambda: g(n + ".weight").t(), d), row(n + ".bias"))
        self.ckpt, self.host_text_kv = ckpt, p.host_text_kv
        names = ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o", "cross_attn.q", "cross_attn.o"]
        if not self.host_text_kv:
            names += ["cross_attn.k", "cross_attn.v"]
        self.blocks = []
        for i in range(n_blocks):
            b = f"blocks.{i}."
            blk = {n: mm(b + n) for n in names}
            blk["ffn.0"], blk["ffn.2"] = mm(b + "ffn.0", fd), mm(b + "ffn.2", fd)
            for n in ("self_attn.norm_q", "self_attn.norm_k", "cross_attn.norm_q", "cross_attn.norm_k"):
                blk[n] = row(b + n + ".weight")
            blk["norm3"] = (row(b + "norm3.weight"), row(b + "norm3.bias"))
            self.blocks.append(blk)
        self.patch = (dev_t("patch_embedding.weight", lambda: g("patch_embedding.weight").reshape(DIM, IN_DIM * 4).t(),
                            ttnn.bfloat16), row("patch_embedding.bias"))
        if not self.host_text_kv:
            self.text = [mm("text_embedding.0"), mm("text_embedding.2")]
        self.head = (dev_t("head.head.weight", lambda: g("head.head.weight").t(), ttnn.bfloat16), row("head.head.bias"))
        self.trans_mat = ttnn.from_torch(_rot_mat(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                         memory_config=self.MEM)
        if self.cache is not None:
            self.cache.finish({"checkpoint": ckpt.path, "blocks": n_blocks, "tensors": self.cache.built})
        self.grid = dev.compute_with_storage_grid_size()
        ck = lambda fid: ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=fid, math_approx_mode=False,
                                                               fp32_dest_acc_en=True, packer_l1_acc=False)
        self.ck_mm = ck(p.mm_fidelity or ttnn.MathFidelity.HiFi2)
        self.ck_hi = ck(ttnn.MathFidelity.HiFi4)
        self.ck_sdpa = ck(p.sdpa_fidelity or ttnn.MathFidelity.HiFi2)  # HiFi2 + fp32 acc: 9% faster, PCC 0.99974
        self.gelu = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH)
        try:
            from models.tt_dit.utils.matmul import get_matmul_config
            self._get_cfg = get_matmul_config
        except ImportError:
            self._get_cfg = None
        self._cfg_cache = {}

    def _mm(self, x, wb, act=None):
        ttnn = self.ttnn
        w, b = wb
        M, K, N = x.shape[-2], x.shape[-1], w.shape[-1]
        key = (M, K, N)
        if key not in self._cfg_cache:
            self._cfg_cache[key] = self._get_cfg(M, K, N, self.grid)
        out = ttnn.experimental.minimal_matmul(x, w, bias_tensor=b, fused_activation=act, config=self._cfg_cache[key],
                                               compute_kernel_config=self.ck_mm, dtype=ttnn.bfloat16, memory_config=self.MEM)
        return out if len(out.shape) == 4 else ttnn.reshape(out, [1, 1] + list(out.shape)[-2:])

    def to_dev(self, t, dtype=None):
        ttnn = self.ttnn
        return ttnn.from_torch(t.to(torch.bfloat16).contiguous(), dtype=dtype or ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=self.dev, memory_config=self.MEM)

    def rows_to_dev(self, blocks, head):
        r = lambda v: self.to_dev(v.reshape(1, 1, 1, -1))
        return [{k: r(v) for k, v in b.items()} for b in blocks], {k: r(v) for k, v in head.items()}

    @torch.no_grad()
    def text_kv_host(self, umt5: torch.Tensor):
        """text_kv computed on the host in float32 (weights read from the checkpoint per block), uploaded as heads."""
        g = lambda k: self.ckpt.get(k, torch.float32)
        h = F.gelu(F.linear(umt5.float(), g("text_embedding.0.weight"), g("text_embedding.0.bias")), approximate="tanh")
        ctx = F.linear(h, g("text_embedding.2.weight"), g("text_embedding.2.bias"))
        L = ctx.shape[0]
        out = []
        for i in range(len(self.blocks)):
            b = f"blocks.{i}.cross_attn."
            k = _rms(F.linear(ctx, g(b + "k.weight"), g(b + "k.bias")), g(b + "norm_k.weight"))
            v = F.linear(ctx, g(b + "v.weight"), g(b + "v.bias"))
            heads = lambda t: self.to_dev(t.view(L, HEADS, HEAD_DIM).transpose(0, 1)[None])
            out.append((heads(k), heads(v)))
        return out

    def text_kv(self, umt5: torch.Tensor):
        """[512, 4096] text encoder output -> per block (k, v) heads for cross-attention (kept for the segment)."""
        if self.host_text_kv:
            return self.text_kv_host(umt5)
        ttnn = self.ttnn
        c = self.to_dev(umt5.reshape(1, 1, umt5.shape[0], -1))
        h = self._mm(c, self.text[0], act=self.gelu)
        ctx = self._mm(h, self.text[1])
        out = []
        for blk in self.blocks:
            k = ttnn.rms_norm(self._mm(ctx, blk["cross_attn.k"]), epsilon=EPS, weight=blk["cross_attn.norm_k"],
                              compute_kernel_config=self.ck_hi)
            v = self._mm(ctx, blk["cross_attn.v"])
            out.append((self._heads(k), self._heads(v)))
        return out

    def _heads3(self, q, k, v, keep=3):
        """Fused head split of three [1, 1, S, dim] tensors (concat + nlp_create_qkv_heads): 11x faster than
        reshape + permute (bench_attn.py). keep=1 returns only the first (cross-attention query)."""
        ttnn = self.ttnn
        qkv = ttnn.concat([q, k, v], dim=-1, memory_config=self.MEM)
        for t in {id(q): q, id(k): k, id(v): v}.values():
            ttnn.deallocate(t)
        qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=HEADS, num_kv_heads=HEADS,
                                                            transpose_k_heads=False, memory_config=self.MEM)
        ttnn.deallocate(qkv)
        if keep == 1:
            ttnn.deallocate(kh)
            ttnn.deallocate(vh)
            return qh
        return qh, kh, vh

    def _heads(self, x):
        ttnn = self.ttnn
        S = x.shape[-2]
        y = ttnn.reshape(x, [1, S, HEADS, HEAD_DIM])
        return ttnn.permute(y, (0, 2, 1, 3))

    def _attn(self, q, k, v, chunk, mask=None):
        ttnn = self.ttnn
        cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=self.grid, q_chunk_size=chunk, k_chunk_size=chunk,
                                     exp_approx_mode=False)
        a = ttnn.transformer.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False, scale=HEAD_DIM ** -0.5,
                                                          program_config=cfg, compute_kernel_config=self.ck_sdpa,
                                                          memory_config=self.MEM)
        return ttnn.transformer.concatenate_heads(a, memory_config=self.MEM)

    def block(self, x, blk, rows, kv, cos, sin, S):
        ttnn, M = self.ttnn, self.MEM
        h = ttnn.layer_norm(x, epsilon=EPS, weight=rows["ln1_w"], bias=rows["ln1_b"], compute_kernel_config=self.ck_hi)
        q = ttnn.rms_norm(self._mm(h, blk["self_attn.q"]), epsilon=EPS, weight=blk["self_attn.norm_q"], compute_kernel_config=self.ck_hi)
        k = ttnn.rms_norm(self._mm(h, blk["self_attn.k"]), epsilon=EPS, weight=blk["self_attn.norm_k"], compute_kernel_config=self.ck_hi)
        v = self._mm(h, blk["self_attn.v"])
        ttnn.deallocate(h)
        q, k, v = self._heads3(q, k, v)
        qr = ttnn.experimental.rotary_embedding_llama(q, cos, sin, self.trans_mat, is_decode_mode=False, compute_kernel_config=self.ck_hi)
        kr = ttnn.experimental.rotary_embedding_llama(k, cos, sin, self.trans_mat, is_decode_mode=False, compute_kernel_config=self.ck_hi)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        if self.n_valid < S:  # pad keys hidden through the logical length (SDPA masks the tile padding)
            kn = ttnn.slice(kr, [0, 0, 0, 0], [1, HEADS, self.n_valid, HEAD_DIM])
            vn = ttnn.slice(v, [0, 0, 0, 0], [1, HEADS, self.n_valid, HEAD_DIM])
            ttnn.deallocate(kr)
            ttnn.deallocate(v)
            kr, v = kn, vn
        o = self._attn(qr, kr, v, self.chunk, self.mask)
        for t in (qr, kr, v):
            ttnn.deallocate(t)
        y = self._mm(o, blk["self_attn.o"])
        ttnn.deallocate(o)
        x2 = ttnn.add(x, ttnn.multiply(y, rows["gate1"]), memory_config=M)
        ttnn.deallocate(y)
        ttnn.deallocate(x)
        h = ttnn.layer_norm(x2, epsilon=EPS, weight=blk["norm3"][0], bias=blk["norm3"][1], compute_kernel_config=self.ck_hi)
        q = ttnn.rms_norm(self._mm(h, blk["cross_attn.q"]), epsilon=EPS, weight=blk["cross_attn.norm_q"], compute_kernel_config=self.ck_hi)
        ttnn.deallocate(h)
        qh = self._heads3(q, q, q, keep=1)
        o = self._attn(qh, kv[0], kv[1], self.chunk_x)
        y = self._mm(o, blk["cross_attn.o"])
        ttnn.deallocate(o)
        x3 = ttnn.add(x2, y, memory_config=M)
        ttnn.deallocate(y)
        ttnn.deallocate(x2)
        h = ttnn.layer_norm(x3, epsilon=EPS, weight=rows["ln2_w"], bias=rows["ln2_b"], compute_kernel_config=self.ck_hi)
        m = self._mm(h, blk["ffn.0"], act=self.gelu)
        ttnn.deallocate(h)
        y = self._mm(m, blk["ffn.2"])
        ttnn.deallocate(m)
        x4 = ttnn.add(x3, ttnn.multiply(y, rows["gate2"]), memory_config=M)
        ttnn.deallocate(y)
        ttnn.deallocate(x3)
        return x4

    def forward(self, patches: torch.Tensor, rows, head_rows, kv, cos, sin, n_valid, taps=None, mask=None):
        """patches [S, 144] (S a tile multiple) -> velocity tokens [n_valid, 64] float32. mask: additive
        [1, 1, S, S] bias hiding the pad keys (None when S == n_valid)."""
        ttnn = self.ttnn
        self.mask = mask
        self.n_valid = n_valid
        S = patches.shape[0]
        self.chunk = next(c for c in (256, 128, 64, 32) if S % c == 0)
        self.chunk_x = next(c for c in (256, 128, 64, 32) if S % c == 0)
        x = self._mm(self.to_dev(patches.reshape(1, 1, S, -1)), self.patch)
        for i, blk in enumerate(self.blocks):
            x = self.block(x, blk, rows[i], kv[i], cos, sin, S)
            if taps is not None:
                taps.append(ttnn.to_torch(x)[0, 0, :n_valid].float())
        h = ttnn.layer_norm(x, epsilon=EPS, weight=head_rows["w"], bias=head_rows["b"], compute_kernel_config=self.ck_hi)
        ttnn.deallocate(x)
        out = self._mm(h, self.head)
        v = ttnn.to_torch(out)[0, 0, :n_valid].float()
        ttnn.deallocate(out)
        ttnn.deallocate(h)
        return v


def _rot_mat():
    m = torch.zeros(1, 1, TILE, TILE)
    m[..., torch.arange(0, TILE, 2), torch.arange(1, TILE, 2)] = 1.0
    m[..., torch.arange(1, TILE, 2), torch.arange(0, TILE, 2)] = -1.0
    return m


def key_mask(n_valid, S):
    m = torch.zeros(S)
    m[n_valid:] = -1e9
    return m.expand(S, S).reshape(1, 1, S, S).to(torch.bfloat16)


def sigmas_sd3(steps=4, shift=5.0):
    """ModelSamplingSD3(shift, multiplier 1000) table, "simple" scheduler picks, trailing 0."""
    t = torch.arange(1, 1001, 1) / 1000
    table = shift * t / (1 + (shift - 1) * t)
    ss = len(table) / steps
    return [float(v) for v in torch.FloatTensor([float(table[-(1 + int(x * ss))]) for x in range(steps)] + [0.0])]
