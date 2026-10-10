# SPDX-License-Identifier: Apache-2.0
"""UMT5-XXL prompt encoder on the host (CPU, float32), the conditioning of the reference ComfyUI Wan graph
(CLIPLoader type "wan" + CLIPTextEncode), without ComfyUI:
  * SentencePiece model stored in the checkpoint ("spiece_model"); ids + EOS (1), padded with 0 to 512
  * 24 layers, per-layer relative position bias (UMT5), T5 RMS layer norm, unscaled attention, gated GELU(tanh)
  * final layer norm; rows past the prompt are zero (ComfyUI zero_out_masked)
Padded keys are masked in ComfyUI, so the valid rows only depend on the valid tokens: the encoder runs on those
rows alone, one layer at a time from the fp8 checkpoint (a few hundred MB of host memory)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .comfy_weights import ComfyCheckpoint

D_MODEL, HEADS, D_KV, LAYERS, EPS, MAX_TOKENS = 4096, 64, 64, 24, 1e-6, 512
BUCKETS, MAX_DISTANCE = 32, 128


def _bucket(rel: torch.Tensor) -> torch.Tensor:
    """T5 bidirectional relative position bucket (32 buckets, max distance 128)."""
    nb = BUCKETS // 2
    out = (rel > 0).long() * nb
    rel = rel.abs()
    max_exact = nb // 2
    large = max_exact + (torch.log(rel.float() / max_exact) / math.log(MAX_DISTANCE / max_exact) * (nb - max_exact)).long()
    large = torch.minimum(large, torch.full_like(large, nb - 1))
    return out + torch.where(rel < max_exact, rel, large)


def _norm(x, w):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS))


class UMT5Encoder:
    def __init__(self, path: str, threads: int = 0):
        import sentencepiece

        self.ck = ComfyCheckpoint(path)
        proto = self.ck.get("spiece_model", None).numpy().tobytes()
        self.sp = sentencepiece.SentencePieceProcessor(model_proto=proto, add_bos=False, add_eos=False)
        self.threads = threads

    def tokens(self, prompt: str):
        ids = self.sp.encode(prompt)[: MAX_TOKENS - 1] + [1]
        return ids

    @torch.no_grad()
    def __call__(self, prompt: str) -> torch.Tensor:
        """prompt -> [512, 4096] float32 conditioning."""
        prev = torch.get_num_threads()
        if self.threads:
            torch.set_num_threads(self.threads)
        try:
            return self._encode(self.tokens(prompt))
        finally:
            torch.set_num_threads(prev)

    def _encode(self, ids):
        g = lambda k: self.ck.get(k, torch.float32)
        L = len(ids)
        sl = self.ck._f.get_slice("shared.weight")
        x = torch.stack([sl[i: i + 1][0] for i in ids]).float()  # [L, 4096]
        pos = torch.arange(L)
        buckets = _bucket(pos[None, :] - pos[:, None])
        for i in range(LAYERS):
            p = f"encoder.block.{i}.layer."
            h = _norm(x, g(p + "0.layer_norm.weight"))
            q, k, v = (F.linear(h, g(p + f"0.SelfAttention.{n}.weight")).view(L, HEADS, D_KV).transpose(0, 1) for n in "qkv")
            bias = g(p + "0.SelfAttention.relative_attention_bias.weight")[buckets].permute(2, 0, 1)  # [H, L, L]
            a = torch.softmax(q @ k.transpose(1, 2) + bias, dim=-1) @ v
            x = x + F.linear(a.transpose(0, 1).reshape(L, -1), g(p + "0.SelfAttention.o.weight"))
            h = _norm(x, g(p + "1.layer_norm.weight"))
            m = F.gelu(F.linear(h, g(p + "1.DenseReluDense.wi_0.weight")), approximate="tanh") * \
                F.linear(h, g(p + "1.DenseReluDense.wi_1.weight"))
            x = x + F.linear(m, g(p + "1.DenseReluDense.wo.weight"))
        x = _norm(x, g("encoder.final_layer_norm.weight"))
        out = torch.zeros(MAX_TOKENS, D_MODEL)
        out[:L] = x
        return out
