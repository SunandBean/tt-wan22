# tt-wan22

**Wan 2.2 image-to-video A14B** with the lightx2v 4-step LoRA, ported to a single
**Tenstorrent Blackhole p100a**. Both experts and the Wan 2.1 VAE encoder and decoder run on the card.

Model card, demo video and the full numbers:
**[sunandbean/wan2.2-i2v-a14b-p100a](https://huggingface.co/sunandbean/wan2.2-i2v-a14b-p100a)**

**What this port adds** — both 14B experts on a single 29.9 GB card by swapping them in and out, a streamed VAE encode and decode so 81 frames never sit on the card at once, the UMT5-XXL host encoder, and the segment renderer.
**What it builds on** — [tt-metal](https://github.com/tenstorrent/tt-metal)'s `tt_dit` `WanEncoder` / `WanDecoder`, imported from the runtime rather than vendored (Apache-2.0), and the fp8 + LoRA loader from the sibling [tt-qwen-image-edit](https://github.com/SunandBean/tt-qwen-image-edit) port.

A 480p 81-frame segment takes **96 s** in steady state, against **84 s** for the same segment on an
RTX 5070 Ti. 720p takes 347 s against 312 s.

## Why this exists

tt-metal's own tt_dit Wan pipeline supports only 4–8 chip Blackhole meshes. This is one expert rewritten
for a single chip, using the same operation set as
[tt-qwen-image-edit](https://github.com/SunandBean/tt-qwen-image-edit).

## How two 14.3 B experts fit in 30 GB

1. **Cross-attention K/V weights stay on the host** — about 4.3 GB used exactly once per prompt.
   Computing the text K/V on the host in float32 and uploading only that drops device residency from
   23.3 GB to 19.0 GB, which is what makes room for the VAE decoder.
2. **Streamed VAE** — `encode_streamed` sends 4 frames at a time instead of all 81 (4.1 GB);
   `decode_streamed` brings each output piece back to the host instead of gathering 81 frames on the card.
3. **Padded keys are truncated, not masked** — a 32768² bf16 mask is 2 GB and OOMs beside the experts.
   Passing K/V at the logical length 32,760 gives the same result (PCC 0.99978 against the mask).

At 720p (75,600 tokens) even that is not enough, so the renderer holds one expert at a time and swaps
at step 2 from the weight cache, 12 s each.

## Tuning

Profiling showed SDPA 42%, **reshape + permute 31%** (head splitting), matmul 18%.

- One `concat` + `nlp_create_qkv_heads` instead of per-tensor reshape + permute: **0.129 s → 0.0115 s**.
- SDPA HiFi4 → HiFi2 with fp32 accumulation: 9% (PCC 0.99974). It was already near the operation limit.
- Step time **22.2 s → 15.3 s**.

A converted-weight disk cache (`ttnn.dump_tensor` after fp8 dequantise, LoRA merge and tile conversion)
takes expert load from 157 s to **12.2 s**, with bit-identical output.

## Install

```bash
pip install -e .
```

Needs a tt-metal / ttnn environment that also provides `models.tt_dit` with the Wan VAE and its conv3d
kernels — the tt-inference-server 0.18 image has one. `deploy/run.sh` is the container runner.

```python
from tt_wan22 import Renderer, open_dev, close_dev

dev = open_dev()
try:
    r = Renderer(dev, progress=lambda **kw: None, swap=False)
    frames, timing = r.segment(prompt, start, None, 832, 480, length=81, seed=42,
                               progress=lambda **kw: None)
finally:
    close_dev(dev)
```

Or as a batch renderer over a `job.json`: `bash deploy/run.sh <job dir> <images dir>`. See
[`PYTHON.md`](PYTHON.md).

## Layout

| Path | |
|---|---|
| `tt_wan22/` | the port |
| `deploy/run.sh` | the container runner (tt-inference-server image, memory cap) |
| `tt-model.yaml` | the container manifest the published image is built from — `tt-model package --container tt-model.yaml` |
| `PYTHON.md` | API reference, environment variables, the job-file contract |
| `experiments/` | the bring-up and benchmarking scripts behind the published numbers — see [`experiments/README.md`](experiments/README.md) for how to run them |

## Licence

Apache-2.0. The weights
([Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged),
Apache-2.0 repackagings of [Wan-AI/Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)) are
not redistributed here.
