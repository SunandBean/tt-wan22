# tt_wan22 — Python reference

Wan 2.2 I2V A14B with the lightx2v 4-step LoRA on one Tenstorrent Blackhole p100a. Both experts and
the Wan 2.1 VAE encoder and decoder run on the card; the UMT5-XXL prompt encoder runs on the host.

## Install

```bash
pip install -e .
```

This needs a tt-metal / ttnn environment that also provides `models.tt_dit` with the Wan VAE and its
conv3d kernels — the tt-inference-server 0.18 image has one. `deploy/run.sh` in the GitHub repo
(github.com/SunandBean/tt-wan22) is a wrapper that runs the renderer inside it; it is not part of the
installable package.

## Weights

Not bundled. `COMFY_MODELS` (default `/comfy`) points at a ComfyUI `models` directory holding the four
files listed in the model card, all from
[`Comfy-Org/Wan_2.2_ComfyUI_Repackaged`](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged).

## Environment

| Variable | |
|---|---|
| `COMFY_MODELS` | the ComfyUI `models` directory (default `/comfy`) |
| `JOB_DIR` | the job directory the batch renderer reads and writes (default `/job`) |
| `WAN_CACHE` | where converted weights are cached — 9.4 GB per expert; load drops from 157 s to 12.2 s |
| `TT_METAL_CACHE` | compiled-kernel cache; without it each new container recompiles (~75 s) |
| `THREADS` | host worker threads (default 12) |
| `L1_SMALL` | L1 small allocation (default 32768) |

## API

### `open_dev()` / `close_dev(dev)`

Opens a 1×1 mesh with `FabricConfig.FABRIC_1D` — tt_dit's CCL manager expects fabric even on one chip.

### `Renderer(dev, progress, swap=False)`

Loads both experts, the VAE and the UMT5 encoder. `progress(**kw)` is called through the run with
keyword state. `swap=True` holds **one** expert at a time and swaps at step 2 from the weight cache;
use it above about 40,000 tokens (720p). `.load_s` reports the load time.

### `Renderer.segment(prompt, start, end, w, h, length, seed, progress) -> (frames, timing)`

| Argument | Meaning |
|---|---|
| `start`, `end` | `[H, W, 3]` float tensors in `[0, 1]`, or `None`. `end` makes it a first/last-frame segment. |
| `length` | frame count — 81 is one 5-second segment at 16 fps |
| `seed` | matches the ComfyUI graph |

Returns `frames` as a `[length, H, W, 3]` uint8 array and a timing dict.

Text K/V is cached per prompt, so later segments of the same shot skip the 16 s host encode.

### `Experts(dev, swap, progress)`

The expert pair and the swap machinery, if you want to drive sampling yourself.

## Batch renderer

`python -m tt_wan22.render` reads `$JOB_DIR/job.json`:

```json
{"gen_w": 832, "gen_h": 480, "fps": 16,
 "segments": [{"prompt": "...", "length": 81, "seed": 42, "start": "/images/first.png", "end": null}]}
```

`start: null` continues from the previous segment's last frame. Per segment it writes `i.frames.rgb`
(raw uint8 RGB), `i_last.png` and `i.json`; `progress.json` follows the run and `result.json` ends it.

## Modules

| Module | Role |
|---|---|
| `render.py` | `Renderer`, `Experts`, conditioning, and the job-file entry point |
| `wan_dit.py` | one expert on the card: `TTWanExpert`, `Prec`, `WeightCache`, `TimeConditioning` |
| `vae.py` | the Wan 2.1 VAE on the card over tt_dit, with `encode_streamed` / `decode_streamed` |
| `umt5.py` | UMT5-XXL prompt encoder on the host, one layer at a time from the fp8 checkpoint |
| `comfy_weights.py` | fp8 + per-tensor scale loading and LoRA merge; shared with the tt-qwen-image-edit port |
| `checks/check_umt5.py` | UMT5 output against ComfyUI |
