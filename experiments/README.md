# Bring-up and benchmarking scripts

Roughly in the order they were used.

| Script | What it does |
|---|---|
| `tt_device.py` | Opens the card with the fabric configuration tt_dit's CCL manager expects |
| `comfy_weights.py` | The fp8 + LoRA loader (the copy that became `tt_wan22/comfy_weights.py`) |
| `wan_tt.py` | The first single-chip expert — the feasibility test for the DiT |
| `wan_check.py` | One expert's forward against the ComfyUI module |
| `sdpa_pad_check.py` | That truncating K/V to the logical length equals an additive padding mask |
| `host_cond.py` | UMT5 text encoding and the I2V conditioning with ComfyUI's own code |
| `vae_cpu.py`, `vae_wan.py` | The CPU VAE reference (147 s to encode, 282 s to decode 81 frames) |
| `vae_tt.py` | The tt_dit Wan VAE on the card, and the streamed encode / decode that made it fit |
| `wan_segment.py`, `segment_tt.py` | A whole segment — first with the CPU VAE, then entirely on the card |
| `comfy_segment.py` | The same segment through ComfyUI on the GPU, for the comparison |
| `profile_step.py`, `profile_vae.py` | Where the time goes: SDPA 42%, reshape+permute 31%, matmul 18% |
| `bench_attn.py` | The head-split rewrite and the SDPA chunk / precision sweep |
| `mux.py` | Builds the side-by-side comparison frames |
| `run_tt.sh` | Runs any of these in the tt-inference-server image |

The device scripts need a p100a. `run_tt.sh` caps the container's memory at a value chosen for the host
this was developed on (`MEMORY=24g`); raise or drop it to suit yours.

These scripts import each other, so run them with this directory on `PYTHONPATH` (running one from
inside `experiments/` is enough). `vae_tt.py` and `wan_tt.py` also import tt-metal's `models/` tree and
`host_cond.py` needs a ComfyUI checkout; both belong on `PYTHONPATH` too. Unlike the other ports in
this family, nothing here imports a module that is not in this repository.
