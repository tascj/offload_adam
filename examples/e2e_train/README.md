# End-to-end training example

A single-file script (`train.py`) that trains a HuggingFace causal LM on
synthetic data using this library's `Adam` or `OffloadAdam`. It reports
per-step timing and a JSON summary that is easy to aggregate.

## Usage

Install the extra dependencies the script needs (`transformers`,
`accelerate`, `liger-kernel`):

```bash
uv sync --group examples
```

```bash
# OffloadAdam (default)
uv run python examples/e2e_train/train.py \
    --tokens-per-sample 4096 --grad-accum-steps 4 --steps 10 --warmup-steps 2

# Plain Adam (no offloading)
uv run python examples/e2e_train/train.py --no-enable-offloading \
    --mode stochastic_rounding --tokens-per-sample 4096
```

Run with `--help` to see all flags. Defaults: `Qwen/Qwen3-8B`, bf16,
sdpa attention, gradient checkpointing on, Liger fused linear CE on.

## Benchmarks

### Environment

Single GPU running Qwen3-8B in bf16 with sdpa attention, gradient
checkpointing, and Liger fused linear cross-entropy.

| Component |                                                       |
| --------- | ----------------------------------------------------- |
| GPU       | NVIDIA RTX PRO 6000 Blackwell Workstation (96 GB)     |
| Host link | PCIe 5.0 x16 (≈64 GB/s per direction, theoretical)    |

Each run does 1 warmup step + 4 measured steps; throughput is averaged
over the measured steps. Loss values are ignored — they are meaningless
on random tokens.

### Varying sequence length (grad_accum=1, mode=stochastic_rounding)

| seq_len | no-offload tok/s | offload tok/s | offload overhead | no-offload peak (GB) | offload peak (GB) |
| ------: | ---------------: | ------------: | ---------------: | -------------------: | ----------------: |
|   1 024 |            1 959 |           851 |             -57% |                61.1 |              20.2 |
|   2 048 |            2 980 |         1 662 |             -44% |                61.1 |              20.5 |
|   4 096 |            3 881 |         2 844 |             -27% |                61.2 |              21.2 |
|   8 192 |            4 201 |         3 904 |              -7% |                61.3 |              22.4 |
|  16 384 |            3 812 |         3 741 |              -2% |                62.2 |              24.9 |
|  32 768 |            3 052 |         3 057 |              ≈0% |                64.8 |              30.0 |

The offloading H2D/D2H traffic is overlapped with forward/backward. At short
sequences (<= 2048) the optimizer.step cost dominates and is not fully hidden,
so throughput drops significantly. From 8192 onward the overhead is
negligible. Peak GPU memory stays around ~21–30 GB with offloading versus
~61 GB without, a ~40 GB saving that is roughly constant across seq_len.

### Varying gradient accumulation (seq_len=4096, mode=stochastic_rounding)

| grad_accum | no-offload tok/s | offload tok/s | offload overhead | no-offload peak (GB) | offload peak (GB) |
| ---------: | ---------------: | ------------: | ---------------: | -------------------: | ----------------: |
|          1 |            3 902 |         2 834 |             -27% |                61.1 |              21.2 |
|          2 |            3 938 |         3 028 |             -23% |                66.9 |              21.2 |
|          4 |            3 924 |         3 285 |             -16% |                66.9 |              21.2 |
|          8 |            3 876 |         3 440 |             -11% |                66.9 |              21.2 |
|         16 |            3 832 |         3 484 |              -9% |                66.9 |              21.2 |
|         32 |            3 808 |         3 499 |              -8% |                66.9 |              21.2 |

Accumulation microsteps and the final (optimizer-step) microstep have very
different PCIe traffic:

- **Accumulation microstep**: H2D the previously-offloaded accumulated
  gradient, add the new gradient on GPU, D2H the sum back. Roughly
  `2 × param_bytes` (bf16 → ~2 bytes/param).
- **Final microstep**: the same gradient traffic plus H2D of every
  optimizer state (momentum + variance, and the fp32 master copy if
  enabled) for the step, and D2H of the updated states. Roughly
  `6 × param_bytes` for `stochastic_rounding` and `~10 × param_bytes`
  for `fp32_master`.

At ga=1 every microstep pays the heavy final-microstep traffic, so
overhead is -27%. As ga grows, only 1/ga microsteps is the heavy one
and the others overlap with forward/backward almost for free, so
overhead falls to -8% at ga=32. The residual -8% is the portion of
the final-microstep state traffic that cannot be hidden behind the
backward of the last microstep.

Offloading also keeps peak memory flat at 21.2 GB regardless of ga,
because gradients and optimizer state live on CPU. Without offloading,
gradients persist on the GPU across microsteps and peak jumps from
61 GB (ga=1) to 67 GB (ga≥2).

### Varying rounding mode (seq_len=4096, grad_accum=1, offloading on)

|                mode | tok/s | peak (GB) |
| ------------------: | ----: | --------: |
| stochastic_rounding | 2 816 |      21.2 |
|         fp32_master | 1 653 |      22.3 |

`fp32_master` keeps an fp32 master copy of the parameters on the host
and transfers it in both directions on every optimizer step, raising
the final-microstep PCIe traffic from ~6× to ~10× param bytes. At ga=1
every microstep is a final one, so the ~1.67× traffic increase maps
almost directly to `fp32_master` being ~1.70× slower — PCIe dominates
this mode.
