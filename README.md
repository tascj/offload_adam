# Offload Adam

Adam optimizer that keeps gradients and optimizer states in pinned host
memory and streams them to the GPU only when they are needed for the step.
Trades a small amount of throughput (when the step doesn't fully overlap
backward) for the ability to do full-parameter training of much larger
models on a single GPU.

## Install

```bash
uv sync
```

## Usage

```python
from offload_adam import OffloadAdam

model = build_model().bfloat16().cuda()

optimizer = OffloadAdam(
    model,                         # pass the module, not its parameters
    lr=1e-4,
    weight_decay=0.01,
    mode="fp32_master",            # see "Modes" below
    decoupled_weight_decay=True,   # AdamW
    max_grad_norm=1.0,             # optional L2 global-norm clipping
)

for step_batches in dataloader:    # outer iter = one optimizer step
    for i, batch in enumerate(step_batches):
        # The hook-driven step needs to know which microbatch is the last.
        optimizer.ready_for_optimizer_step = i == len(step_batches) - 1
        loss = model(**batch).loss / len(step_batches)
        loss.backward()
    optimizer.step()               # no-op without clipping (step ran in
                                    # the backward hook); runs the chunked
                                    # step when max_grad_norm is set
```

`offload_adam.Adam` is the same Adam kernel without offloading, useful as a
baseline for bf16 params (which `torch.optim.AdamW` does not handle).

### Modes

| `mode` | states stored on host | use when |
|---|---|---|
| `stochastic_rounding` (default) | bf16 grad + bf16 exp_avg / exp_avg_sq | smallest pinned footprint, slight noise from rounding |
| `fp32_master` | adds fp32 master copy of params | bit-for-bit AdamW math, ~2× host memory |
| `fp31_master` / `fp32_master_custom_rounding` | bf16 states + int16 rounding error | midway between the two above |

### Useful knobs

- `numa_node`: `'auto'` (default) binds pinned allocations to the GPU's NUMA
  node. Pass an int to override or `None` to disable.
- `inplace_param_threshold`: params smaller than this stay fully on GPU.
  Default 1M elements (covers norm weights and biases). Set `0` to force
  every param onto the offload path.
- `prefetch_policy`: `'eager'` (default) starts h2d at the first
  `pre_backward_hook` fire; `'lazy'` defers prefetch to the leaf module
  that first owns each param in forward order. For shared params (tied
  embeddings) `'lazy'` collapses GPU residency of the shared param's
  non-grad states.

## Examples

- [`examples/e2e_train`](examples/e2e_train) — end-to-end training of a
  HuggingFace causal LM on synthetic data; throughput and peak-memory
  benchmarks across model sizes, sequence lengths, and rounding modes.
- [`examples/qat`](examples/qat) — quantization-aware training with
  int4 / int8 / NF4 / NVFP4 weight-only quantization on top of
  `OffloadAdam`.

## How it works

`OffloadAdam` has two execution paths, selected by whether `max_grad_norm`
is set:

- **`max_grad_norm=None` (default)**: the per-param Adam step runs inside
  that param's `post_accumulate_grad_hook` on the last microbatch. Loads,
  compute, and writebacks overlap with the rest of backward, and
  `optimizer.step()` is a no-op.
- **`max_grad_norm=<float>`**: backward only accumulates gradients and
  records per-param L2 norms. `optimizer.step()` reduces the global norm,
  applies the clip coefficient, and walks each param in fixed-size chunks
  (`step_chunk_size`) of load → compute → writeback. Chunking keeps peak
  GPU memory bounded for very large params (e.g. embedding tables).

Both paths share the same host↔device transfer primitives. The optimizer
step itself always runs on GPU.

## References

- [torchao](https://github.com/pytorch/ao)
- [optimi](https://github.com/warner-benjamin/optimi)
- [LMCache](https://github.com/LMCache/LMCache) — CPU pinned-memory design (mmap + mbind + cudaHostRegister) referenced when building `pinned_alloc.py`.
