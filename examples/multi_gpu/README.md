# Multi-GPU training with DistributedOffloadAdam (experimental)

`DistributedOffloadAdam` is experimental — tested on a narrow set of
hardware (2× RTX 6000 Ada, PCIe-only; 2× A100-SXM with NVLink) and
exercised only with Qwen3-8B / Qwen3-14B. Expect rough edges on other
topologies, models, or world sizes.

`DistributedOffloadAdam` is a ZeRO-1 style variant of `OffloadAdam`: each
rank keeps the full model parameters on its GPU but only a 1/N slice of the
optimizer state in pinned host memory. Per parameter the per-step flow is:

1. H2D prefetch of the local state slice.
2. `reduce_scatter` the full local gradient to the 1/N slice.
3. Adam step on the slice (GPU).
4. D2H the updated state slice.
5. `all_gather` the updated parameter slice back to every rank.

No `DistributedDataParallel` wrapper is required — gradient communication
is driven by the optimizer itself (step 2). Wrapping with DDP is not
supported: it would only add a redundant all-reduce on top of the
reduce-scatter ZeRO-1 already does.

Launch with `torchrun`:

```bash
torchrun --nproc_per_node=2 examples/multi_gpu/train.py \
    --model-name Qwen/Qwen3-8B \
    --per-rank-batch-size 1 \
    --tokens-per-sample 2048 \
    --grad-accum-steps 1 \
    --steps 10
```

## Benchmarks

### Environment

2× NVIDIA RTX 6000 Ada Generation (48 GB each), PCIe 4.0 × 16,
**no NVLink** — cross-GPU collectives go through PCIe + CPU root
complex. `NCCL_P2P_DISABLE=1`.

Qwen3-8B in bf16, sdpa attention, gradient checkpointing, Liger fused
linear cross-entropy, `mode=stochastic_rounding`. 3 warmup + 5 measured
steps; `per-rank-batch-size=1`, `grad-accum=1`.


| seq_len | 1-GPU tok/s | 2-GPU tok/s | scale | peak (GB, per rank) |
| ------- | ----------- | ----------- | ----- | ------------------- |
| 8 192   | 1 302       | 2 327       | 1.79× | 22.4                |
| 16 384  | 1 188       | 2 297       | 1.93× | 24.9                |


## Limitations

- Offload params whose `shape[0]` doesn't divide `world_size` stay  
replicated per rank instead of sharded. Their gradients are still  
reduced across ranks (`all_reduce` in the grad hook), so training  
is correct, but those params don't get the 1/N host memory savings.
- On some platforms multiple GPUs share a single x16 PCIe link to the  
CPU; the combined H2D/D2H offloading traffic then competes for that  
shared lane and throughput drops accordingly.
- `load_state_dict` is not supported (same as single-GPU `OffloadAdam`).

When the GPU-local NUMA node can't hold the per-rank pinned budget,  
pages spill to remote nodes via `MPOL_PREFERRED`. Training still runs  
but H2D bandwidth drops ~10-15% on the spilled pages. A warning prints  
at init with the expected spill size.

QAT is supported: `p.data` (packed storage) is replicated per rank; only  
the fp32 master is sharded. After step_fn, the master slices are  
all-gathered and `p.data.copy_(full_master)` re-encodes the packed weight.