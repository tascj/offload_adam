# Quantization-aware training (QAT) example

A single-file training script (`train.py`) that swaps eligible
`nn.Linear` weights for a low-bit quantized tensor subclass and trains
with `Adam` or `OffloadAdam` in `fp32_master` mode.

## How this differs from fake-quantize

Stock fake-quantize keeps the weight at full precision on GPU and
simulates the quantizer in the forward; memory usage barely changes —
you get the numeric behaviour of a quant model without any training-
time VRAM savings.

Here the weight actually lives in packed low-bit storage on GPU
(`Int4QWeight` / `NF4QWeight` / ... are real `torch.Tensor` wrapper
subclasses backing `nn.Linear.weight.data`). Forward dispatches to a
dequant-then-matmul; backward produces a plain bf16
gradient via a custom `autograd.Function`. The fp32 master used by
Adam lives in the optimizer state — on GPU for `Adam`, on pinned host
for `OffloadAdam` — and is re-quantized back into the packed storage
through the tensor subclass's `aten.copy_` handler after every optimizer step.

Net effect for an 8B bf16 model on a 96 GB Blackwell: ~10 GB of
resident GPU weight memory saved by going to 4-bit (≈9.5 GB of weight
bytes), and `OffloadAdam` drops the *total* GPU footprint from 22 GB
to 13 GB because the fp32 master never lands on-device.

## The QAT API in two calls

```python
from offload_adam import Adam, OffloadAdam
from offload_adam.qweight import Int4QWeight, quantize_linears

model = ...  # a bf16 nn.Module on GPU (e.g. HF `from_pretrained` output)

# 1. Replace every eligible nn.Linear.weight in place. Returns a
#    QuantizeReport listing quantized / excluded (skip_patterns) /
#    incompatible (can_quantize) layers.
quantize_linears(
    model, Int4QWeight,
    group_size=128,
    skip_patterns=("lm_head",),  # output head kept in full precision
    strict=True,
)

# 2. Construct the optimizer — must be fp32_master mode. For quant
#    params, the optimizer seeds master by calling `copy_` on p.data,
#    which dispatches into the QWeight handler and dequantizes —
#    lossy but training-ready. Non-quant params get a lossless
#    bf16 → fp32 widen.
optim = Adam(model.parameters(), mode="fp32_master", lr=1e-4)
# or
optim = OffloadAdam(model, mode="fp32_master", lr=1e-4)
```

That is enough to start training. The first `optim.step()` emits a
one-shot warning pointing at the lossless upgrade path.

### Using lossless master

```python
# Same shape AutoModelForCausalLM.from_pretrained accepts — a local
# file, a directory (single or sharded), or an HF Hub repo ID.
optim.load_master_from_pretrained("Qwen/Qwen3-8B", model)
```

`load_master_from_pretrained` points at the same bf16/fp16 checkpoint
`from_pretrained` consumed before `quantize_linears` ran. It reads
original bf16/fp16 tensors from the checkpoint, rather than dequantize from
`p.data`.


## Supported dtypes

| class         | bits | layout                      | scale                                | constraints                          |
| :------------ | :--- | :-------------------------- | :----------------------------------- | :----------------------------------- |
| `Int4QWeight` | 4    | GPTQ groupwise int4         | fp32/bf16 per group                  | `in_f % group_size == 0`, `% 8 == 0` |
| `Int8QWeight` | 8    | per-channel int8            | fp32/bf16 per output row             | —                                    |
| `NF4QWeight`  | 4    | bnb NF4 blockwise           | fp32 per block                       | `in_f % blocksize == 0`, `% 2 == 0`  |
| `NVFP4QWeight`| 4    | compressed-tensors NVFP4    | fp8\_e4m3 per 16-block + fp32 global | `in_f % 16 == 0`, `% 2 == 0`         |

Plug any of them into `quantize_linears(model, <Cls>, ...)`; the
training path is identical across dtypes.

## Constraints

- **`mode="fp32_master"` required** for any QAT optimizer; other modes
  don't carry the high-precision copy QAT depends on.
- **`max_grad_norm` + quant params** runs `OffloadAdam`'s unchunked
  full-param step path for quant layers — peak device memory during
  the clipped update is higher than the normal chunked path.

## Usage

```bash
uv sync --group examples
uv run python examples/qat/train.py \
    --quant int4 --enable-offloading \
    --tokens-per-sample 8192 --steps 6 --warmup-steps 2

# Other dtypes: --quant {int8,nf4,nvfp4}
# Lossless master: --load-master-from Qwen/Qwen3-8B
```

Run with `--help` for all flags. Defaults: `Qwen/Qwen3-8B`, bf16, sdpa
attention, gradient checkpointing on, Liger fused linear CE on,
`--mode fp32_master`.

## Benchmarks

Qwen3-8B on a single NVIDIA RTX PRO 6000 Blackwell (96 GB), bf16,
sdpa attention, gradient checkpointing, Liger fused linear
cross-entropy, `mode=fp32_master`. 2 warmup + 4 measured steps,
`seq_len=8192`, `batch_size=1`, `grad_accum=1`, synthetic tokens.

Every attention/MLP projection (Q, K, V, O, gate, up, down) is
quantized; `lm_head` stays in bf16.

|   optimizer | quant | tok/s | peak (GB) |
| ----------: | ----: | ----: | --------: |
|        Adam |  none |  4144 |      91.8 |
|        Adam |  int8 |  3779 |      85.3 |
|        Adam |  int4 |  3803 |      82.2 |
|        Adam |   nf4 |  3735 |      82.5 |
|        Adam | nvfp4 |  3645 |      82.5 |
| OffloadAdam |  none |  2759 |      23.4 |
| OffloadAdam |  int8 |  2632 |      17.0 |
| OffloadAdam |  int4 |  2668 |      13.9 |
| OffloadAdam |   nf4 |  2643 |      14.2 |
| OffloadAdam | nvfp4 |  2602 |      14.2 |

4-bit dtypes save ≈9.6 GB of weight bytes vs bf16; int8 saves ≈6.5 GB.
`OffloadAdam` stacks the weight-byte savings on top of its already-
offloaded master, so `Adam → OffloadAdam → +int4` compounds into a
~6.6× total reduction (91.8 → 13.9 GB) at ~64% throughput.

## Storage layouts

Each subclass stores data in the layout the corresponding vllm loader
expects, and `model.state_dict()` emits per-dtype canonical keys
(end-to-end loading into vllm is not yet CI-verified):

- `Int4QWeight` → GPTQ raw layout (`qweight (in_f//8, out_f) int32`,
  `scales (n_groups, out_f)`; symmetric `qzeros` synthesized at save).
- `Int8QWeight` → plain `(out_f, in_f) int8` in VRAM; `state_dict`
  re-packs to compressed-tensors W8A16 (`weight_packed (out_f, in_f//4)
  int32`, `weight_scale (out_f, 1)`).
- `NF4QWeight` → bnb NF4 (`weight_packed (out_f, in_f//2) uint8`,
  `absmax (out_f·n_groups,) fp32`, even → high nibble).
- `NVFP4QWeight` → compressed-tensors NVFP4 (`weight_packed (out_f,
  in_f//2) uint8`, `block_scale_fp8 (out_f, n_groups) fp8_e4m3`,
  `global_scale () fp32`, even → low nibble).
