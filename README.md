# Offload Adam

Adam optimizer that offloads gradients and optimizer states to CPU memory, enabling full-parameter training of larger models with limited GPU memory.

## Local development

This repository is managed with `uv`.

Create the local development environment:

```bash
uv sync
```

Run ad-hoc checks from the project environment:

```bash
uv run python -c "import offload_adam; print(offload_adam.__version__)"
uv run pytest
uv run ruff check .
```

If you only want to refresh the lockfile after dependency changes:

```bash
uv lock
```


## Usage

### Adam

```python
from offload_adam import Adam

# Create a model
model = create_model().bfloat16().cuda()

# Initialize the optimizer
optimizer = Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    mode="stochastic_rounding",
    decoupled_weight_decay=True,  # AdamW
)

# Training loop
for input_data, target in dataloader:
    # Forward pass
    output = model(input_data)
    loss = loss_function(output, target)
    
    # Backward pass
    if gradient_accumulation:
        loss.backward()
    else:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### OffloadAdam

```python
from offload_adam import OffloadAdam

# Create a model
model = create_model().bfloat16().cuda()

# Initialize the optimizer
optimizer = OffloadAdam(
    model,  # pass model instead of model.parameters()
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    mode="stochastic_rounding",
    decoupled_weight_decay=True,  # AdamW
)

# Training loop
for input_data, target in dataloader:
    # Forward pass
    output = model(input_data)
    loss = loss_function(output, target)
    

    # Backward pass
    if gradient_accumulation:
        optimizer.ready_for_optimizer_step = False
        loss.backward()
    else:
        optimizer.ready_for_optimizer_step = True
        loss.backward()
        optimizer.step()
```

## How it works

`OffloadAdam` has two execution paths selected by whether
`gradient_clipping` is supplied:

* `gradient_clipping=None` (default): the whole optimizer step for each
  parameter runs inside that parameter's post-accumulate-grad hook on the
  last micro-batch. Optimizer-state loads, compute, and writebacks overlap
  with the remaining backward work, and `optimizer.step()` is a no-op.
* `gradient_clipping=dict(max_norm=..., norm_type=...)`: backward only
  accumulates gradients and records per-parameter norms. `optimizer.step()`
  then reduces the global norm, applies the clip coefficient, and walks each
  parameter in fixed-size chunks (`step_chunk_size`) of
  load → compute → writeback. Chunking keeps peak GPU memory bounded for
  very large parameters (e.g. embedding tables).

Both paths share the same host↔device transfer primitives. The optimizer
step always runs on GPU.

With offloading, it's possible do full-parameter training of:
* 7B models using single 24GB GPU and 42GB+ host memory
* 14B models using single 48GB GPU and 84GB+ host memory
* 32B models using single 80GB GPU and 192GB+ host memory


## Analysis

The overhead of offloading depends on the input size (total number of tokens in a batch) and GPU compute speed.

### Data transfer costs

gradients, momentum and variance in BF16

| Stage                 | H2D (bytes per param) | D2H (bytes per param) |
|-----------------------|-----------------------|-----------------------|
| gradient reset        | 0                     | 2                     |
| gradient accumulation | 2                     | 2                     |
| optimizer step (stochastic rounding) | 6      | 4                     |
| optimizer step (fp32 master weights) | 10     | 8                     |

### Overlapping data transfer and backward computation

#### 1. nn.Linear

Per-token backward time:

$\frac{4 \times H_{in} \times H_{out}}{TFLOPS \times 10^{12}}$

Weight transfer time:

$\frac{H_{in} \times H_{out} \times 2}{Bandwidth_{GB/s} \times 10^9}$

Number of tokens to overlap weight transfer:

$\frac{H_{in} \times H_{out} \times 2}{Bandwidth_{GB/s} \times 10^9} \div \frac{4 \times H_{in} \times H_{out}}{TFLOPS \times 10^{12}}$

$= \frac{TFLOPS \times 10^{12}}{2 \times Bandwidth_{GB/s} \times 10^9}$

$= \frac{TFLOPS}{2 \times Bandwidth_{GB/s}} \times 1000$


##### RTX4090 case

With theoretical values (165 TFLOPS, 32 GB/s PCIe 4.0):
* ~2578 tokens to overlap gradient transfer

With measured values (175 TFLOPS, 25 GB/s):
* ~3500 tokens to overlap gradient transfer

![Linear](assets/linear_backward.png)

In actual training, bandwidth will be lower.

### 2. nn.Embedding

Large memory consumption but small computational cost.

Actual involved tokens are usually smaller than the full table:
* For gradient accumulation, only used tokens in the current batch are involved.
* For optimizer step, all ever used tokens are involved because of momentum.

There are optimization chances but not implemented yet.

## TODO

- DTensor support

## References

- [torchao](https://github.com/pytorch/ao)
- [optimi](https://github.com/warner-benjamin/optimi)
- [LMCache](https://github.com/LMCache/LMCache) — CPU pinned-memory design (mmap + mbind + cudaHostRegister) referenced when building `pinned_alloc.py`.
