"""End-to-end training script with synthetic data.

Loads a HuggingFace causal LM (default Qwen3-8B, optionally Liger-patched),
feeds it random token batches, and runs Adam or OffloadAdam from this
library. Reports per-step timing and a final JSON summary for easy
aggregation across runs.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from offload_adam import Adam, OffloadAdam


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--enable-offloading",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use OffloadAdam when enabled, plain Adam otherwise.",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "stochastic_rounding",
            "fp32_master",
            "fp31_master",
            "fp32_master_custom_rounding",
        ],
        default="fp32_master",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tokens-per-sample", type=int, default=2048)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help=(
            "Enable L2 global-norm gradient clipping (OffloadAdam only). "
            "When set, OffloadAdam runs the chunked .step() path; when None "
            "(default), it runs the step-in-backward path."
        ),
    )
    parser.add_argument(
        "--numa-node",
        default="auto",
        help="NUMA policy for pinned allocations: 'auto', an int node id, or 'none'.",
    )
    parser.add_argument(
        "--prefetch-policy",
        choices=["eager", "lazy"],
        default="eager",
        help=(
            "OffloadAdam prefetch scheduling. 'eager' issues h2d at the "
            "first pre_backward fire; 'lazy' restricts to the leaf module "
            "that owns each param earliest in forward order."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--liger-fused-linear-ce",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def validate_args(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.tokens_per_sample < 1:
        raise ValueError("--tokens-per-sample must be >= 1")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.warmup_steps < 0 or args.warmup_steps >= args.steps:
        raise ValueError("--warmup-steps must be in [0, --steps)")
    if args.liger_fused_linear_ce and "qwen3" not in args.model_name.lower():
        raise ValueError(
            "--liger-fused-linear-ce is currently wired only for Qwen3 models"
        )


def maybe_apply_liger(args):
    if not args.liger_fused_linear_ce:
        return
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3

    apply_liger_kernel_to_qwen3(
        cross_entropy=False,
        fused_linear_cross_entropy=True,
    )


def build_model(args):
    maybe_apply_liger(args)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.train()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def build_optimizer(args, model):
    common = {
        "lr": args.lr,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": args.weight_decay,
        "mode": args.mode,
        "decoupled_weight_decay": True,
    }
    if args.enable_offloading:
        numa_node = args.numa_node
        if numa_node == "none":
            numa_node = None
        elif numa_node != "auto":
            numa_node = int(numa_node)
        return OffloadAdam(
            model,
            numa_node=numa_node,
            verbose=1,
            max_grad_norm=args.max_grad_norm,
            prefetch_policy=args.prefetch_policy,
            **common,
        )
    return Adam(model.parameters(), **common)


def synthetic_batch_iter(args, vocab_size):
    while True:
        input_ids = torch.randint(
            0,
            vocab_size,
            (args.batch_size, args.tokens_per_sample),
            device="cuda",
            dtype=torch.long,
        )
        yield {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
            "labels": input_ids.clone(),
        }


def run(args, model, optimizer):
    vocab_size = model.config.vocab_size
    data_iter = synthetic_batch_iter(args, vocab_size)
    tokens_per_step = args.batch_size * args.tokens_per_sample * args.grad_accum_steps

    step_times = []
    step_losses = []
    torch.cuda.reset_peak_memory_stats()

    for step_idx in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        last_loss = None

        for micro in range(args.grad_accum_steps):
            if hasattr(optimizer, "ready_for_optimizer_step"):
                optimizer.ready_for_optimizer_step = micro == args.grad_accum_steps - 1
            batch = next(data_iter)
            out = model(**batch)
            loss = out.loss / args.grad_accum_steps
            last_loss = out.loss.detach().float().item()
            loss.backward()

        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        phase = "warmup" if step_idx < args.warmup_steps else "measure"
        print(
            f"step={step_idx + 1}/{args.steps} phase={phase} "
            f"loss={last_loss:.4f} step_time_s={elapsed:.4f} "
            f"tokens_per_s={tokens_per_step / elapsed:.1f} "
            f"cuda_peak_gb={torch.cuda.max_memory_allocated() / (1024**3):.2f}",
            flush=True,
        )

        if step_idx >= args.warmup_steps:
            step_times.append(elapsed)
            step_losses.append(last_loss)

    total_tokens = tokens_per_step * len(step_times)
    total_s = sum(step_times)
    return {
        "model_name": args.model_name,
        "enable_offloading": args.enable_offloading,
        "mode": args.mode,
        "batch_size": args.batch_size,
        "tokens_per_sample": args.tokens_per_sample,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_tokens": tokens_per_step,
        "gradient_checkpointing": args.gradient_checkpointing,
        "liger_fused_linear_ce": args.liger_fused_linear_ce,
        "measured_steps": len(step_times),
        "warmup_steps": args.warmup_steps,
        "avg_loss": sum(step_losses) / len(step_losses),
        "avg_step_time_s": total_s / len(step_times),
        "tokens_per_second": total_tokens / total_s,
        "max_cuda_memory_gb": torch.cuda.max_memory_allocated() / (1024**3),
    }


def main():
    args = parse_args()
    validate_args(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = build_model(args)
    optimizer = build_optimizer(args, model)

    print(
        f"model={args.model_name} device={torch.cuda.get_device_name(0)} "
        f"enable_offloading={args.enable_offloading} mode={args.mode}",
        flush=True,
    )

    summary = run(args, model, optimizer)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
