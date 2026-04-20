"""Multi-GPU training script using DistributedOffloadAdam.

Launches via torchrun:

    torchrun --nproc_per_node=N examples/multi_gpu/train.py [flags]

Each rank holds full model parameters but a 1/N slice of the optimizer
state. Replicated grad is reduce-scattered and updated params are
all-gathered inside each parameter's backward hook.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from offload_adam import DistributedOffloadAdam


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
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
    parser.add_argument(
        "--per-rank-batch-size",
        type=int,
        default=1,
        help="Samples processed per rank per microbatch.",
    )
    parser.add_argument("--tokens-per-sample", type=int, default=2048)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument(
        "--numa-node",
        default="auto",
        help="NUMA policy for pinned allocations: 'auto', an int node id, or 'none'.",
    )
    parser.add_argument(
        "--prefetch-policy",
        choices=["eager", "lazy"],
        default="eager",
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


def setup_distributed():
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, dist.get_rank(), dist.get_world_size()


def maybe_apply_liger(args):
    if not args.liger_fused_linear_ce:
        return
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3

    apply_liger_kernel_to_qwen3(
        cross_entropy=False,
        fused_linear_cross_entropy=True,
    )


def build_model(args, local_rank):
    maybe_apply_liger(args)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map={"": f"cuda:{local_rank}"},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.train()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def build_optimizer(args, model):
    numa_node = args.numa_node
    if numa_node == "none":
        numa_node = None
    elif numa_node != "auto":
        numa_node = int(numa_node)
    return DistributedOffloadAdam(
        model,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
        mode=args.mode,
        decoupled_weight_decay=True,
        max_grad_norm=args.max_grad_norm,
        numa_node=numa_node,
        prefetch_policy=args.prefetch_policy,
        verbose=1,
    )


def synthetic_batch_iter(args, vocab_size, device):
    while True:
        input_ids = torch.randint(
            0,
            vocab_size,
            (args.per_rank_batch_size, args.tokens_per_sample),
            device=device,
            dtype=torch.long,
        )
        yield {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
            "labels": input_ids.clone(),
        }


def run(args, model, optimizer, rank, world_size, device):
    vocab_size = model.config.vocab_size
    data_iter = synthetic_batch_iter(args, vocab_size, device)
    tokens_per_step = (
        args.per_rank_batch_size
        * args.tokens_per_sample
        * args.grad_accum_steps
        * world_size
    )

    step_times = []
    step_losses = []
    torch.cuda.reset_peak_memory_stats()

    for step_idx in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        last_loss = None

        for micro in range(args.grad_accum_steps):
            optimizer.ready_for_optimizer_step = micro == args.grad_accum_steps - 1
            batch = next(data_iter)
            out = model(**batch)
            loss = out.loss / args.grad_accum_steps
            last_loss = out.loss.detach().float()
            loss.backward()

        optimizer.step()
        torch.cuda.synchronize()
        dist.barrier()
        elapsed = time.perf_counter() - t0

        # Averaged loss across ranks for logging.
        dist.all_reduce(last_loss, op=dist.ReduceOp.AVG)
        last_loss = last_loss.item()

        phase = "warmup" if step_idx < args.warmup_steps else "measure"
        if rank == 0:
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
        "world_size": world_size,
        "mode": args.mode,
        "per_rank_batch_size": args.per_rank_batch_size,
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = build_model(args, local_rank)
    optimizer = build_optimizer(args, model)

    if rank == 0:
        print(
            f"model={args.model_name} world_size={world_size} "
            f"device={torch.cuda.get_device_name(local_rank)} "
            f"mode={args.mode}",
            flush=True,
        )

    summary = run(args, model, optimizer, rank, world_size, device)

    if rank == 0:
        print(json.dumps(summary, indent=2, sort_keys=True))
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
