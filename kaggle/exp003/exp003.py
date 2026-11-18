"""EXP-003: test a larger transformer on the same Kaggle T4 GPU.

No downloads, external datasets, package installs, or cloud setup are performed.
This pilot is NOT the final GPT-2 124M benchmark or a held-out agent evaluation.
Run --self-check locally without torch; a real run requires a CUDA GPU + PyTorch.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENT_ID = "exp003-kaggle-model-size"
SEQ_LEN = 128
TOKENS_PER_STEP = 8192
MICRO_BATCHES = (4, 8, 16)
WARMUP_STEPS = 5
TIMED_STEPS = 40
N_REPEATS = 3
SEEDS = (1234, 1235, 1236)
LOSS_REL_TOL = 0.02


@dataclass(frozen=True)
class Config:
    micro_batch_size: int
    grad_accum_steps: int

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps * SEQ_LEN


def make_config(micro_batch: int, tokens_per_step: int = TOKENS_PER_STEP) -> Config:
    if isinstance(micro_batch, bool) or not isinstance(micro_batch, int) or micro_batch < 1:
        raise ValueError("micro-batch size must be a positive integer")
    if tokens_per_step <= 0 or tokens_per_step % (micro_batch * SEQ_LEN):
        raise ValueError("cannot preserve fixed tokens/step with integer accumulation")
    config = Config(micro_batch, tokens_per_step // (micro_batch * SEQ_LEN))
    if config.grad_accum_steps < 1:
        raise ValueError("invalid accumulation count")
    return config


def plan() -> list[Config]:
    return [make_config(mb) for mb in MICRO_BATCHES]


def model_factory(torch):
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self, width=384, heads=6):
            super().__init__()
            self.heads = heads
            self.norm1 = nn.LayerNorm(width)
            self.qkv = nn.Linear(width, 3 * width)
            self.proj = nn.Linear(width, width)
            self.norm2 = nn.LayerNorm(width)
            self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                     nn.Linear(4 * width, width))

        def forward(self, x):
            b, t, c = x.shape
            q, k, v = self.qkv(self.norm1(x)).split(c, dim=-1)
            reshape = lambda z: z.view(b, t, self.heads, c // self.heads).transpose(1, 2)
            a = F.scaled_dot_product_attention(reshape(q), reshape(k), reshape(v),
                                               is_causal=True, dropout_p=0.0)
            x = x + self.proj(a.transpose(1, 2).contiguous().view(b, t, c))
            return x + self.mlp(self.norm2(x))

    class TinyGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.token = nn.Embedding(2048, 384)
            self.pos = nn.Embedding(SEQ_LEN, 384)
            self.blocks = nn.Sequential(*(Block() for _ in range(8)))
            self.norm = nn.LayerNorm(384)
            self.head = nn.Linear(384, 2048, bias=False)
            self.head.weight = self.token.weight

        def forward(self, tokens):
            positions = torch.arange(tokens.shape[1], device=tokens.device)
            hidden = self.token(tokens) + self.pos(positions)
            return self.head(self.norm(self.blocks(hidden)))

    return TinyGPT()


def one_run(config: Config, seed: int, torch) -> dict:
    import torch.nn.functional as F

    if config.tokens_per_step != TOKENS_PER_STEP:
        raise ValueError("workload changed")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")
    model = model_factory(torch).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    effective_batch = TOKENS_PER_STEP // SEQ_LEN

    # Identical token stream for each config at the same seed.
    generator = torch.Generator(device=device).manual_seed(seed + 10000)
    training = torch.randint(0, 2048, (WARMUP_STEPS + TIMED_STEPS,
                            effective_batch, SEQ_LEN + 1), generator=generator, device=device)
    validation = torch.randint(0, 2048, (8, SEQ_LEN + 1),
                               generator=generator, device=device)

    def step(step_index: int):
        optimizer.zero_grad(set_to_none=True)
        for offset in range(0, effective_batch, config.micro_batch_size):
            batch = training[step_index, offset:offset + config.micro_batch_size]
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, 2048), batch[:, 1:].reshape(-1))
            (loss / config.grad_accum_steps).backward()
        optimizer.step()

    torch.cuda.reset_peak_memory_stats(device)
    for i in range(WARMUP_STEPS):
        step(i)
    torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start_event.record()
    for i in range(WARMUP_STEPS, WARMUP_STEPS + TIMED_STEPS):
        step(i)
    end_event.record()
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - wall_start
    cuda_seconds = start_event.elapsed_time(end_event) / 1000

    model.eval()
    with torch.no_grad():
        logits = model(validation[:, :-1])
        validation_loss = F.cross_entropy(logits.reshape(-1, 2048),
                                          validation[:, 1:].reshape(-1)).item()
    torch.cuda.synchronize(device)
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    tokens = TOKENS_PER_STEP * TIMED_STEPS
    measurements = {
        "seed": seed,
        "config": asdict(config),
        "parameter_count": parameter_count,
        "tokens_per_step": TOKENS_PER_STEP,
        "timed_optimizer_steps": TIMED_STEPS,
        "timed_tokens": tokens,
        "cuda_seconds": cuda_seconds,
        "wall_seconds": wall_seconds,
        "tokens_per_second": tokens / cuda_seconds,
        "wall_tokens_per_second": tokens / wall_seconds,
        "peak_allocated_gb": peak_gb,
        "validation_loss": validation_loss,
    }
    del optimizer, model, training, validation, logits
    gc.collect()
    torch.cuda.empty_cache()
    return measurements


def summarize(runs: list[dict]) -> list[dict]:
    """Judge loss and timing relative to baseline; failures never count as gains."""
    medians = {}
    for mb in MICRO_BATCHES:
        rows = [r for r in runs if r["config"]["micro_batch_size"] == mb and r["status"] == "ok"]
        if len(rows) != N_REPEATS:
            medians[mb] = None
            continue
        medians[mb] = {
            "median_tps": statistics.median(r["tokens_per_second"] for r in rows),
            "median_loss": statistics.median(r["validation_loss"] for r in rows),
            "max_timing_difference": max(abs(r["cuda_seconds"] - r["wall_seconds"]) /
                                        r["wall_seconds"] for r in rows),
        }
    base = medians[MICRO_BATCHES[0]]
    summaries = []
    for mb, values in medians.items():
        if values is None or base is None:
            summaries.append({"micro_batch_size": mb, "status": "incomplete", "speedup": None})
            continue
        loss_error = abs(values["median_loss"] - base["median_loss"]) / max(abs(base["median_loss"]), 1e-12)
        valid = (math.isfinite(values["median_tps"]) and values["median_tps"] > 0 and
                 math.isfinite(values["median_loss"]) and
                 values["max_timing_difference"] <= 0.10 and loss_error <= LOSS_REL_TOL)
        summaries.append({
            "micro_batch_size": mb,
            "status": "accepted" if valid else "rejected_guardrail",
            "median_tps": values["median_tps"],
            "median_loss": values["median_loss"],
            "relative_loss_change": loss_error,
            "max_timing_difference": values["max_timing_difference"],
            "speedup": values["median_tps"] / base["median_tps"] if valid else None,
        })
    return summaries


def run(output: Path) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required: select a Kaggle GPU accelerator")
    print("GPU:", torch.cuda.get_device_name(0), "torch:", torch.__version__, flush=True)
    runs = []
    # Rotate configuration order across seeds; identical data per seed and configuration.
    for seed_index, seed in enumerate(SEEDS):
        configurations = plan()
        if seed_index % 2:
            configurations.reverse()
        else:
            configurations = configurations[seed_index:] + configurations[:seed_index]
        for order_index, cfg in enumerate(configurations):
            print(f"Running seed={seed} micro_batch={cfg.micro_batch_size}", flush=True)
            try:
                row = one_run(cfg, seed, torch)
                row["status"] = "ok"
                row["execution_order"] = order_index
            except (RuntimeError, ValueError) as exc:
                row = {"seed": seed, "config": asdict(cfg),
                       "status": "error", "execution_order": order_index,
                       "error": str(exc)[:500]}
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
            runs.append(row)
            # Persist partial results after each candidate, even on failure.
            result = {
                "experiment_id": EXPERIMENT_ID,
                "status": "larger_model_pilot_not_final_gpt2",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "hardware": {"gpu": torch.cuda.get_device_name(0),
                             "visible_gpu_count": torch.cuda.device_count(),
                             "torch": torch.__version__,
                             "python": platform.python_version()},
                "model": "8-layer 384-wide causal transformer (6 heads)",
                "data": "fixed-seed synthetic uniform token IDs; no external downloads",
                "workload": {"seq_len": SEQ_LEN, "tokens_per_step": TOKENS_PER_STEP,
                             "warmup_steps": WARMUP_STEPS, "timed_steps": TIMED_STEPS,
                             "repeats": N_REPEATS, "precision": "fp32",
                             "optimizer": "AdamW lr=1e-3"},
                "runs": runs,
                "summary": summarize(runs),
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2))
    print("Saved:", output, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true", help="validate workload without torch/GPU")
    ap.add_argument("--output", default="/kaggle/working/exp003.json")
    args = ap.parse_args()
    if args.self_check:
        for cfg in plan():
            assert cfg.tokens_per_step == TOKENS_PER_STEP
            print(cfg)
        return
    run(Path(args.output))


if __name__ == "__main__":
    main()
