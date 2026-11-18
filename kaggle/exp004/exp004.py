"""EXP-004: diagnostic CPU/CUDA profile of two model sizes and two micro-batches.

No external data or packages installed here. Requires Kaggle CUDA PyTorch to run.
A profiler changes timing: do NOT treat its timings as uninstrumented throughput.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENT_ID = "exp004-bottleneck-profile"
SEQ_LEN = 128
TOKENS_PER_STEP = 8192
VOCAB = 2048
SEED = 1234
WARMUP_STEPS = 2
PROFILE_STEPS = 2
MODELS = {"small": (4, 256, 4), "large": (8, 384, 6)}
MICRO_BATCHES = (4, 16)


def configs():
    result = []
    for model, (layers, width, heads) in MODELS.items():
        for micro_batch in MICRO_BATCHES:
            denominator = micro_batch * SEQ_LEN
            assert TOKENS_PER_STEP % denominator == 0
            result.append({
                "model": model, "layers": layers, "width": width, "heads": heads,
                "micro_batch_size": micro_batch,
                "grad_accum_steps": TOKENS_PER_STEP // denominator,
                "tokens_per_step": TOKENS_PER_STEP,
            })
    return result


def make_model(torch, cfg):
    import torch.nn as nn
    import torch.nn.functional as F

    width, heads = cfg["width"], cfg["heads"]

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(width)
            self.qkv = nn.Linear(width, 3 * width)
            self.proj = nn.Linear(width, width)
            self.norm2 = nn.LayerNorm(width)
            self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                     nn.Linear(4 * width, width))

        def forward(self, x):
            b, t, c = x.shape
            q, k, v = self.qkv(self.norm1(x)).split(c, dim=-1)
            def reshape(z):
                return z.view(b, t, heads, c // heads).transpose(1, 2)
            z = F.scaled_dot_product_attention(
                reshape(q), reshape(k), reshape(v), is_causal=True, dropout_p=0.0
            )
            x = x + self.proj(z.transpose(1, 2).contiguous().view(b, t, c))
            return x + self.mlp(self.norm2(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.token = nn.Embedding(VOCAB, width)
            self.pos = nn.Embedding(SEQ_LEN, width)
            self.blocks = nn.Sequential(*(Block() for _ in range(cfg["layers"])))
            self.norm = nn.LayerNorm(width)
            self.head = nn.Linear(width, VOCAB, bias=False)
            self.head.weight = self.token.weight

        def forward(self, tokens):
            pos = torch.arange(tokens.shape[1], device=tokens.device)
            x = self.token(tokens) + self.pos(pos)
            return self.head(self.norm(self.blocks(x)))

    return Model()


def profile_config(torch, cfg):
    import torch.nn.functional as F
    from torch.profiler import ProfilerActivity, profile, record_function

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:0")
    model = make_model(torch, cfg).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    effective_batch = TOKENS_PER_STEP // SEQ_LEN
    gen = torch.Generator(device=device).manual_seed(SEED + 10000)
    tokens = torch.randint(0, VOCAB,
        (WARMUP_STEPS + PROFILE_STEPS, effective_batch, SEQ_LEN + 1),
        generator=gen, device=device)

    def step(index, record=False):
        optimizer.zero_grad(set_to_none=True)
        # Labeled scopes identify CPU-side operators; CUDA events measure phase
        # elapsed time. They are diagnostic and include profiling overhead.
        phases = {"forward": [], "backward": [], "optimizer": []}
        for offset in range(0, effective_batch, cfg["micro_batch_size"]):
            batch = tokens[index, offset:offset + cfg["micro_batch_size"]]
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with record_function("training_forward"):
                a.record()
                logits = model(batch[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, VOCAB), batch[:, 1:].reshape(-1))
                b.record()
            if record:
                phases["forward"].append((a, b))
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with record_function("training_backward"):
                a.record()
                (loss / cfg["grad_accum_steps"]).backward()
                b.record()
            if record:
                phases["backward"].append((a, b))
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with record_function("training_optimizer"):
            a.record()
            optimizer.step()
            b.record()
        if record:
            phases["optimizer"].append((a, b))
        return phases

    for i in range(WARMUP_STEPS):
        step(i)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    phase_events = {"forward": [], "backward": [], "optimizer": []}
    wall_start = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, profile_memory=False, with_stack=False) as prof:
        for i in range(WARMUP_STEPS, WARMUP_STEPS + PROFILE_STEPS):
            events = step(i, record=True)
            for name, values in events.items():
                phase_events[name].extend(values)
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - wall_start
    phase_ms = {
        name: sum(a.elapsed_time(b) for a, b in event_pairs)
        for name, event_pairs in phase_events.items()
    }

    # CUDA operator times are inclusive of operator invocations, but do not
    # represent an exclusive partition of the elapsed wall-clock timeline.
    operations = []
    for key in prof.key_averages():
        gpu_us = getattr(key, "self_device_time_total", None)
        if gpu_us is None:
            gpu_us = getattr(key, "self_cuda_time_total", 0)
        if gpu_us and gpu_us > 0:
            operations.append({"name": key.key, "self_cuda_ms": gpu_us / 1000,
                               "calls": key.count})
    operations.sort(key=lambda x: x["self_cuda_ms"], reverse=True)
    result = {
        "config": cfg,
        "seed": SEED,
        "parameter_count": parameter_count,
        "profiled_steps": PROFILE_STEPS,
        "instrumented_wall_seconds": wall_seconds,
        "phase_cuda_event_ms": phase_ms,
        "top_cuda_operators": operations[:10],
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "status": "ok",
    }
    del model, optimizer, tokens, prof
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(output: Path):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for profiling")
    report = {
        "experiment_id": EXPERIMENT_ID,
        "status": "diagnostic_not_throughput_benchmark",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "workload": {"sequence_length": SEQ_LEN, "tokens_per_step": TOKENS_PER_STEP,
                     "warmup_steps": WARMUP_STEPS, "profiled_steps": PROFILE_STEPS,
                     "precision": "fp32", "seed": SEED, "data": "synthetic token IDs"},
        "results": [],
    }
    for cfg in configs():
        print("Profiling:", cfg["model"], "micro-batch", cfg["micro_batch_size"], flush=True)
        try:
            item = profile_config(torch, cfg)
        except (RuntimeError, ValueError) as exc:
            item = {"config": cfg, "status": "error", "error": str(exc)[:700]}
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
        report["results"].append(item)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
    print("Saved", output, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--output", default="/kaggle/working/exp004.json")
    args = ap.parse_args()
    if args.self_check:
        for cfg in configs():
            print(cfg)
        return
    run(Path(args.output))


if __name__ == "__main__":
    main()
