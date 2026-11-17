"""EVALUATOR-OWNED entrypoint (listed in the frozen manifest). Agent patches cannot touch it.

Skeleton: fill the TODOs after vendoring nanoGPT into train/. Structure matters more than detail:
  1. build model + data from the TASK spec (not from agent config) so everyone does the same work
  2. apply only whitelisted config knobs
  3. time with CUDA events around a fixed number of steps after a fixed warmup
  4. report loss at a fixed step for the guardrail
"""
import argparse
import json
import os
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    task = json.load(open(f"tasks/resolved/{a.task}.json"))  # frozen copy written by the freeze step
    torch.manual_seed(a.seed)

    # TODO(1): model, data, optimizer built from `task` (fixed arch, fixed data order per seed)
    # TODO(2): wrap with cfg["parallelism"], cfg["precision"], cfg["compile_mode"], ...
    # TODO(3): import hooks from train/hooks/ (the only agent-writable code) at documented call sites

    def train_step():  # TODO: one optimizer step including grad accumulation
        raise NotImplementedError

    for _ in range(task["warmup_steps"]):
        train_step()
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    w0 = time.perf_counter()
    start.record()
    for _ in range(task["timed_steps"]):
        train_step()
    end.record()
    torch.cuda.synchronize()
    w1 = time.perf_counter()

    timed_tokens = task["timed_steps"] * task["global_batch_tokens"]
    loss_at_check = 0.0  # TODO: continue to task["loss_check_steps"]; eval loss on a fixed val batch

    if int(os.environ.get("RANK", 0)) == 0:
        json.dump({
            "tokens_per_sec": timed_tokens / (start.elapsed_time(end) / 1e3),
            "wall_timed_s": w1 - w0,
            "timed_tokens": timed_tokens,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
            "loss_at_check": loss_at_check,
        }, open(a.out, "w"))


if __name__ == "__main__":
    main()
