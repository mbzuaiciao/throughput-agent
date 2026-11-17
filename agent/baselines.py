"""Non-LLM baselines sharing the Agent interface and the same attempt budget."""
from __future__ import annotations
import random
from harness.schema import Task, Submission
from harness.validate import CONFIG_SPEC
from .loop import new_id


class RandomSearch:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def propose(self, task: Task, history: list[dict]) -> Submission:
        cfg: dict = {}
        # Sample micro-batch from divisors so tokens/step stays exact.
        total_mb = task.global_batch_tokens // (task.n_gpus * task.seq_len)
        divisors = [d for d in range(1, total_mb + 1)
                    if total_mb % d == 0 and d <= 256 and total_mb // d <= 128]
        mb = self.rng.choice(divisors)
        cfg["micro_batch_size"], cfg["grad_accum_steps"] = mb, total_mb // mb
        for k, (typ, rng) in CONFIG_SPEC.items():
            if k in cfg:
                continue
            if typ is bool:
                cfg[k] = self.rng.random() < 0.5
            elif isinstance(rng, set):
                cfg[k] = self.rng.choice(sorted(rng))
            elif typ is int:
                cfg[k] = self.rng.randint(*rng)
        if task.n_gpus == 1:
            cfg["parallelism"] = "ddp"
        if cfg["parallelism"] == "ddp":
            cfg.pop("fsdp_sharding", None)
        return Submission(new_id(), task.task_id, cfg, rationale="uniform random")
