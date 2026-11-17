"""Core data contracts. Everything crossing the agent/evaluator boundary is defined here."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    ACCEPTED = "accepted"
    REJECTED_SCHEMA = "rejected_schema"        # config/patch violates whitelist
    REJECTED_FROZEN = "rejected_frozen_file"   # touched evaluator-owned code
    REJECTED_LOSS = "rejected_loss"            # loss deviates beyond tolerance
    REJECTED_TIMING = "rejected_timing"        # in-process vs parent clock mismatch
    OOM = "oom"
    CRASH = "crash"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class Task:
    task_id: str
    split: str                 # "dev" | "heldout" | "heldout_shift"
    model: str                 # e.g. "gpt2-124m"
    n_gpus: int
    gpu_type: str              # recorded, not enforced; used for shift analysis
    seq_len: int
    global_batch_tokens: int   # tokens per optimizer step; fixed per task
    warmup_steps: int
    timed_steps: int
    loss_check_steps: int      # step at which loss is compared to baseline
    loss_rel_tol: float        # e.g. 0.02
    n_repeats: int             # seeds per evaluation
    attempt_budget: int        # max proposals per task


@dataclass(frozen=True)
class Submission:
    submission_id: str
    task_id: str
    config: dict[str, Any]           # validated against CONFIG_SPEC
    patch: str | None = None         # unified diff, restricted to ALLOWED_PATCH_GLOBS
    parent_id: str | None = None
    rationale: str = ""              # agent's stated reason; logged, never trusted


@dataclass
class RunMeasurement:
    seed: int
    tokens_per_sec: float              # from evaluator-owned CUDA-event timing
    parent_wall_tokens_per_sec: float  # from wall clock around the timed loop (cross-check)
    peak_mem_gb: float
    loss_at_check: float


@dataclass
class EvalResult:
    submission_id: str
    task_id: str
    verdict: Verdict
    measurements: list[RunMeasurement] = field(default_factory=list)
    median_tps: float | None = None
    speedup_vs_baseline: float | None = None
    detail: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d

    def feedback_for_agent(self) -> dict:
        """The ONLY view of a result the agent may see."""
        return {
            "verdict": self.verdict.value,
            "median_tps": self.median_tps,
            "speedup_vs_baseline": self.speedup_vs_baseline,
            "detail": self.detail[:500],
        }
