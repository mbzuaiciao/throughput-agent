"""Agent-side code. Imports schema only; never the evaluator internals."""
from __future__ import annotations
import json
import uuid
from pathlib import Path
from typing import Callable, Protocol
from harness.schema import Task, Submission, EvalResult


class Agent(Protocol):
    name: str

    def propose(self, task: Task, history: list[dict]) -> Submission | None:
        """Next submission, or None to stop early. `history` holds feedback_for_agent() dicts."""
        ...


def new_id() -> str:
    return uuid.uuid4().hex[:8]


def run_search(agent: Agent, task: Task, evaluate: Callable[[Task, Submission], EvalResult],
               log_path: Path) -> list[EvalResult]:
    """Fixed attempt budget; logs every proposal and full result (the log is the analysis input)."""
    history: list[dict] = []
    results: list[EvalResult] = []
    with log_path.open("a") as log:
        for i in range(task.attempt_budget):
            sub = agent.propose(task, history)
            if sub is None:
                break
            res = evaluate(task, sub)
            results.append(res)
            history.append({"config": sub.config, "rationale": sub.rationale, **res.feedback_for_agent()})
            log.write(json.dumps({"agent": agent.name, "attempt": i,
                                  "sub": {"id": sub.submission_id, "config": sub.config,
                                          "has_patch": sub.patch is not None, "rationale": sub.rationale},
                                  "result": res.to_json()}) + "\n")
    return results


def best_speedup(results: list[EvalResult]) -> float:
    """Rejected/failed attempts fall back to baseline (speedup 1.0)."""
    ok = [r.speedup_vs_baseline for r in results if r.speedup_vs_baseline is not None]
    return max(ok, default=1.0)
