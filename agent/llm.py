"""Configuration-only LLM proposal agent. No network calls or GPU jobs."""
import json
from typing import Callable
from harness.schema import Task, Submission
from harness.validate import validate_config, CONFIG_SPEC
from .loop import new_id


class LLMConfigAgent:
    name = "llm-config"

    def __init__(self, complete: Callable[[str], str]):
        self.complete = complete

    def propose(self, task: Task, history: list[dict]) -> Submission | None:
        if task.split != "dev":
            raise ValueError("Only development tasks are supported")
        if len(history) >= task.attempt_budget:
            return None

        request = {
            "instruction": (
                "Suggest one NEW training configuration. Return only JSON with keys "
                "config (object) and rationale (string), or {\"stop\":true}. "
                "Keep tokens per optimizer step fixed, propose no code patches."
            ),
            "task": {
                "task_id": task.task_id, "model": task.model,
                "gpu_type": task.gpu_type, "n_gpus": task.n_gpus,
                "seq_len": task.seq_len,
                "global_batch_tokens": task.global_batch_tokens,
                "attempts_remaining": task.attempt_budget - len(history),
            },
            "allowed_config_keys": sorted(CONFIG_SPEC),
            "history": [
                {key: h.get(key) for key in ("config", "verdict", "speedup_vs_baseline")}
                for h in history
            ],
        }
        try:
            proposal = json.loads(self.complete(json.dumps(request, sort_keys=True)))
        except (TypeError, ValueError):
            return None
        if proposal == {"stop": True}:
            return None
        if not isinstance(proposal, dict) or set(proposal) != {"config", "rationale"}:
            return None
        cfg, why = proposal["config"], proposal["rationale"]
        if not isinstance(cfg, dict) or not isinstance(why, str):
            return None
        if validate_config(task, cfg):
            return None
        if any(h.get("config") == cfg for h in history):
            return None
        return Submission(new_id(), task.task_id, cfg, rationale=why[:500])
