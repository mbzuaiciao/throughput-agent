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
        self.last_proposal_status: str | None = None

    def propose(self, task: Task, history: list[dict]) -> Submission | None:
        self.last_proposal_status = None
        if task.split != "dev":
            raise ValueError("Only development tasks are supported")
        if len(history) >= task.attempt_budget:
            self.last_proposal_status = "attempt_budget_exhausted"
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
            self.last_proposal_status = "invalid_json"
            return None
        if proposal == {"stop": True}:
            self.last_proposal_status = "explicit_stop"
            return None
        if not isinstance(proposal, dict) or set(proposal) != {"config", "rationale"}:
            self.last_proposal_status = "invalid_response_shape"
            return None
        cfg, why = proposal["config"], proposal["rationale"]
        if not isinstance(cfg, dict) or not isinstance(why, str):
            self.last_proposal_status = "invalid_response_shape"
            return None
        errors = validate_config(task, cfg)
        if errors:
            self.last_proposal_status = (
                "token_budget_mismatch" if any(e.startswith("tokens/step ") for e in errors)
                else "invalid_config"
            )
            return None
        if any(h.get("config") == cfg for h in history):
            self.last_proposal_status = "duplicate_config"
            return None
        self.last_proposal_status = "valid"
        return Submission(new_id(), task.task_id, cfg, rationale=why[:500])
