"""Configuration-only LLM proposal agent. No network calls or GPU jobs."""
import json
from typing import Callable
from harness.schema import Task, Submission
from harness.validate import validate_config, CONFIG_SPEC
from .loop import new_id


def _constraint_description() -> dict[str, object]:
    """Describe the actual evaluator schema rather than only the key names."""
    result = {}
    for name, (kind, bounds) in CONFIG_SPEC.items():
        if kind is bool:
            result[name] = {"type": "boolean"}
        elif isinstance(bounds, set):
            result[name] = {"type": "string", "allowed": sorted(bounds)}
        else:
            result[name] = {"type": "integer", "min": bounds[0], "max": bounds[1]}
    return result


def _invalid_field(errors: list[str]) -> str | None:
    """Return only an approved field name, never a rejected value or error text."""
    for error in errors:
        prefix = error.split(":", 1)[0]
        if prefix in CONFIG_SPEC:
            return prefix
        if error.startswith("unknown key:"):
            return "unknown_key"
        if error.startswith("micro_batch_size and grad_accum_steps"):
            return "micro_batch_size_or_grad_accum_steps"
        if error.startswith("tokens/step "):
            return "micro_batch_size_or_grad_accum_steps"
        if error.startswith("fsdp_sharding set"):
            return "fsdp_sharding"
    return "configuration"




class LLMConfigAgent:
    name = "llm-config"

    def __init__(self, complete: Callable[[str], str]):
        self.complete = complete
        self.last_proposal_status: str | None = None
        self.last_invalid_field: str | None = None

    def propose(self, task: Task, history: list[dict]) -> Submission | None:
        self.last_proposal_status = None
        self.last_invalid_field = None
        if task.split != "dev":
            raise ValueError("Only development tasks are supported")
        if len(history) >= task.attempt_budget:
            self.last_proposal_status = "attempt_budget_exhausted"
            return None

        request = {
            "instruction": (
                "Suggest one NEW training configuration. Return only JSON with keys "
                "config (object) and rationale (string), or {\"stop\":true}. "
                "Keep tokens per optimizer step fixed, propose no code patches. "
                "The mandatory fields are micro_batch_size and grad_accum_steps. "
                "For single-GPU tasks, parallelism=ddp is appropriate. "
                "Never set fsdp_sharding when parallelism=ddp."
            ),
            "task": {
                "task_id": task.task_id, "model": task.model,
                "gpu_type": task.gpu_type, "n_gpus": task.n_gpus,
                "seq_len": task.seq_len,
                "global_batch_tokens": task.global_batch_tokens,
                "attempts_remaining": task.attempt_budget - len(history),
            },
            "allowed_config": _constraint_description(),
            "token_equation": (
                "micro_batch_size * grad_accum_steps * n_gpus * seq_len "
                "= global_batch_tokens"
            ),
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
            self.last_invalid_field = _invalid_field(errors)
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
