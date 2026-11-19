"""Opt-in, bounded Meta Muse API proposal-validity probe. No GPU training."""
from __future__ import annotations

import argparse
import json
import os
import time

from agent.llm import LLMConfigAgent
from harness.schema import Task

DEFAULT_MODEL = "muse-spark-1.3"
MAX_REQUESTS = 5
# This probe is deliberately limited to config fields that the baseline supports.
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "training_proposal",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "micro_batch_size": {"type": "integer"},
                        "grad_accum_steps": {"type": "integer"},
                        "precision": {"type": "string"},
                        "parallelism": {"type": "string"}
                    },
                    "required": ["micro_batch_size", "grad_accum_steps", "precision", "parallelism"],
                    "additionalProperties": False
                },
                "rationale": {"type": "string"}
            },
            "required": ["config", "rationale"],
            "additionalProperties": False
        }
    }
}


def make_dev_task() -> Task:
    return Task("muse-probe", "dev", "4-layer-synthetic-transformer",
                1, "T4", 128, 8192, 5, 40, 45, 0.02, 1, MAX_REQUESTS)


def run_probe(client, model: str, max_requests: int = MAX_REQUESTS) -> list[dict]:
    """Make no more than max_requests calls. Fail closed on API errors."""
    if not 1 <= max_requests <= MAX_REQUESTS:
        raise ValueError("max_requests must be between 1 and 5")
    history: list[dict] = []
    reports: list[dict] = []
    agent = LLMConfigAgent(complete=lambda prompt: _complete(client, model, prompt, reports))
    for _ in range(max_requests):
        before = len(reports)
        try:
            proposal = agent.propose(make_dev_task(), history)
        except Exception as exc:
            # Do not log full exception contents: providers may echo request data.
            reports.append({"status": "api_error", "error_type": type(exc).__name__})
            break
        if len(reports) == before:
            break
        record = reports[-1]
        if record.get("status") != "completed":
            break
        if proposal is None:
            record["proposal_status"] = agent.last_proposal_status or "unknown"
            # A malformed response should not lead to unbounded retries.
            break
        record["proposal_status"] = agent.last_proposal_status or "valid"
        record["config"] = proposal.config
        history.append({"config": proposal.config, "verdict": "not_evaluated"})
    return reports


def _complete(client, model: str, prompt: str, reports: list[dict]) -> str:
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=RESPONSE_FORMAT,
        max_completion_tokens=2048,
    )
    choice = response.choices[0]
    usage = response.usage
    reports.append({
        "status": "completed" if choice.finish_reason == "stop" else "incomplete",
        "finish_reason": choice.finish_reason,
        "latency_seconds": round(time.perf_counter() - start, 3),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    })
    if choice.finish_reason != "stop":
        return ""
    return choice.message.content or ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="explicitly allow paid Meta API calls")
    parser.add_argument("--max-requests", type=int, default=MAX_REQUESTS)
    args = parser.parse_args()
    if not 1 <= args.max_requests <= MAX_REQUESTS:
        parser.error("--max-requests must be 1..5")
    if not args.live:
        print("Dry run: no API requests made. Pass --live to authorize up to five requests.")
        return

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    key = os.getenv("MODEL_API_KEY")
    if not key:
        parser.error("MODEL_API_KEY is missing from environment or .env")
    model = os.getenv("META_MODEL", DEFAULT_MODEL)
    client = OpenAI(base_url="https://api.meta.ai/v1", api_key=key,
                    timeout=60.0, max_retries=0)
    records = run_probe(client, model, args.max_requests)
    print(json.dumps({"model": model, "max_requests": args.max_requests,
                      "requests_recorded": len(records), "results": records}, indent=2))


if __name__ == "__main__":
    main()
