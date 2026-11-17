# throughput-agent

**Research question:** Does an LLM agent discover training-throughput improvements that generalize to unfamiliar model and GPU configurations, or does it overfit development tasks?

The repo is currently **a research scaffold, not a completed GPU benchmark**. `bench/entry.py` still contains TODOs and the experiment does not yet have measured throughput results.

## Start here

Python tooling uses [uv](https://docs.astral.sh/uv/). From the repository root:

```sh
uv sync --locked
uv run --with pytest==9.0.2 python -m pytest -q tests
```

The base environment intentionally has no heavy runtime dependencies. The test runner is supplied using `uv run --with`; that runner is pinned in the command but is **not included in uv.lock**. GPU-training dependencies (PyTorch/CUDA) will be added when the benchmark is implemented and pinned for a specific training environment. Nothing automatically provisions or rents a GPU.

## Repository guide

- `harness/`: evaluator-owned task schema, config/patch validation, measurement and held-out ledger. Agent code must never import evaluator internals.
- `bench/entry.py`: frozen training/timing entrypoint; currently unfinished.
- `train/`: training implementation and the limited agent-writable `train/hooks/` area.
- `agent/`: search loop and baselines (random search is present; LLM agent and BO are not yet implemented).
- `tasks/`: dev and shifted held-out task specifications.
- `analysis/`: statistics for bootstrap intervals and dev-to-held-out generalization.
- `tests/`: static/unit tests.

## Evaluation protocol

1. Develop the agent scaffold on development tasks only.
2. Freeze prompts, memory and search policy; record their hash.
3. Run the frozen agent on held-out tasks, allowing within-task feedback under the same attempt budget as baselines.
4. Report throughput speedups, loss validity, failures, uncertainty and dev-to-held-out gap. Failed or rejected configurations fall back to speedup 1.0 in best-result reporting.

**Known limits:** Agent-writable Python hooks currently execute in the same process as the timer, so the planned timing checks do not prove isolation. A small held-out set supports pilot conclusions only.

## Next milestones

1. Establish a reproducible Python environment and learning-focused Astro frontend.
2. Complete a single-GPU reference benchmark with trustworthy timing and loss checks.
3. Compare untouched, manually tuned and random-search configurations.
4. Introduce the LLM agent and perform a genuinely held-out evaluation.

See [GitHub Issues](https://github.com/mbzuaiciao/throughput-agent/issues) for scoped coding tasks.
