# Minimal LLM configuration agent

`LLMConfigAgent` accepts an injected `complete(prompt: str) -> str` function. It does not connect to a provider or incur API costs. A completion must return JSON such as:

```json
{"config":{"micro_batch_size":16,"grad_accum_steps":4,"precision":"fp32"},"rationale":"reduce accumulation passes"}
```

The agent operates on **dev tasks only**, validates workload through the existing schema, rejects duplicate proposals, accepts a stop instruction and proposes no code patches. Malformed/invalid output stops early; it is not scored as a benchmark failure in this first adapter.

Run offline tests from the repository root:

```sh
uv run --with pytest==9.0.2 python -m pytest -q tests
```

**Not measured:** No real LLM, benchmark, API or GPU run has been executed. The full evaluator remains unfinished. Next, connect an explicitly approved model/provider and perform a budget-matched comparison to random/manual baselines only when the evaluator works.
