# throughput-agent

Does an LLM agent find training-throughput gains that **generalize to held-out setups**, or does it overfit the dev set?
Same question as the AFML harness, applied to a systems objective.

## Layout
- `harness/` evaluator-owned: schema, static validation, measurement, held-out ledger. Agent code never imports `evaluator.py`.
- `bench/entry.py` evaluator-owned timing entrypoint (frozen by manifest).
- `train/` vendored nanoGPT-style code. Only `train/hooks/*.py` is agent-writable (free-form arm).
- `agent/` loop and baselines (random search now; add BO/evolutionary and the LLM agent).
- `tasks/{dev,heldout,heldout_shift}/` task specs. Shift tasks differ in model size, GPU type and GPU count.
- `analysis/stats.py` bootstrap CIs and the dev-vs-held-out gap statistic.
- `results/` JSONL logs (every proposal and full result).

## Protocol
1. Develop the agent scaffold (prompts, memory, parameters) on dev tasks only.
2. Freeze it: `scaffold_hash(...)`. Held-out is scored once per hash via `HeldoutLedger.claim`.
3. On held-out tasks the agent gets within-task feedback (same attempt budget as the baselines), but the scaffold is fixed.
4. Rejected or failed attempts count as speedup 1.0. Report CIs, dev-to-held-out gap, failure taxonomy, $/task.

## Known limits (state them in the write-up)
- Free-form patches run in the same process as the timer. Mitigations: CUDA-event vs wall-clock agreement, loss guardrail, token-budget equality, frozen entrypoint. Not a proof of integrity.
- ~6 held-out tasks means wide CIs: pilot-scale result.

## Order of work
1. Vendor nanoGPT; implement the `bench/entry.py` TODOs; confirm timing variance <2-3% on one task.
2. Baselines: untouched, hand-tuned, random, BO. Write `baselines.json` for the Evaluator.
3. Freeze step (hash frozen files into the manifest, resolve task YAMLs to `tasks/resolved/*.json`), then the LLM agent.
4. Held-out run, analysis, write-up.

Run tests: `PYTHONPATH=. python -m pytest -q tests`
