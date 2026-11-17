# Coding-agent instructions

Read this file and the relevant code before making changes. Follow any more specific instructions in nested AGENTS.md files.

## Research contract
- This project tests whether training-throughput improvements found on development tasks generalize to held-out configurations.
- Keep the evaluator-owned benchmark and validation separate from agent-controlled proposals. Do not relax the workload, loss, or timing checks to make a run appear faster.
- Develop on dev tasks only. Do not tune the agent based on held-out results; treat held-out reuse as contamination.
- Never invent benchmark numbers or claim that a GPU run happened without a recorded measurement.
- Never start paid cloud resources, install paid services, or schedule GPU jobs without explicit user approval.
- Preserve source files, experiments and historical data unless a scoped issue explicitly requires changing them.

## Working style
- Implement one GitHub Issue at a time, favor the smallest working change and keep refactors separate from behavior changes.
- Explain code plainly so the owner can learn from the implementation. Link key concepts to the relevant source.
- Python dependencies use uv. The static Astro learning site, when present, uses npm under web/.
- Run relevant tests and builds, report exactly what ran, and note unverified GPU behavior.
- Stop when the issue's acceptance criteria are met; avoid unrelated architecture, services and infrastructure.
