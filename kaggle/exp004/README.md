# EXP-004 — Where does the GPU spend its time?

**Question:** Why did the observed micro-batch speedup shrink from ~1.56× (4-layer small model) to ~1.13× (8-layer larger model)?

One Kaggle T4 GPU; two model sizes × micro-batches 4 and 16. Same 8,192 tokens/optimizer step (accumulation 16 or 4), synthetic data, FP32 and AdamW. Two warmup steps and two *profiled* steps, one seed. Uses built-in PyTorch CPU/CUDA profiler, named forward/backward/optimizer scopes, CUDA events, and top CUDA operator summaries. No large trace files are exported.

**Important:** Profiling adds overhead. Profiled times must **not** be compared as if they were the uninstrumented throughput timings from EXP-002/003. The purpose is to identify plausible bottlenecks, not to prove a causal explanation.

## Steps (in order)

1. Before merging the PR, verify locally without a GPU:

   ```sh
   uv run python kaggle/exp004/exp004.py --self-check
   uv run --with pytest==9.0.2 python -m pytest -q tests
   ```

2. After tests pass, merge the PR into `main`, then `git switch main && git pull origin main`.
3. Only then, **manually** launch the bounded Kaggle GPU profiling job (uses your Kaggle GPU quota):

   ```sh
   kaggle kernels push -p kaggle/exp004 --accelerator NvidiaTeslaT4 --timeout 1800
   ```

4. Once its status is COMPLETE, download results:

   ```sh
   kaggle kernels status chingchao/throughput-agent-exp004-profile
   kaggle kernels output chingchao/throughput-agent-exp004-profile -p results/exp004
   ```

5. Inspect `exp004.json`. Report any unsupported-profiler errors or missing CUDA operator records. Do not automatically retry or expand the experiment.

**Stop:** Interpret the four profiles once; decide whether forward/backward/kernel overhead plausibly explains the observed pattern or whether evidence is inconclusive. EXP-004 is diagnostic only; no agent or held-out evaluation.
