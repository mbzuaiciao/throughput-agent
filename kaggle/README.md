# EXP-001 Kaggle pilot

**Scope:** This is a small **measurement-validation pilot**, not the GPT-2 124M model or the final held-out experiment from Issue #3. No trained weights are saved. Data are deterministic synthetic token IDs to avoid external downloads and leakage.

## Run locally (does not require PyTorch)

From repository root:

```sh
uv run python kaggle/exp001.py --self-check
uv run --with pytest==9.0.2 python -m pytest -q tests
```

## Run on Kaggle

1. Use the latest `feat/kaggle-exp001` branch or the merged version.
2. Make sure the Kaggle CLI is authenticated as `chingchao` and that your GPU usage quota is available.
3. First inspect the script and metadata: this command starts a remote GPU execution and consumes Kaggle quota.

```sh
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4 --timeout 1800
kaggle kernels status chingchao/throughput-agent-exp001-pilot
kaggle kernels output chingchao/throughput-agent-exp001-pilot -p results/exp001
```

Kaggle allocates **T4 ×2**, but the script intentionally uses **only cuda:0** for a single-GPU study. Its notebook output is saved to `/kaggle/working/exp001.json`. Kaggle's container must provide working CUDA PyTorch. The first command uploads the code and launches the run. To upload without executing, use `--no-run`.

## What it measures

All candidates process **8,192 tokens per optimizer step**, sequence length 128, four short optimizer steps including one warmup, and two matched seeds:
- micro-batch 4, accumulation 16
- micro-batch 8, accumulation 8
- micro-batch 16, accumulation 4

Outputs include CUDA-event throughput, host wall-clock throughput, peak allocated GPU memory, held-out fixed-batch validation loss, errors and a simple within-pilot loss/timing guardrail. Both clocks are measured in the same process; they are a diagnostic cross-check, **not an independent trusted evaluator**. FP32, model architecture, seed, optimizer and data stream are held fixed.

The run is intentionally tiny. With only two repeats and a synthetic dataset, any speedups are exploratory; there is **no claim** of statistical significance, quality generalization, or agent performance.

## Next

Inspect the JSON and whether both timings agree reasonably. Then implement the trustworthy evaluator-owned GPT-2 training benchmark and expanded replicates under Issue #3. Do not use this pilot to score held-out LLM-agent behavior.
