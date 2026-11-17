# EXP-002: check whether the speedup persists

EXP-001 found a **pilot-only** 1.56× median throughput ratio between micro-batch 16 and micro-batch 4 (two seeds, three timed steps). EXP-002 retains the same tiny model, synthetic data, tokens per optimizer step and T4 GPU, but measures for **40 timed steps** after **5 warm-up steps** over **three seeds**.

This is still a measurement study, **not** the final 124M GPT-2 or held-out LLM-agent evaluation. It does not train on meaningful language data.

## Local no-GPU check

From the repository root:

```sh
uv run python kaggle/exp002/exp002.py --self-check
uv run --with pytest==9.0.2 python -m pytest -q tests
```

## Submit to Kaggle (consumes GPU quota)

```sh
kaggle kernels push -p kaggle/exp002 --accelerator NvidiaTeslaT4 --timeout 1800
kaggle kernels status chingchao/throughput-agent-exp002-timing
kaggle kernels output chingchao/throughput-agent-exp002-timing -p results/exp002
```

Only `cuda:0` is used; no GPU job starts until you issue the `push` command. The output is `results/exp002/exp002.json`; preserve EXP-001 independently.

## Analysis

Compare per-seed throughput for micro-batches 4, 8, 16; medians and ratios; allocated memory; timing agreement; and validation loss within each seed. Config order is reversed for seed 1235 and normal for 1234/1236 to partially reduce fixed-order bias. Differences between seeds or runs should be reported rather than tuned away.

**Stopping rule:** three successful measured trials per configuration with valid loss/timing checks, or document any failures and why the evidence is insufficient. Do not add more tuning or new experiments until the data have been reviewed.
