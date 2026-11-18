# EXP-003 — larger model, same Kaggle T4

## Question
In EXP-002, micro-batch 16 was ~1.56× faster than micro-batch 4. Does the ranking persist for a larger Transformer?

This changes the model from **4 layers, width 256, 4 attention heads** to **8 layers, width 384, 6 attention heads**. The exact parameter count is recorded for every successful run. Same GPU, sequence length 128, vocabulary 2048, FP32, AdamW, synthetic token data, tokens per optimizer step (8192), 5 warmups, 40 timed steps, and 3 matched seeds. Micro-batches remain 4/8/16, with accumulation 16/8/4.

This is still a **synthetic training throughput study**, not a GPT-2 124M benchmark or an agent generalization evaluation. We have not yet tested EXP-003 on GPU. Out-of-memory failures are useful results and should be reported, not hidden.

## Step 1 — Test locally (does not need a GPU)

From the repository root:

```sh
uv run python kaggle/exp003/exp003.py --self-check
uv run --with pytest==9.0.2 python -m pytest -q tests
```

## Step 2 — After the PR is merged, run on Kaggle

From the repository root; this command **submits and starts** a remote GPU job that uses your Kaggle quota:

```sh
kaggle kernels push -p kaggle/exp003 --accelerator NvidiaTeslaT4 --timeout 1800
```

The script uses **cuda:0 only**. The timeout is 30 minutes, not a guarantee that it will finish in that time.

## Step 3 — Only when Kaggle reports COMPLETE, download results

```sh
kaggle kernels status chingchao/throughput-agent-exp003-model-size
kaggle kernels output chingchao/throughput-agent-exp003-model-size -p results/exp003
```

The expected output is `results/exp003/exp003.json`. Keep the file separate from EXP-001 and EXP-002. Review logs for sensitive information before sharing.

## Stop and interpret
If all configurations complete, compare per-seed and median throughput, speedups, memory allocation, timing agreement and loss, particularly versus EXP-002. If any configuration fails or exceeds the timeout, document that result and stop rather than changing the model until the findings are reviewed. There is no LLM agent in this experiment.
