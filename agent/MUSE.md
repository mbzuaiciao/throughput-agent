# Muse-001: Opt-in Meta Muse API validity probe

This is a minimal connection to Meta Muse Spark via **Chat Completions**. It injects a completion into the existing `LLMConfigAgent`; the existing Python validator checks the fixed-token workload and configuration schema. It does **not** train on a GPU or assess throughput.

## Credentials

Create `throughput-agent/.env` (repository root):

```dotenv
MODEL_API_KEY=replace_with_your_meta_api_key
META_MODEL=muse-spark-1.3
```

Do **not** push this file to GitHub. The root `.gitignore` excludes `.env` and variants. Never use the key in Astro or browser code.

## Test locally without API requests

```sh
uv run --with pytest==9.0.2 python -m pytest -q tests
uv run --with openai --with python-dotenv python -m agent.muse_probe
```

The second command prints a dry-run notice and makes no API calls.

## Optional: live probe (uses billed API inference)

After confirming pricing and billing on Meta, *explicitly invoke*:

```sh
uv run --with openai --with python-dotenv python -m agent.muse_probe --live --max-requests 5
```

There is no automatic retry. At most five API requests are sent in one invocation. Invalid/duplicate configurations, incomplete answers, or API errors stop early. Outputs summarize response status, token usage, latency and the proposed **nonsecret** configurations. This is not a monetary spending cap; check account pricing before running.

The probe uses a minimal JSON Schema to request required config fields; it never treats structured output as equivalent to a valid training workload. Only development tasks are allowed. The current adapter terminates early on invalid proposals; that is an initial limitation rather than an optimization score.

**Stop:** Once the connection and proposal validation work, do not expand to optimizing throughput until a trusted GPU evaluator and budget-matched random baseline are available.
