"""No-GPU structural checks for EXP-003."""
import importlib.util
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "kaggle" / "exp003" / "exp003.py"
spec = importlib.util.spec_from_file_location("exp003", PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_exp003_fixed_workload():
    assert module.SEQ_LEN == 128
    assert module.TOKENS_PER_STEP == 8192
    assert module.MICRO_BATCHES == (4, 8, 16)
    assert [c.grad_accum_steps for c in module.plan()] == [16, 8, 4]
    assert all(c.tokens_per_step == 8192 for c in module.plan())
    assert (module.WARMUP_STEPS, module.TIMED_STEPS) == (5, 40)
    assert module.SEEDS == (1234, 1235, 1236)


def test_exp003_missing_runs_not_scored():
    assert all(row["speedup"] is None for row in module.summarize([]))


def test_exp003_model_shape():
    # Model factory cannot be exercised without torch; verify defining code is unchanged
    # except for the intended layer count, width and heads.
    source = PATH.read_text()
    assert "width=384, heads=6" in source
    assert "range(8)" in source
    assert "nn.Embedding(2048, 384)" in source
