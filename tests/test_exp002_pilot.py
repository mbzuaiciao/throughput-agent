"""No-GPU checks for EXP-002 measurement-plan invariants."""
import importlib.util
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "kaggle" / "exp002" / "exp002.py"
spec = importlib.util.spec_from_file_location("exp002", PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

def test_exp002_workload_and_length():
    assert module.WARMUP_STEPS == 5
    assert module.TIMED_STEPS == 40
    assert len(module.SEEDS) == module.N_REPEATS == 3
    assert [c.grad_accum_steps for c in module.plan()] == [16, 8, 4]
    assert all(c.tokens_per_step == module.TOKENS_PER_STEP for c in module.plan())

def test_exp002_incomplete_not_accepted():
    assert all(row["status"] == "incomplete" for row in module.summarize([]))
