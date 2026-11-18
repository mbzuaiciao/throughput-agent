"""No-GPU invariants for the EXP-004 diagnostic."""
import importlib.util
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "kaggle" / "exp004" / "exp004.py"
spec = importlib.util.spec_from_file_location("exp004", PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_profile_plan_has_four_fixed_workload_configs():
    planned = module.configs()
    assert len(planned) == 4
    assert {(x["model"], x["micro_batch_size"]) for x in planned} == {
        ("small", 4), ("small", 16), ("large", 4), ("large", 16)
    }
    assert all(x["tokens_per_step"] == 8192 for x in planned)
    assert all(x["micro_batch_size"] * x["grad_accum_steps"] * 128 == 8192
               for x in planned)


def test_model_dimensions_match_exp002_and_exp003():
    assert module.MODELS == {"small": (4, 256, 4), "large": (8, 384, 6)}
    assert module.WARMUP_STEPS == 2
    assert module.PROFILE_STEPS == 2
