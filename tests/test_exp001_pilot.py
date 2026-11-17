"""No-GPU checks for the standalone EXP-001 pilot."""
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "kaggle" / "exp001.py"
spec = importlib.util.spec_from_file_location("exp001", PATH)
module = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_all_configs_preserve_training_tokens():
    configs = module.plan()
    assert [c.micro_batch_size for c in configs] == [4, 8, 16]
    assert [c.grad_accum_steps for c in configs] == [16, 8, 4]
    assert all(c.tokens_per_step == 8192 for c in configs)


@pytest.mark.parametrize("micro", [0, -1, 3, True, 1.5])
def test_invalid_workloads_rejected(micro):
    with pytest.raises((TypeError, ValueError)):
        module.make_config(micro)


def test_incomplete_runs_not_scored():
    summary = module.summarize([])
    assert all(x["status"] == "incomplete" and x["speedup"] is None for x in summary)
