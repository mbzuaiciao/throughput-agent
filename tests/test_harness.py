from harness.schema import Task
from harness.validate import validate_config, validate_patch
from analysis.stats import bootstrap_ci, dev_heldout_gap
from agent.baselines import RandomSearch

T = Task("t", "dev", "gpt2-124m", 1, "A100", 1024, 524288, 10, 50, 200, 0.02, 5, 20)


def good():
    return {"micro_batch_size": 16, "grad_accum_steps": 32, "precision": "bf16", "parallelism": "ddp"}


def test_good_config_passes():
    assert validate_config(T, good()) == []


def test_token_budget_enforced():
    c = good(); c["grad_accum_steps"] = 16   # halves the work: must be rejected
    assert any("tokens/step" in e for e in validate_config(T, c))


def test_unknown_key_rejected():
    c = good(); c["skip_eval"] = True
    assert any("unknown key" in e for e in validate_config(T, c))


def test_bool_not_accepted_as_int():
    c = good(); c["micro_batch_size"] = True
    assert validate_config(T, c)


def test_patch_outside_whitelist_rejected():
    p = "--- a/bench/entry.py\n+++ b/bench/entry.py\n@@ -1 +1 @@\n-x\n+y\n"
    assert validate_patch(p)


def test_patch_inside_whitelist_ok():
    p = "--- a/train/hooks/x.py\n+++ b/train/hooks/x.py\n@@ -1 +1 @@\n-x\n+y\n"
    assert validate_patch(p) == []


def test_bootstrap_ci_brackets_mean():
    m, lo, hi = bootstrap_ci([1.0, 1.1, 1.2, 1.3, 1.4], n_boot=2000)
    assert lo <= m <= hi


def test_gap_positive_when_dev_exceeds_heldout():
    g, lo, hi = dev_heldout_gap([1.5, 1.6, 1.4], [1.0, 1.05, 0.98], n_boot=2000)
    assert g > 0


def test_random_search_proposals_always_schema_valid():
    rs = RandomSearch(seed=1)
    for _ in range(200):
        s = rs.propose(T, [])
        assert validate_config(T, s.config) == [], s.config
