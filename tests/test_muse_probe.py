"""No-network tests for Muse's bounded probe."""
from types import SimpleNamespace
import json
import pytest

from agent.muse_probe import run_probe, make_dev_task


class FakeClient:
    def __init__(self, responses):
        self.calls = 0
        self.responses = responses
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        assert kwargs["response_format"]["type"] == "json_schema"
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=self.responses[self.calls - 1])
        )], usage=SimpleNamespace(prompt_tokens=20, completion_tokens=30))


def reply(mb):
    return json.dumps({"config": {"micro_batch_size": mb, "grad_accum_steps": 64 // mb,
                                  "precision": "fp32", "parallelism": "ddp"},
                       "rationale": "reduce accumulation"})


def test_five_valid_proposals_are_bounded():
    client = FakeClient([reply(x) for x in (1, 2, 4, 8, 16)])
    results = run_probe(client, "muse-spark-1.3")
    assert client.calls == len(results) == 5
    assert all(r["proposal_status"] == "valid" for r in results)
    assert all(r["prompt_tokens"] == 20 for r in results)


def test_duplicate_ends_without_retrying():
    client = FakeClient([reply(4), reply(4)])
    results = run_probe(client, "muse-spark-1.3")
    assert client.calls == 2
    assert results[-1]["proposal_status"] == "duplicate_config"


def test_invalid_token_budget_is_rejected():
    bad = json.dumps({"config": {"micro_batch_size": 16, "grad_accum_steps": 2,
                                  "precision": "fp32", "parallelism": "ddp"},
                      "rationale": "bad"})
    client = FakeClient([bad])
    results = run_probe(client, "muse-spark-1.3")
    assert len(results) == 1 and results[0]["proposal_status"] == "token_budget_mismatch"


@pytest.mark.parametrize("count", [0, 6])
def test_invalid_count_rejected(count):
    with pytest.raises(ValueError):
        run_probe(FakeClient([]), "muse-spark-1.3", count)


def test_dev_only_task():
    assert make_dev_task().split == "dev"
    assert make_dev_task().global_batch_tokens == 8192


def test_explicit_stop_vs_invalid_json_vs_bad_shape():
    for reply_text, expected in [
        ('{"stop": true}', "explicit_stop"),
        ("this is not JSON", "invalid_json"),
        ('{"config": {}, "rationale": "bad", "patch": "unexpected"}', "invalid_response_shape"),
        ('{"config": {"micro_batch_size": 16, "grad_accum_steps": 4, "precision": "illegal", "parallelism": "ddp"}, "rationale": "bad"}', "invalid_config"),
    ]:
        client = FakeClient([reply_text])
        outcome = run_probe(client, "muse-spark-1.3", max_requests=1)
        assert client.calls == 1
        assert outcome[0]["proposal_status"] == expected
        assert "response" not in outcome[0]
        assert "prompt" not in outcome[0]
