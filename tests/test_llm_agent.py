"""Offline fake-LLM tests; do not call a paid API."""
import json
import pytest
from agent.llm import LLMConfigAgent
from harness.schema import Task

TASK = Task("dev1", "dev", "tiny", 1, "T4", 128, 8192, 5, 40, 45, 0.02, 3, 3)
CFG = {"micro_batch_size": 16, "grad_accum_steps": 4, "precision": "fp32"}


def agent(response):
    return LLMConfigAgent(lambda _: response)


def test_valid_proposal():
    seen = []
    a = LLMConfigAgent(lambda prompt: seen.append(json.loads(prompt)) or
                       json.dumps({"config": CFG, "rationale": "reduce passes"}))
    sub = a.propose(TASK, [])
    assert sub.config == CFG and sub.patch is None
    assert seen[0]["task"]["attempts_remaining"] == 3


@pytest.mark.parametrize("reply", [
    "not json", "[]", '{"stop": true}',
    '{"config": {"micro_batch_size": 16, "grad_accum_steps": 2}, "rationale": "bad"}',
    '{"config": {"micro_batch_size": 16, "grad_accum_steps": 4}, "rationale": "ok", "patch": "x"}',
    '{"config": {"micro_batch_size": 16, "grad_accum_steps": 4}, "rationale": 5}',
])
def test_rejected_reply(reply):
    assert agent(reply).propose(TASK, []) is None


def test_duplicate_budget_and_heldout():
    a = agent(json.dumps({"config": CFG, "rationale": "same"}))
    assert a.propose(TASK, [{"config": CFG}]) is None
    assert a.propose(TASK, [{"config": {}}] * 3) is None
    other = Task("held", "heldout_shift", "tiny", 1, "T4", 128, 8192, 5, 40, 45, 0.02, 3, 3)
    with pytest.raises(ValueError):
        a.propose(other, [])
