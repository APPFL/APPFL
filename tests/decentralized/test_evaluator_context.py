"""The evaluation context ADKOAgent passes at step 9 (agent_id, round_idx, eval_idx), and
the adapter that lets a plain ``f(point)`` evaluator ignore it. Built through the real
builder from the toy 1-D example config, so the whole local path is exercised."""

import json
from pathlib import Path

from omegaconf import OmegaConf

from appfl.decentralized.algorithm.adko import create_agent
from appfl.decentralized.config import accept_context
from appfl.decentralized.metrics import Meter, metered_evaluator
from appfl.decentralized.neighbor import resolve_neighbors

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
_TOY_CONFIG = "decentralized/resources/configs/toy1d/distributed/node_0.yaml"

_RECORDING_EVALUATOR = '''
import json


def get_evaluator(log_path):
    def evaluate(point, **context):
        with open(log_path, "a") as log:
            log.write(json.dumps(context) + "\\n")
        return 50.0
    return evaluate
'''


def test_plain_evaluator_drops_the_context():
    evaluator = accept_context(lambda x: x * 2)
    assert evaluator(3.0, agent_id="a", round_idx=1, eval_idx=0) == 6.0


def test_kwargs_evaluator_is_returned_unwrapped():
    def evaluate(point, **context):
        return context["eval_idx"]

    assert accept_context(evaluate) is evaluate


_CONTEXT = {"agent_id": "agent-0", "round_idx": 12, "eval_idx": 3}


def test_named_context_parameters_are_forwarded():
    def evaluate(point, agent_id, round_idx, eval_idx):
        return (point, agent_id, round_idx, eval_idx)

    assert accept_context(evaluate)(0.5, **_CONTEXT) == (0.5, "agent-0", 12, 3)


def test_only_the_named_subset_is_forwarded():
    def evaluate(point, eval_idx):
        return eval_idx

    assert accept_context(evaluate)(0.5, **_CONTEXT) == 3


def test_metered_plain_evaluator_takes_the_full_context():
    # The metering wrapper accepts **context itself, so it must adapt what it wraps --
    # otherwise the agent's adapter passes straight through it to a plain f(point).
    meter = Meter()
    evaluator = accept_context(metered_evaluator(lambda x: x * 2, meter))
    assert evaluator(3.0, **_CONTEXT) == 6.0
    assert meter.evaluations == 1


def test_defaulted_and_keyword_only_parameters_are_filled():
    def evaluate(point, eval_idx=None, *, round_idx=None):
        return (eval_idx, round_idx)

    assert accept_context(evaluate)(0.5, **_CONTEXT) == (3, 12)
    # A key the agent does not send keeps its default.
    assert accept_context(evaluate)(0.5, agent_id="agent-0") == (None, None)


def _toy_agent(monkeypatch, **evaluator_configs):
    monkeypatch.chdir(_EXAMPLES)
    node_config = OmegaConf.load(_TOY_CONFIG)
    node_config.evaluator_configs = OmegaConf.merge(
        node_config.evaluator_configs, evaluator_configs
    )
    neighbors = resolve_neighbors(node_config.neighbors, str(node_config.node_id))
    return create_agent(node_config, neighbors, Meter())


def test_agent_passes_its_own_context(monkeypatch, tmp_path):
    path = tmp_path / "recording_evaluator.py"
    path.write_text(_RECORDING_EVALUATOR)
    log_path = tmp_path / "calls.jsonl"
    agent = _toy_agent(
        monkeypatch,
        evaluator_path=str(path),
        evaluator_kwargs={"log_path": str(log_path)},
    )
    for round_idx in range(10, 14):
        agent.act(round_idx)

    calls = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert calls == [
        {"agent_id": "agent-0", "round_idx": r, "eval_idx": k}
        for k, r in enumerate(range(10, 14))
    ]


def test_plain_evaluator_still_works_through_the_builder(monkeypatch):
    agent = _toy_agent(monkeypatch)  # toy_yield.true_yield takes only the point
    for round_idx in range(3):
        agent.act(round_idx)
    assert agent.best_so_far() is not None
