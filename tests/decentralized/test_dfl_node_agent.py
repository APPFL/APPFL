"""Tests for `DFLNodeAgent`.

The serial example exercises the synchronous path only: one thread, one node at a time. The
round-tagged publication, the lock, the futures and the two-round buffer exist for the
distributed launchers, where several nodes run at once -- so they are tested here with real
threads rather than left for a network layer to discover.
"""

import threading
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from appfl.agent import DFLNodeAgent
from appfl.decentralized.neighbor import (
    generated_node_ids,
    resolve_all_neighbors_from_topology,
)
from appfl.decentralized.topology import build_topology

RESOURCES = Path(__file__).parent / "resources"


def node_config(node_id, index, num_nodes, output_dir, model="TinyNet", **overrides):
    config = OmegaConf.create(
        {
            "node_id": node_id,
            "aggregator": "DFLNodeFedAvgAggregator",
            "aggregator_kwargs": {},
            "train_configs": {
                "trainer": "VanillaTrainer",
                "mode": "step",
                "num_local_steps": 1,
                "optim": "SGD",
                "optim_args": {"lr": 0.05},
                "loss_fn_path": str(RESOURCES / "celoss.py"),
                "loss_fn_name": "CELoss",
                "metric_path": str(RESOURCES / "acc.py"),
                "metric_name": "accuracy",
                "do_validation": False,
                "do_pre_validation": False,
                "train_batch_size": 8,
                "val_batch_size": 8,
                # Off so the arms are deterministic: the only thing under test is mixing.
                "train_data_shuffle": False,
                "val_data_shuffle": False,
                "device": "cpu",
                "logging_output_dirname": str(output_dir),
                "logging_id": node_id,
            },
            "model_configs": {
                "seed": 42,
                "model_path": str(RESOURCES / "tiny_model.py"),
                "model_name": model,
            },
            "data_configs": {
                "dataset_path": str(RESOURCES / "tiny_dataset.py"),
                "dataset_name": "get_tiny",
                "dataset_kwargs": {"num_clients": num_nodes, "client_id": index},
            },
        }
    )
    for key, node in overrides.items():
        config[key] = node
    return config


def build_federation(tmp_path, num_nodes=3, topology_name="fully_connected", **kwargs):
    ids = generated_node_ids(num_nodes)
    topology = build_topology(topology_name, ids, **kwargs)
    views = resolve_all_neighbors_from_topology(topology)
    agents = [
        DFLNodeAgent(
            dfl_node_agent_config=node_config(node_id, i, num_nodes, tmp_path),
            neighbors=views[node_id],
        )
        for i, node_id in enumerate(ids)
    ]
    return topology, {a.get_id(): a for a in agents}


# ---------------------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------------------


def test_a_node_is_a_client_agent(tmp_path):
    from appfl.agent import ClientAgent

    _, agents = build_federation(tmp_path, num_nodes=2)
    agent = agents["Node0"]
    assert isinstance(agent, ClientAgent)
    assert agent.trainer is not None and agent.model is not None
    assert agent.aggregator is not None  # what a client does not have


def test_every_node_starts_from_the_same_model(tmp_path):
    """With no server to ship the initial model, the shared seed is what aligns the nodes."""
    _, agents = build_federation(tmp_path, num_nodes=3)
    reference = agents["Node0"].model.state_dict()
    for agent in agents.values():
        for name, tensor in agent.model.state_dict().items():
            assert torch.allclose(tensor, reference[name])


def test_seed_can_be_opted_out_of(tmp_path):
    config_a = node_config("Node0", 0, 2, tmp_path)
    config_b = node_config("Node1", 1, 2, tmp_path)
    config_a.model_configs.seed = None
    config_b.model_configs.seed = None
    torch.manual_seed(1)
    a = DFLNodeAgent(dfl_node_agent_config=config_a)
    b = DFLNodeAgent(dfl_node_agent_config=config_b)
    differs = any(
        not torch.allclose(t, b.model.state_dict()[n])
        for n, t in a.model.state_dict().items()
    )
    assert differs


def test_neighbors_are_read_from_config_when_not_supplied(tmp_path):
    config = node_config("Node0", 0, 2, tmp_path)
    config.neighbors = OmegaConf.create(
        {
            "mode": "explicit",
            "send_to": ["Node1"],
            "recv_from": [{"node_id": "Node1", "server_uri": "b:50051"}],
        }
    )
    agent = DFLNodeAgent(dfl_node_agent_config=config)
    assert list(agent.neighbors.recv_from) == ["Node1"]
    assert agent.neighbors.recv_from["Node1"] == "b:50051"


def test_node_id_falls_back_to_client_id(tmp_path):
    config = node_config("Node0", 0, 2, tmp_path)
    del config["node_id"]
    config.client_id = "site-a"
    assert DFLNodeAgent(dfl_node_agent_config=config).get_id() == "site-a"


# ---------------------------------------------------------------------------------------
# round-tagged publication
# ---------------------------------------------------------------------------------------


def test_a_published_round_is_served_even_after_the_node_moves_on(tmp_path):
    """The property the whole barrier rests on: round r means round r, not 'latest'."""
    _, agents = build_federation(tmp_path, num_nodes=2)
    agent = agents["Node0"]

    agent.train()  # publishes round 0
    round_zero = {k: v.clone() for k, v in agent.get_parameters(round_id=0).items()}
    agent.aggregate_parameters({})  # an isolated mix, so only the round tag changes
    agent.train()  # publishes round 1

    served = agent.get_parameters(round_id=0)
    for name, tensor in round_zero.items():
        assert torch.allclose(served[name], tensor), "round 0 was overwritten by a later round"
    later = agent.get_parameters(round_id=1)["fc.weight"]
    assert not torch.allclose(later, round_zero["fc.weight"])


def test_aggregation_does_not_corrupt_what_was_already_published(tmp_path):
    """Aggregation loads new parameters into the model in place; the published copy must not
    follow, or a neighbor that has not collected yet receives the mixed model instead."""
    _, agents = build_federation(tmp_path, num_nodes=2)
    a, b = agents["Node0"], agents["Node1"]
    a.train()
    b.train()
    published = {k: v.clone() for k, v in a.get_parameters(round_id=0).items()}
    a.aggregate_parameters({"Node1": b.get_parameters(round_id=0)})
    for name, tensor in published.items():
        assert torch.allclose(a.get_parameters(round_id=0)[name], tensor)


def test_a_round_older_than_the_buffer_is_an_error_not_a_wrong_answer(tmp_path):
    _, agents = build_federation(tmp_path, num_nodes=2)
    agent = agents["Node0"]
    for _ in range(4):
        agent.train()
    with pytest.raises(ValueError, match="already moved past it"):
        agent.get_parameters(round_id=0)


def test_get_parameters_without_a_round_returns_the_latest(tmp_path):
    _, agents = build_federation(tmp_path, num_nodes=2)
    agent = agents["Node0"]
    agent.train()
    agent.train()
    assert torch.allclose(
        agent.get_parameters()["fc.weight"], agent.get_parameters(round_id=1)["fc.weight"]
    )


def test_a_future_resolves_when_the_round_is_published(tmp_path):
    """The non-blocking path, which no serial run uses and every concurrent one does."""
    _, agents = build_federation(tmp_path, num_nodes=2)
    agent = agents["Node0"]
    future = agent.get_parameters(round_id=0, blocking=False)
    assert not future.done()
    agent.train()
    assert future.done()
    published = agent.get_parameters(round_id=0)["fc.weight"]
    assert torch.allclose(future.result()["fc.weight"], published)


# ---------------------------------------------------------------------------------------
# access control and shutdown
# ---------------------------------------------------------------------------------------


def test_a_peer_outside_send_to_is_refused(tmp_path):
    config = node_config("Node0", 0, 2, tmp_path)
    config.neighbors = OmegaConf.create(
        {"mode": "explicit", "send_to": ["Node1"], "recv_from": [{"node_id": "Node1"}]}
    )
    agent = DFLNodeAgent(dfl_node_agent_config=config)
    agent.train()
    assert agent.get_parameters(round_id=0, requester_id="Node1") is not None
    with pytest.raises(PermissionError, match="does not serve"):
        agent.get_parameters(round_id=0, requester_id="Node9")


def test_a_local_call_is_never_access_checked(tmp_path):
    _, agents = build_federation(tmp_path, num_nodes=2)
    agent = agents["Node0"]
    agent.train()
    assert agent.get_parameters(round_id=0) is not None


def test_a_node_stops_serving_once_every_peer_it_serves_has_closed(tmp_path):
    _, agents = build_federation(tmp_path, num_nodes=3)
    agent = agents["Node0"]
    assert agent.server_terminated() is False
    agent.close_connection("Node1")
    assert agent.server_terminated() is False
    agent.close_connection("Node2")
    assert agent.server_terminated() is True


def test_closing_twice_does_not_count_twice(tmp_path):
    _, agents = build_federation(tmp_path, num_nodes=3)
    agent = agents["Node0"]
    agent.close_connection("Node1")
    agent.close_connection("Node1")
    assert agent.server_terminated() is False


# ---------------------------------------------------------------------------------------
# the concurrent path
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("num_nodes,rounds", [(3, 4), (4, 3)])
def test_nodes_running_in_parallel_stay_round_aligned(tmp_path, num_nodes, rounds):
    """Every node in its own thread, pulling from peers that may not have published yet.

    This is the case the serial launcher cannot reach: `get_parameters` must block until the
    requested round exists, the two-round buffer must be deep enough that a node one round
    ahead has not yet discarded what a slower peer still needs, and the lock must hold while
    all of that happens. On a fully connected graph every node must also end each round with
    identical parameters -- if the barrier leaks, they will not.
    """
    _, agents = build_federation(tmp_path, num_nodes=num_nodes)
    results = {node_id: [] for node_id in agents}
    errors = []

    def run(agent):
        try:
            for round_idx in range(rounds):
                agent.train()
                peers = {
                    peer_id: agents[peer_id].get_parameters(
                        round_id=round_idx, requester_id=agent.get_id()
                    )
                    for peer_id in agent.neighbors.recv_from
                }
                mixed = agent.aggregate_parameters(peers)
                results[agent.get_id()].append(
                    {k: v.clone() for k, v in mixed.items()}
                )
        except Exception as exc:  # noqa: BLE001 -- re-raised in the main thread below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(a,)) for a in agents.values()]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not [t for t in threads if t.is_alive()], "a node deadlocked waiting for a peer"
    assert not errors, f"a node raised: {errors[0]!r}"

    reference = results["Node0"]
    assert len(reference) == rounds
    for round_idx in range(rounds):
        for node_id, rounds_seen in results.items():
            for name, tensor in rounds_seen[round_idx].items():
                assert torch.allclose(tensor, reference[round_idx][name], atol=1e-6), (
                    f"{node_id} disagrees with Node0 at round {round_idx} on {name}"
                )


def test_parallel_nodes_on_a_ring_do_not_deadlock(tmp_path):
    """A sparse graph: nodes wait on different peers, so the wait pattern is not uniform."""
    _, agents = build_federation(tmp_path, num_nodes=5, topology_name="ring", k=1)
    errors = []

    def run(agent):
        try:
            for round_idx in range(3):
                agent.train()
                peers = {
                    peer_id: agents[peer_id].get_parameters(round_id=round_idx)
                    for peer_id in agent.neighbors.recv_from
                }
                agent.aggregate_parameters(peers)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(a,)) for a in agents.values()]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not [t for t in threads if t.is_alive()], "ring federation deadlocked"
    assert not errors, f"a node raised: {errors[0]!r}"


def test_latest_before_anything_is_published_waits_for_the_first_round(tmp_path):
    """`round_id=None` has no answer yet, so it waits on round 0 rather than failing."""
    _, agents = build_federation(tmp_path, num_nodes=2)
    agent = agents["Node0"]
    future = agent.get_parameters(blocking=False)
    assert not future.done()
    agent.train()
    assert future.done()
    assert torch.allclose(
        future.result()["fc.weight"], agent.get_parameters(round_id=0)["fc.weight"]
    )
