"""Unit tests for registering client sample sizes (appfl.vsim.register_sample_sizes).

Aggregators weight by data size only for clients whose size was registered, and
fall back to equal weights for the rest without a warning, so these tests check
that registration reaches a real ServerAgent's aggregator and changes the result.
"""

import pytest
import torch
from omegaconf import OmegaConf

from appfl.agent import ServerAgent
from appfl.vsim import register_sample_sizes


class _SizedClient:
    def __init__(self, cid, size):
        self.cid, self.size = cid, size

    def get_id(self):
        return self.cid

    def get_sample_size(self):
        return self.size


class _RecordingServer:
    """Exposes only the public `set_sample_size`, recording every call."""

    def __init__(self):
        self.calls = []

    def set_sample_size(self, client_id, sample_size, **kwargs):
        self.calls.append((client_id, sample_size, kwargs))


def _make_server(client_weights_mode, num_clients):
    """A real ServerAgent with only a scheduler and aggregator: no model or data."""
    config = OmegaConf.create(
        {
            "client_configs": {},
            "server_configs": {
                "num_clients": num_clients,
                "num_global_epochs": 1,
                "scheduler": "SyncScheduler",
                "aggregator": "FedAvgAggregator",
                "aggregator_kwargs": {"client_weights_mode": client_weights_mode},
            },
        }
    )
    return ServerAgent(server_agent_config=config)


def test_every_client_is_registered_without_blocking():
    server = _RecordingServer()
    clients = [
        _SizedClient("C0", 50),
        _SizedClient("C1", 300),
        _SizedClient("C2", 1200),
    ]

    sizes = register_sample_sizes(server, clients)

    assert list(sizes.items()) == [("C0", 50), ("C1", 300), ("C2", 1200)]
    # sync=True would hand each client a Future that resolves only once all of
    # them have reported, which a single process can't reach while blocking.
    assert server.calls == [
        ("C0", 50, {"sync": False}),
        ("C1", 300, {"sync": False}),
        ("C2", 1200, {"sync": False}),
    ]


@pytest.mark.parametrize("mode, expected", [("sample_size", 3.0), ("equal", 2.0)])
def test_registered_sizes_reach_the_aggregator(mode, expected):
    """The sizes land on the aggregator, which uses them only in sample_size mode."""
    server = _make_server(mode, num_clients=2)
    clients = [_SizedClient("Client1", 100), _SizedClient("Client2", 300)]

    register_sample_sizes(server, clients)
    assert server.aggregator.client_sample_size == {"Client1": 100, "Client2": 300}

    first = server.global_update("Client1", {"w": torch.tensor([0.0])}, blocking=False)
    last = server.global_update("Client2", {"w": torch.tensor([4.0])}, blocking=False)
    # Both models are in, so the round has fired and result() cannot block.
    assert first.done() and last.done()
    # 100:300 gives 0.25 * 0 + 0.75 * 4 = 3; equal weights give the plain mean, 2.
    assert last.result()["w"].item() == pytest.approx(expected)
