"""Tests for `GRPCPeerCommunicator`, run as several peers inside one process.

Each peer gets its own gRPC server on its own port, so these are real round trips over the
wire rather than mocks -- without needing subprocess orchestration.
"""

import logging
import socket
import threading

import pytest
import torch

from appfl.comm.grpc import GRPCPeerCommunicator
from appfl.comm.grpc.utils import serialize_model


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def build(node_ids, edges, logger=None):
    """One communicator per node. `edges` is {node: [peers it collects from]}."""
    uris = {node: f"localhost:{free_port()}" for node in node_ids}
    communicators = {}
    for node in node_ids:
        communicators[node] = GRPCPeerCommunicator(
            node_id=node,
            server_uri=uris[node],
            # Whoever collects from me is whom I serve.
            send_to=[peer for peer in node_ids if node in edges.get(peer, [])],
            recv_from={peer: uris[peer] for peer in edges.get(node, [])},
            connect_timeout=30,
            wait_timeout=30,
            logger=logger,
        )
    return communicators


def run_concurrently(communicators, rounds, payload_of):
    """Every peer exchanges in its own thread; blocking is the point, so they must overlap."""
    results = {node: [] for node in communicators}
    errors = []

    def run(node, communicator):
        """One node's whole life, as its own process would live it.

        `start()` blocks until the peers it collects from are serving, and `shutdown()` blocks
        until the peers it serves have finished collecting. Both are deliberate, and both mean
        a node's lifecycle has to overlap its neighbors' -- so each node gets a thread, which
        is what separate processes give for free in a deployment.
        """
        try:
            communicator.start()
            for round_id in range(rounds):
                results[node].append(
                    communicator.exchange(round_id, payload_of(node, round_id))
                )
        except Exception as exc:  # noqa: BLE001 -- surfaced in the main thread
            errors.append((node, exc))
        finally:
            try:
                communicator.shutdown()
            except Exception as exc:  # noqa: BLE001
                errors.append((node, exc))

    threads = [
        threading.Thread(target=run, args=(node, c)) for node, c in communicators.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=90)
    alive = [t for t in threads if t.is_alive()]
    assert not alive, "a peer deadlocked"
    assert not errors, f"a peer raised: {errors[0]}"
    return results


def test_two_peers_exchange_over_the_wire():
    nodes = ["Node0", "Node1"]
    communicators = build(nodes, {"Node0": ["Node1"], "Node1": ["Node0"]})
    results = run_concurrently(
        communicators, 1, lambda node, r: {"w": torch.tensor([float(node[-1])])}
    )
    assert results["Node0"][0]["Node1"]["w"].item() == pytest.approx(1.0)
    assert results["Node1"][0]["Node0"]["w"].item() == pytest.approx(0.0)


def test_rounds_do_not_bleed_into_each_other():
    nodes = ["Node0", "Node1"]
    communicators = build(nodes, {"Node0": ["Node1"], "Node1": ["Node0"]})
    results = run_concurrently(communicators, 4, lambda node, r: {"from": node, "round": r})
    for node, rounds in results.items():
        for round_id, received in enumerate(rounds):
            for peer, payload in received.items():
                assert payload["round"] == round_id
                assert payload["from"] == peer


def test_the_payload_is_opaque():
    nodes = ["Node0", "Node1"]
    communicators = build(nodes, {"Node0": ["Node1"], "Node1": ["Node0"]})
    results = run_concurrently(
        communicators, 1, lambda node, r: {"note": f"hello from {node}", "items": [1, 2]}
    )
    assert results["Node0"][0]["Node1"]["note"] == "hello from Node1"
    assert results["Node0"][0]["Node1"]["items"] == [1, 2]


def test_a_ring_of_three_exchanges_only_with_its_neighbors():
    nodes = ["Node0", "Node1", "Node2"]
    edges = {"Node0": ["Node1", "Node2"], "Node1": ["Node0", "Node2"], "Node2": ["Node0", "Node1"]}
    communicators = build(nodes, edges)
    results = run_concurrently(communicators, 2, lambda node, r: {"from": node})
    for node, rounds in results.items():
        for received in rounds:
            assert sorted(received) == sorted(edges[node])


def test_a_peer_outside_send_to_is_refused():
    """The ACL is enforced at the transport, where the request arrives from the network."""
    import grpc

    uris = {"Node0": f"localhost:{free_port()}", "Intruder": f"localhost:{free_port()}"}
    server = GRPCPeerCommunicator(
        node_id="Node0",
        server_uri=uris["Node0"],
        send_to=["Node1"],  # not the intruder
        recv_from={},
        connect_timeout=30,
        wait_timeout=10,
    )
    intruder = GRPCPeerCommunicator(
        node_id="Intruder",
        server_uri=uris["Intruder"],
        send_to=[],
        recv_from={"Node0": uris["Node0"]},
        connect_timeout=30,
        wait_timeout=10,
    )
    server.start()
    intruder.start()
    try:
        server.store.publish(0, b"secret")
        with pytest.raises(grpc.RpcError) as caught:
            intruder.exchange(0, {"w": torch.zeros(1)})
        assert caught.value.code() == grpc.StatusCode.PERMISSION_DENIED
    finally:
        intruder.shutdown()
        server.shutdown()


def test_worker_count_is_derived_from_the_peers_served():
    communicator = GRPCPeerCommunicator(
        node_id="Node0", server_uri="localhost:1", send_to=[f"N{i}" for i in range(20)],
        recv_from={},
    )
    assert communicator.max_workers == 20 + GRPCPeerCommunicator.WORKER_HEADROOM

    small = GRPCPeerCommunicator(
        node_id="Node0", server_uri="localhost:1", send_to=["N1"], recv_from={}
    )
    assert small.max_workers == GRPCPeerCommunicator.MIN_WORKERS


def test_too_few_workers_warns_because_starvation_is_silent(caplog):
    with caplog.at_level(logging.WARNING):
        GRPCPeerCommunicator(
            node_id="Node0",
            server_uri="localhost:1",
            send_to=[f"N{i}" for i in range(8)],
            recv_from={},
            max_workers=4,
        )
    assert "stall with no error reported" in caplog.text


def test_peers_that_disagree_about_message_size_still_exchange():
    """Each site writes its own config, so a server's limit need not match its collector's.

    The chunk size has to follow the *receiver*, or the sender builds a chunk the receiver's
    channel refuses -- which surfaces as RESOURCE_EXHAUSTED only after the bytes are sent.
    """
    import torch

    uris = {"Big": f"localhost:{free_port()}", "Small": f"localhost:{free_port()}"}
    big = GRPCPeerCommunicator(
        node_id="Big", server_uri=uris["Big"], send_to=["Small"],
        recv_from={}, max_message_size=4 * 1024 * 1024,
        connect_timeout=30, wait_timeout=30,
    )
    small = GRPCPeerCommunicator(
        node_id="Small", server_uri=uris["Small"], send_to=[],
        recv_from={"Big": uris["Big"]}, max_message_size=64 * 1024,
        connect_timeout=30, wait_timeout=30,
    )
    big.start()
    small.start()
    try:
        # Comfortably over the collector's 64 KiB limit, well under the server's 4 MiB one.
        big.store.publish(0, serialize_model({"w": torch.zeros(200_000)}))
        received = small.exchange(0, {"w": torch.zeros(1)})
        assert received["Big"]["w"].shape == (200_000,)
    finally:
        small.shutdown()
        big.shutdown()
