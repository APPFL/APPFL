"""Tests for the gRPC relay, run as several nodes and a relay inside one process."""

import socket
import threading
import time

import grpc
import pytest
import torch

from appfl.comm.grpc import GRPCPeerCommunicator, GRPCRelayServicer, serve_relay


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def start_relay(expected_nodes, max_message_size=2 * 1024 * 1024):
    uri = f"localhost:{free_port()}"
    servicer = GRPCRelayServicer(
        max_message_size=max_message_size, expected_nodes=expected_nodes, wait_timeout=30
    )
    thread = threading.Thread(
        target=serve_relay,
        args=(servicer, uri),
        kwargs={"max_workers": 16, "poll_seconds": 0.1},
        daemon=True,
    )
    thread.start()
    return uri, servicer, thread


def relay_node(node_id, peers, relay_uri):
    return GRPCPeerCommunicator(
        node_id=node_id,
        send_to=peers,
        recv_from={peer: relay_uri for peer in peers},
        relay_server_uri=relay_uri,
        connect_timeout=30,
        wait_timeout=30,
    )


def run_together(communicators, rounds, payload_of):
    results, errors = {n: [] for n in communicators}, []

    def run(node, c):
        try:
            c.start()
            for r in range(rounds):
                results[node].append(c.exchange(r, payload_of(node, r)))
        except Exception as exc:  # noqa: BLE001
            errors.append((node, exc))
        finally:
            try:
                c.shutdown()
            except Exception as exc:  # noqa: BLE001
                errors.append((node, exc))

    threads = [threading.Thread(target=run, args=(n, c)) for n, c in communicators.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=90)
    assert not [t for t in threads if t.is_alive()], "a node deadlocked"
    assert not errors, f"a node raised: {errors[0]}"
    return results


def test_two_nodes_exchange_through_a_relay_without_serving():
    """Neither node listens; the relay is the only thing either of them can be reached at."""
    relay_uri, _, thread = start_relay(expected_nodes=2)
    nodes = {
        "Node0": relay_node("Node0", ["Node1"], relay_uri),
        "Node1": relay_node("Node1", ["Node0"], relay_uri),
    }
    for c in nodes.values():
        assert not hasattr(c, "servicer"), "a relay node should not build a servicer"
    results = run_together(nodes, 2, lambda n, r: {"from": n, "round": r})
    for node, rounds in results.items():
        for r, received in enumerate(rounds):
            for peer, payload in received.items():
                assert payload["from"] == peer and payload["round"] == r
    thread.join(timeout=30)


def test_the_relay_refuses_a_collector_the_publisher_did_not_authorize():
    """The access list arrives with each publish, so the relay enforces a graph it never had."""
    relay_uri, servicer, _ = start_relay(expected_nodes=None)
    # Publishes only: giving it a recv_from would make it wait on a peer that never exists.
    publisher = GRPCPeerCommunicator(
        node_id="Node0", send_to=["Node1"], recv_from={},
        relay_server_uri=relay_uri, connect_timeout=30, wait_timeout=10,
    )  # authorizes Node1 only
    intruder = GRPCPeerCommunicator(
        node_id="Intruder", send_to=[], recv_from={"Node0": relay_uri},
        relay_server_uri=relay_uri, connect_timeout=30, wait_timeout=10,
    )
    publisher.start()
    intruder.start()
    try:
        publisher.exchange(0, {"w": torch.zeros(1)})  # publishes, declaring send_to=[Node1]
        with pytest.raises(grpc.RpcError) as caught:
            intruder.exchange(0, {"w": torch.zeros(1)})
        assert caught.value.code() == grpc.StatusCode.PERMISSION_DENIED
    finally:
        intruder.shutdown()
        publisher.shutdown()


def test_a_publisher_configured_larger_than_the_relay_is_accepted():
    """A publisher chunks by its own config, which the relay has no say in.

    Tying the relay's receive limit to its own chunk size would reject any node configured
    more generously than itself -- so the receive bound is a safety limit, set well above any
    plausible chunk, while the chunk size is what gets negotiated.
    """
    relay_uri, _, _ = start_relay(expected_nodes=None, max_message_size=64 * 1024)
    big = GRPCPeerCommunicator(
        node_id="Big", send_to=["Other"], recv_from={}, relay_server_uri=relay_uri,
        max_message_size=4 * 1024 * 1024, connect_timeout=30, wait_timeout=30,
    )
    big.start()
    try:
        # Far over the relay's 64 KiB chunk size; the publish must still be accepted.
        big.exchange(0, {"w": torch.zeros(300_000)})
    finally:
        big.shutdown()


def test_the_relay_needs_to_be_told_whose_payload_is_wanted():
    """Dialling a peer implies the target; dialling a relay does not."""
    relay_uri, servicer, _ = start_relay(expected_nodes=None)
    node = relay_node("Node0", ["Node1"], relay_uri)
    node.start()
    try:
        import yaml
        from appfl.comm.grpc.grpc_communicator_pb2 import (
            ClientHeader,
            GetGlobalModelRequest,
        )

        request = GetGlobalModelRequest(
            header=ClientHeader(client_id="Node0"),
            meta_data=yaml.dump({"round_id": 0}),  # no target
        )
        with pytest.raises(grpc.RpcError) as caught:
            list(node._stubs[relay_uri].GetGlobalModel(request, timeout=10))
        assert caught.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    finally:
        node.shutdown()


def test_an_intruder_that_asks_before_the_publish_is_still_refused():
    """The fast-path check cannot be the only one: a requester can arrive before the list does.

    ``send_to`` travels with each publish, so until ``Node0`` publishes there is no access list
    to check and the intruder's request is already parked in ``store.get``. The check *after*
    the wait is what refuses it -- delete that and asking early becomes a way in.
    """
    relay_uri, _, _ = start_relay(expected_nodes=None)
    publisher = GRPCPeerCommunicator(
        node_id="Node0", send_to=["Node1"], recv_from={},
        relay_server_uri=relay_uri, connect_timeout=30, wait_timeout=20,
    )
    intruder = GRPCPeerCommunicator(
        node_id="Intruder", send_to=[], recv_from={"Node0": relay_uri},
        relay_server_uri=relay_uri, connect_timeout=30, wait_timeout=20,
    )
    publisher.start()
    intruder.start()
    outcome = []

    def collect_before_anyone_published():
        try:
            intruder.exchange(0, {"w": torch.zeros(1)})
            outcome.append("SERVED")  # the bypass this test exists to catch
        except grpc.RpcError as exc:
            outcome.append(exc.code())

    asking = threading.Thread(target=collect_before_anyone_published, daemon=True)
    try:
        asking.start()
        time.sleep(1.0)  # let it park in store.get while the relay still has no list
        publisher.exchange(0, {"w": torch.zeros(1)})  # the list arrives now: send_to=[Node1]
        asking.join(timeout=30)
        assert not asking.is_alive(), "the intruder's request never returned"
        assert outcome == [grpc.StatusCode.PERMISSION_DENIED], (
            f"a requester that arrived before the access list was not refused: {outcome}"
        )
    finally:
        intruder.shutdown()
        publisher.shutdown()


def test_a_target_nobody_publishes_does_not_keep_the_relay_waiting():
    """Being asked about must not enrol a node as one the relay waits for.

    A typo in ``recv_from`` names a target that never publishes and never closes. If merely
    being asked about enrolled it, ``everyone_finished`` could never become true and the relay
    would outlive the federation it was serving.
    """
    relay_uri, servicer, _ = start_relay(expected_nodes=None)
    node = GRPCPeerCommunicator(
        node_id="Node0", send_to=[], recv_from={"Ghost": relay_uri},
        relay_server_uri=relay_uri, connect_timeout=30, wait_timeout=3,
    )
    node.start()
    try:
        with pytest.raises(grpc.RpcError):
            node.exchange(0, {"w": torch.zeros(1)})  # waits on Ghost, then gives up
    finally:
        node.shutdown()

    assert servicer.everyone_finished(), (
        "the relay is still waiting on a node that was only ever asked about"
    )


def test_the_relay_waits_for_collectors_not_only_publishers():
    """Every publisher finishing is not the relay's cue to stop.

    In a hybrid federation the relay-hosted nodes publish and close while a self-serving node
    is still collecting its last round through the relay. Counting only publishers lets the
    relay stop out from under a collector that still needs it.
    """
    relay_uri, servicer, _ = start_relay(expected_nodes=None)
    publisher = GRPCPeerCommunicator(
        node_id="Pub", send_to=["Col"], recv_from={},
        relay_server_uri=relay_uri, connect_timeout=30, wait_timeout=10,
    )
    collector = GRPCPeerCommunicator(  # serves itself; only collects through the relay
        node_id="Col", send_to=[], recv_from={"Pub": relay_uri},
        server_uri=f"localhost:{free_port()}", connect_timeout=30, wait_timeout=10,
    )
    publisher.start()
    collector.start()
    try:
        publisher.exchange(0, {"w": torch.zeros(1)})
        collector.exchange(0, {"w": torch.zeros(1)})
        publisher.shutdown()
        assert not servicer.everyone_finished(), (
            "the relay would stop while a collector has not finished with it"
        )
    finally:
        collector.shutdown()
    assert servicer.everyone_finished(), "the relay never became finished"
