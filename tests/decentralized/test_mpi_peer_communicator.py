"""Tests for `MPIPeerCommunicator`, run under MPI.

    mpirun -n 2 python -m pytest tests/decentralized/test_mpi_peer_communicator.py --with-mpi

Written to be meaningful at two ranks, which is the smallest federation where an exchange
happens at all and what CI runs.
"""

import pytest
import torch

mpi4py = pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

from appfl.comm.mpi import MPIPeerCommunicator  # noqa: E402

pytestmark = pytest.mark.mpi


@pytest.fixture
def world():
    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("needs at least 2 ranks")
    ids = [f"Node{i}" for i in range(comm.Get_size())]
    return comm, ids, ids[comm.Get_rank()]


def fully_connected(me, ids):
    return [n for n in ids if n != me]


def test_every_peer_receives_what_the_others_sent(world):
    comm, ids, me = world
    peers = fully_connected(me, ids)
    communicator = MPIPeerCommunicator(
        comm, node_id=me, node_ids=ids, send_to=peers, recv_from=peers
    )
    # A payload only this rank could have produced, so a crossed wire is visible.
    received = communicator.exchange(0, {"w": torch.tensor([float(comm.Get_rank())])})
    assert sorted(received) == sorted(peers)
    for peer_id, payload in received.items():
        assert payload["w"].item() == pytest.approx(float(ids.index(peer_id)))


def test_the_payload_is_opaque(world):
    """Not model-specific: anything picklable goes through, which is what lets a different
    decentralized method reuse this without changing it."""
    comm, ids, me = world
    peers = fully_connected(me, ids)
    communicator = MPIPeerCommunicator(
        comm, node_id=me, node_ids=ids, send_to=peers, recv_from=peers
    )
    received = communicator.exchange(0, {"from": me, "round": 0, "extra": [1, 2, 3]})
    for peer_id, payload in received.items():
        assert payload["from"] == peer_id
        assert payload["extra"] == [1, 2, 3]


def test_consecutive_rounds_do_not_bleed_into_each_other(world):
    """The tag has to separate rounds, or a fast peer's next payload answers this round."""
    comm, ids, me = world
    peers = fully_connected(me, ids)
    communicator = MPIPeerCommunicator(
        comm, node_id=me, node_ids=ids, send_to=peers, recv_from=peers
    )
    for round_id in range(5):
        received = communicator.exchange(round_id, {"round": round_id, "from": me})
        for payload in received.values():
            assert payload["round"] == round_id


def test_a_directed_pair_exchanges_one_way(world):
    """send_to and recv_from need not match: rank 0 speaks, rank 1 listens."""
    comm, ids, me = world
    if comm.Get_size() != 2:
        pytest.skip("this case is written for exactly 2 ranks")
    speaker, listener = ids[0], ids[1]
    communicator = MPIPeerCommunicator(
        comm,
        node_id=me,
        node_ids=ids,
        send_to=[listener] if me == speaker else [],
        recv_from=[speaker] if me == listener else [],
    )
    received = communicator.exchange(0, {"from": me})
    assert received == {} if me == speaker else received["Node0"]["from"] == speaker


def test_rank_count_must_match_the_roster(world):
    comm, ids, me = world
    with pytest.raises(ValueError, match="rank\\(s\\) but the federation has"):
        MPIPeerCommunicator(
            comm, node_id=me, node_ids=ids + ["Extra"], send_to=[], recv_from=[]
        )


def test_identity_must_match_this_rank(world):
    comm, ids, me = world
    other = ids[(comm.Get_rank() + 1) % len(ids)]
    with pytest.raises(ValueError, match="in the roster but this process is rank"):
        MPIPeerCommunicator(comm, node_id=other, node_ids=ids, send_to=[], recv_from=[])


def test_an_unknown_peer_is_rejected(world):
    comm, ids, me = world
    with pytest.raises(ValueError, match="not in the roster"):
        MPIPeerCommunicator(
            comm, node_id=me, node_ids=ids, send_to=["Ghost"], recv_from=[]
        )


def test_an_oversized_payload_is_refused_rather_than_failing_inside_mpi(
    world, monkeypatch
):
    comm, ids, me = world
    peers = fully_connected(me, ids)
    communicator = MPIPeerCommunicator(
        comm, node_id=me, node_ids=ids, send_to=peers, recv_from=peers
    )
    # Shrunk below any real payload so the guard is reachable. Every rank raises before
    # sending anything, so no peer is left waiting on a message that never comes.
    monkeypatch.setattr(MPIPeerCommunicator, "MAX_MESSAGE_BYTES", 8)
    with pytest.raises(ValueError, match="needs chunking"):
        communicator.exchange(0, {"w": torch.zeros(100)})
