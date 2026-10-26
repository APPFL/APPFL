"""Tests for `appfl.decentralized.neighbor`."""

import pytest
from omegaconf import OmegaConf

from appfl.decentralized.neighbor import (
    Neighbors,
    build_topology_from_config,
    generated_node_ids,
    resolve_all_neighbors_from_topology,
    resolve_neighbors,
)
from appfl.decentralized.topology import Custom, build_topology


def block(**kwargs):
    return OmegaConf.create(kwargs)


# ---------------------------------------------------------------------------------------
# from_topo
# ---------------------------------------------------------------------------------------


def test_from_topo_derives_neighbors_and_builds_its_own_graph():
    config = block(mode="from_topo", topology="ring", topology_kwargs={"k": 1}, num_nodes=6)
    view = resolve_neighbors(config, "Node0")
    assert list(view.recv_from) == ["Node1", "Node5"]
    assert view.send_to == ["Node1", "Node5"]


def test_from_topo_uses_a_supplied_graph_rather_than_rebuilding_it():
    config = block(mode="from_topo", topology="fully_connected", num_nodes=4)
    topology = build_topology("ring", generated_node_ids(4), k=1)
    # The passed graph wins: a launcher builds one for the whole federation, and each node's
    # view has to come from that same object rather than a private rebuild.
    view = resolve_neighbors(config, "Node0", topology=topology)
    assert list(view.recv_from) == ["Node1", "Node3"]


def test_from_topo_rejects_a_node_id_it_did_not_generate():
    config = block(mode="from_topo", topology="fully_connected", num_nodes=4)
    with pytest.raises(ValueError, match="not among the ids"):
        resolve_neighbors(config, "site-a")


def test_from_topo_produces_symmetric_lists_for_an_undirected_graph():
    views = resolve_all_neighbors_from_topology(build_topology("star", generated_node_ids(5)))
    for node_id, view in views.items():
        assert view.send_to == list(view.recv_from)


def test_from_topo_produces_asymmetric_lists_for_a_directed_graph():
    ids = generated_node_ids(3)
    cycle = Custom(ids, directed=True, edges=[[ids[i], ids[(i + 1) % 3]] for i in range(3)])
    views = resolve_all_neighbors_from_topology(cycle)
    # Node0 sends to Node1 and receives from Node2 -- the distinction the two lists exist for.
    assert views["Node0"].send_to == ["Node1"]
    assert list(views["Node0"].recv_from) == ["Node2"]


def test_build_topology_from_config_refuses_an_explicit_block():
    with pytest.raises(ValueError, match="only 'from_topo' names a graph"):
        build_topology_from_config(block(mode="explicit", send_to=[], recv_from=[]))


def test_generated_node_ids_matches_what_from_topo_assumes():
    assert generated_node_ids(3) == ["Node0", "Node1", "Node2"]


# ---------------------------------------------------------------------------------------
# explicit
# ---------------------------------------------------------------------------------------


def test_explicit_parses_ids_endpoints_and_weights():
    config = block(
        mode="explicit",
        send_to=["Node1"],
        recv_from=[{"node_id": "Node1", "server_uri": "b:50051", "weight": 0.25}],
    )
    view = resolve_neighbors(config, "Node0")
    assert list(view.recv_from) == ["Node1"]
    assert view.recv_from["Node1"] == "b:50051"
    assert view.weights["Node1"] == 0.25
    assert "Node9" not in view.recv_from


def test_explicit_accepts_a_bare_id_for_relay_mode():
    """In relay mode the peer is named, not addressed -- the relay is the only endpoint."""
    view = resolve_neighbors(
        block(mode="explicit", send_to=["Node1"], recv_from=["Node1"]), "Node0"
    )
    assert list(view.recv_from) == ["Node1"]
    assert view.recv_from["Node1"] is None


def test_missing_mode_is_rejected_with_both_options_named():
    with pytest.raises(ValueError, match="has no `mode`"):
        resolve_neighbors(block(send_to=["Node1"]), "Node0")


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="not understood"):
        resolve_neighbors(block(mode="gossip"), "Node0")


def test_absent_block_gives_an_empty_view():
    view = resolve_neighbors(None, "Node0")
    assert view.recv_from == {} and view.send_to == []


# ---------------------------------------------------------------------------------------
# directed declarations
# ---------------------------------------------------------------------------------------


def test_undeclared_asymmetry_is_rejected():
    """The realistic mistake: a peer added to recv_from and forgotten in send_to."""
    config = block(
        mode="explicit",
        send_to=["Node1"],
        recv_from=[{"node_id": "Node1"}, {"node_id": "Node2"}],
    )
    with pytest.raises(ValueError, match="directed exchange"):
        resolve_neighbors(config, "Node0")


def test_declared_asymmetry_is_accepted():
    config = block(
        mode="explicit",
        directed=True,
        send_to=["Node1"],
        recv_from=[{"node_id": "Node2"}],
    )
    view = resolve_neighbors(config, "Node0")
    assert view.send_to == ["Node1"] and list(view.recv_from) == ["Node2"]


def test_empty_send_to_is_unrestricted_not_asymmetric():
    """The simulation default: no access list, so nothing to disagree with."""
    view = resolve_neighbors(
        block(mode="explicit", recv_from=[{"node_id": "Node1"}]), "Node0"
    )
    assert view.may_serve("anyone") is True


def test_send_to_is_enforced_when_it_is_given():
    view = resolve_neighbors(
        block(mode="explicit", send_to=["Node1"], recv_from=[{"node_id": "Node1"}]), "Node0"
    )
    assert view.may_serve("Node1") is True
    assert view.may_serve("Node9") is False


# ---------------------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------------------


def test_weights_default_to_uniform_over_the_closed_neighborhood():
    view = Neighbors(
        send_to=["B", "C"],
        recv_from={"B": None, "C": None},
    )
    third = pytest.approx(1 / 3)
    assert view.mixing_weights("A") == {"B": third, "C": third, "A": third}


def test_explicit_weights_leave_the_remainder_to_the_node_itself():
    view = Neighbors(
        send_to=["B"], recv_from={"B": None}, weights={"B": 0.25}
    )
    assert view.mixing_weights("A") == {"B": 0.25, "A": pytest.approx(0.75)}


def test_partially_specified_weights_are_rejected():
    view = Neighbors(
        send_to=["B", "C"],
        recv_from={"B": None, "C": None},
        weights={"B": 0.25},
    )
    with pytest.raises(ValueError, match="weights are set for some peers"):
        view.mixing_weights("A")


def test_weights_that_leave_nothing_for_the_node_are_rejected():
    view = Neighbors(
        send_to=["B"], recv_from={"B": None}, weights={"B": 1.5}
    )
    with pytest.raises(ValueError, match="no weight for"):
        view.mixing_weights("A")


def test_an_isolated_node_keeps_all_of_its_own_weight():
    assert Neighbors().mixing_weights("A") == {"A": 1.0}


def test_a_weight_for_a_peer_that_sends_nothing_is_rejected():
    """The one risk of keeping weights in their own mapping: it can name a peer `recv_from`
    does not, and a weight with nothing to apply to is a mistake rather than a no-op."""
    view = Neighbors(send_to=["B"], recv_from={"B": None}, weights={"B": 0.3, "Ghost": 0.2})
    with pytest.raises(ValueError, match="which this node does not"):
        view.mixing_weights("A")
