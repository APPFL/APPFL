"""Tests for `appfl.decentralized.topology`."""

import math

import pytest

from appfl.decentralized.topology import (
    Custom,
    FullyConnected,
    Ring,
    Star,
    Topology,
    build_topology,
)

IDS = [f"Node{i}" for i in range(6)]


def test_build_topology_resolves_every_registered_name():
    for name in ("fully_connected", "complete", "ring", "star", "random_geometric", "custom"):
        kwargs = (
            {"edges": [["Node0", "Node1"], ["Node1", "Node0"]]} if name == "custom" else {}
        )
        assert isinstance(build_topology(name, IDS, **kwargs), Topology)


def test_unknown_topology_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown topology"):
        build_topology("mesh", IDS)


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("fully_connected", {}),
        ("ring", {"k": 1}),
        ("ring", {"k": 2}),
        ("star", {}),
        ("random_geometric", {"radius": 0.5, "seed": 0}),
    ],
)
def test_named_topologies_are_undirected(name, kwargs):
    """Every neighbor relation is symmetric, and in/out neighbors coincide."""
    topology = build_topology(name, IDS, **kwargs)
    assert topology.directed is False
    for node in IDS:
        assert topology.in_neighbors(node) == topology.neighbors(node)
        assert topology.out_neighbors(node) == topology.neighbors(node)
        for peer in topology.neighbors(node):
            assert node in topology.neighbors(peer)


def test_no_topology_makes_a_node_its_own_neighbor():
    for name, kwargs in [("fully_connected", {}), ("ring", {"k": 2}), ("star", {})]:
        topology = build_topology(name, IDS, **kwargs)
        for node in IDS:
            assert node not in topology.neighbors(node)


def test_fully_connected_fiedler_equals_node_count():
    for n in (3, 5, 9):
        ids = [f"N{i}" for i in range(n)]
        assert build_topology("fully_connected", ids).fiedler_value() == pytest.approx(n)


def test_ring_connectivity_decays_with_size_and_grows_with_k():
    """The property that makes a ring the slow-mixing case, and `k` the dial against it."""
    small = build_topology("ring", [f"N{i}" for i in range(4)]).fiedler_value()
    large = build_topology("ring", [f"N{i}" for i in range(32)]).fiedler_value()
    assert large < small
    wider = build_topology("ring", IDS, k=2).fiedler_value()
    assert wider > build_topology("ring", IDS, k=1).fiedler_value()


def test_star_stays_at_fiedler_one_however_large():
    """Unlike a ring, a star does not get harder to mix as it grows -- only more lopsided."""
    for n in (4, 8, 32):
        ids = [f"N{i}" for i in range(n)]
        topology = build_topology("star", ids)
        assert topology.fiedler_value() == pytest.approx(1.0)
        assert topology.degree(ids[0]) == n - 1
        assert topology.degree(ids[1]) == 1
        # The hub spreads its weight over everyone; a spoke splits its own with the hub.
        assert topology.mixing_weight(ids[0], ids[1]) == pytest.approx(1.0 / n)
        assert topology.mixing_weight(ids[1], ids[0]) == pytest.approx(0.5)


def test_star_rejects_a_hub_that_is_not_a_node():
    with pytest.raises(ValueError, match="not one of"):
        Star(IDS, hub="NotANode")


def test_mixing_weights_sum_to_one_over_the_closed_neighborhood():
    for name, kwargs in [("ring", {"k": 2}), ("star", {}), ("random_geometric", {"radius": 0.4})]:
        topology = build_topology(name, IDS, **kwargs)
        for node in IDS:
            total = topology.mixing_weight(node, node) + sum(
                topology.mixing_weight(node, peer) for peer in topology.in_neighbors(node)
            )
            assert total == pytest.approx(1.0)


def test_mixing_weight_is_zero_for_a_non_neighbor():
    ring = build_topology("ring", IDS, k=1)
    assert ring.mixing_weight("Node0", "Node3") == 0.0


def test_random_geometric_is_always_connected():
    """A raw radius graph is often disconnected; the class bridges components on purpose."""
    for seed in range(8):
        topology = build_topology("random_geometric", IDS, radius=0.05, seed=seed)
        assert topology.is_connected()


def test_random_geometric_draws_at_its_real_positions():
    topology = build_topology("random_geometric", IDS, radius=0.5, seed=1)
    layout = topology.layout()
    assert set(layout) == set(IDS)
    # Not the circular fallback: points do not sit on the unit circle.
    assert any(abs(math.hypot(x, y) - 1.0) > 1e-6 for x, y in layout.values())


def test_star_puts_the_hub_in_the_middle():
    assert build_topology("star", IDS).layout()["Node0"] == (0.0, 0.0)


# ---------------------------------------------------------------------------------------
# Custom, undirected and directed
# ---------------------------------------------------------------------------------------


def test_custom_symmetrizes_one_way_edges_and_says_so():
    with pytest.warns(UserWarning, match="one direction only"):
        topology = Custom(IDS, edges=[["Node0", "Node1"], ["Node1", "Node2"]])
    assert topology.neighbors("Node2") == ["Node1"]
    assert topology.neighbors("Node0") == ["Node1"]


def test_custom_stays_silent_when_the_input_was_already_symmetric(recwarn):
    Custom(IDS, edges=[["Node0", "Node1"], ["Node1", "Node0"]])
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_custom_repairs_an_inconsistent_adjacency_map():
    with pytest.warns(UserWarning):
        topology = Custom(IDS, adjacency={"Node0": ["Node1"], "Node1": [], "Node2": []})
    assert topology.neighbors("Node1") == ["Node0"]


def test_custom_directed_keeps_edges_one_way():
    topology = Custom(IDS[:3], directed=True, edges=[["Node0", "Node1"], ["Node1", "Node2"]])
    assert topology.directed is True
    assert topology.out_neighbors("Node0") == ["Node1"]
    assert topology.in_neighbors("Node0") == []
    assert topology.in_neighbors("Node1") == ["Node0"]
    # `neighbors` is the union, for anything that only cares who is involved.
    assert set(topology.neighbors("Node1")) == {"Node0", "Node2"}


def test_directed_needs_strong_connectivity_not_symmetrized_connectivity():
    """The check that a symmetrized Fiedler value would get wrong."""
    chain = Custom(IDS[:3], directed=True, edges=[["Node0", "Node1"], ["Node1", "Node2"]])
    assert chain.is_connected() is False
    assert chain.is_strongly_connected() is False
    assert chain.fiedler_value() > 0  # symmetrized, it looks connected -- and is not

    cycle = Custom(
        IDS[:3],
        directed=True,
        edges=[["Node0", "Node1"], ["Node1", "Node2"], ["Node2", "Node0"]],
    )
    assert cycle.is_connected() is True


def test_disconnected_undirected_graph_is_detected():
    topology = Custom(
        IDS[:4],
        edges=[["Node0", "Node1"], ["Node2", "Node3"]],
        warn_on_repair=False,
    )
    assert topology.is_connected() is False
    assert topology.fiedler_value() == pytest.approx(0.0)


def test_custom_rejects_malformed_input():
    with pytest.raises(ValueError, match="unknown node"):
        Custom(IDS, edges=[["Node0", "Ghost"]])
    with pytest.raises(ValueError, match="its own neighbor"):
        Custom(IDS, edges=[["Node0", "Node0"]])
    with pytest.raises(ValueError, match="exactly one of"):
        Custom(IDS)
    with pytest.raises(ValueError, match="exactly one of"):
        Custom(IDS, edges=[["Node0", "Node1"]], adjacency={"Node0": ["Node1"]})


# ---------------------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------------------


def test_describe_reports_the_connectivity_measure_that_applies():
    undirected = build_topology("ring", IDS)
    assert "fiedler_value" in undirected.describe()
    assert "strongly_connected" not in undirected.describe()

    directed = Custom(IDS[:3], directed=True, edges=[["Node0", "Node1"]])
    assert "strongly_connected" in directed.describe()
    assert "fiedler_value" not in directed.describe()


def test_edge_count_counts_each_undirected_edge_once():
    n = len(IDS)
    assert build_topology("fully_connected", IDS).describe()["n_edges"] == n * (n - 1) // 2
    assert build_topology("ring", IDS, k=1).describe()["n_edges"] == len(IDS)
    directed = Custom(IDS[:3], directed=True, edges=[["Node0", "Node1"], ["Node1", "Node2"]])
    assert directed.describe()["n_edges"] == 2


def test_to_ascii_renders_a_matrix_and_gives_up_when_it_would_be_unreadable():
    matrix = build_topology("ring", IDS).to_ascii()
    assert "row sends to column" in matrix
    assert len(matrix.splitlines()) == len(IDS) + 3

    big = build_topology("ring", [f"N{i}" for i in range(40)]).to_ascii()
    assert "matrix omitted" in big


def test_draw_produces_axes_and_a_file(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    path = tmp_path / "topology.png"
    ax = build_topology("ring", IDS).draw(path=str(path))
    assert path.exists() and path.stat().st_size > 0
    assert ax.get_title().startswith("Ring")


def test_draw_handles_a_directed_graph_and_long_labels(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    sites = ["anl-alcf", "ornl-olcf", "nersc"]
    topology = Custom(
        sites,
        directed=True,
        edges=[["anl-alcf", "ornl-olcf"], ["ornl-olcf", "nersc"], ["nersc", "anl-alcf"]],
    )
    path = tmp_path / "directed.png"
    topology.draw(path=str(path))
    assert path.exists()


def test_single_node_topology_degenerates_without_error():
    topology = FullyConnected(["Only"])
    assert topology.neighbors("Only") == []
    assert topology.is_connected() is True
    assert topology.layout() == {"Only": (0.0, 0.0)}


def test_ring_with_k_larger_than_the_ring_is_fully_connected():
    ring = Ring(IDS, k=len(IDS))
    assert set(ring.neighbors("Node0")) == set(IDS) - {"Node0"}
