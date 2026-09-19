"""Calibration reports, suppression, and pooled reference checks."""

import json
from copy import deepcopy

import numpy as np
import pytest

from appfl.misc.data_readiness.calibration import (
    aggregate_calibration_reports,
    build_calibration_report,
)


def _aggregate(report, **kwargs):
    return aggregate_calibration_reports({"site": report}, **kwargs)


def _report():
    return build_calibration_report([0.2] * 5 + [0.8] * 4, [0] * 5 + [1] * 4, ["A"] * 9)


def test_suppression_happens_before_serialization():
    report = json.loads(json.dumps(_report(), allow_nan=False))
    assert set(report) == {
        "schema_version",
        "n_bins",
        "min_cell_count",
        "min_outcome_count",
        "cells",
    }
    assert report["schema_version"] == 1
    assert set(report["cells"]["A"]) == {"2"}
    cell = report["cells"]["A"]["2"]
    assert set(cell) == {"n", "sum_p", "sum_y", "sum_sq_err"}
    assert cell["n"] == 5
    assert cell["sum_p"] == pytest.approx(1.0)
    assert cell["sum_y"] == 0
    assert cell["sum_sq_err"] == pytest.approx(0.2)
    result = _aggregate(report)
    assert result["scope"] == "retained_patients"
    assert result["groups"]["A"]["n"] == 5
    assert result["groups"]["A"]["ece"] == pytest.approx(0.2)
    assert result["groups"]["A"]["brier"] == pytest.approx(0.04)


@pytest.mark.parametrize("empty_input", [False, True])
def test_empty_and_fully_suppressed_reports_are_unavailable(empty_input):
    values = ([], [], []) if empty_input else ([0.2], [0], ["A"])
    report = build_calibration_report(*values)
    assert report["cells"] == {}
    result = _aggregate(report)
    assert result["status"] == "unavailable"
    assert result["groups"] == {}


@pytest.mark.parametrize("minimum", [0, -1, True, 1.5, "5"])
def test_invalid_cell_threshold(minimum):
    with pytest.raises(ValueError):
        build_calibration_report([0.5], [0], ["A"], min_cell_count=minimum)


@pytest.mark.parametrize("bins", [0, -1, True, 1.5, "10"])
def test_invalid_bin_count(bins):
    with pytest.raises(ValueError):
        build_calibration_report([0.5], [0], ["A"], n_bins=bins)


@pytest.mark.parametrize("bins", [0, -1, 3, 20, True, 2.5, "5"])
def test_invalid_release_bin_count(bins):
    with pytest.raises(ValueError):
        build_calibration_report([0.5], [0], ["A"], release_n_bins=bins)


@pytest.mark.parametrize("minimum", [-1, True, 1.5, "1"])
def test_invalid_outcome_threshold(minimum):
    with pytest.raises(ValueError):
        build_calibration_report([0.5], [0], ["A"], min_outcome_count=minimum)


def test_merging_precedes_suppression_and_preserves_inputs():
    predictions = np.array([0.05] * 3 + [0.15] * 3 + [0.95])
    outcomes = np.array([0, 0, 1, 0, 1, 0, 1])
    groups = np.array(["A"] * 7)
    originals = [array.copy() for array in (predictions, outcomes, groups)]
    assert build_calibration_report(predictions, outcomes, groups)["cells"] == {}
    merged = build_calibration_report(predictions, outcomes, groups, release_n_bins=5)
    for original, array in zip(originals, (predictions, outcomes, groups)):
        np.testing.assert_array_equal(array, original)
    assert merged["n_bins"] == 5
    assert set(merged["cells"]["A"]) == {"0"}
    assert merged["cells"]["A"]["0"]["n"] == 6
    assert _aggregate(merged)["groups"]["A"]["n"] == 6


def test_sites_cannot_combine_sparse_cells_to_pass_the_local_cutoff():
    report = build_calibration_report(
        [0.15] * 3, [0, 1, 0], ["A"] * 3, release_n_bins=2
    )
    result = aggregate_calibration_reports({"first": report, "second": report})
    assert report["cells"] == {}
    assert result["status"] == "unavailable"


def test_coarsening_can_hide_calibration_error_but_preserves_brier():
    values = ([0.05, 0.15], [0, 1], ["A", "A"])
    fine = _aggregate(build_calibration_report(*values, min_cell_count=1))
    coarse = _aggregate(
        build_calibration_report(*values, min_cell_count=1, release_n_bins=5)
    )
    assert fine["groups"]["A"]["ece"] == pytest.approx(0.45)
    assert coarse["groups"]["A"]["ece"] == pytest.approx(0.4)
    assert fine["groups"]["A"]["brier"] == pytest.approx(coarse["groups"]["A"]["brier"])


def test_endpoints_and_subgroups_remain_separate():
    report = build_calibration_report(
        [0.0, 1.0, 0.1],
        [0, 1, 0],
        ["A", "A", "B"],
        min_cell_count=1,
        release_n_bins=5,
    )
    assert set(report["cells"]["A"]) == {"0", "4"}
    assert set(report["cells"]["B"]) == {"0"}
    assert report["cells"]["A"]["0"]["n"] == 1
    assert report["cells"]["B"]["0"]["n"] == 1
    result = _aggregate(report)
    assert result["groups"]["A"]["ece"] == 0
    assert result["groups"]["A"]["brier"] == 0


@pytest.mark.parametrize(
    "n_bins,released,boundary,below",
    [
        (100, 100, 0.29, 0.289),
        (100, 50, 0.58, 0.579),
        (10, 10, 0.9, np.nextafter(0.9, 0)),
    ],
)
def test_shared_bin_boundaries_match_retained_patient_reference(
    n_bins, released, boundary, below
):
    predictions = np.array([below] * 5 + [boundary] * 5 + [0.0, 1.0])
    outcomes = np.array([0] * 5 + [1] * 5 + [0, 1])
    report = build_calibration_report(
        predictions,
        outcomes,
        ["A"] * len(predictions),
        n_bins=n_bins,
        release_n_bins=released,
        min_cell_count=1,
    )
    cells = report["cells"]["A"]
    expected_ece = 0.0
    for index in range(released):
        mask = (predictions >= index / released) & (
            (predictions < (index + 1) / released)
            | ((index == released - 1) & (predictions == 1))
        )
        if not mask.any():
            assert str(index) not in cells
            continue
        assert cells[str(index)]["n"] == int(mask.sum())
        expected_ece += abs(float(np.sum(predictions[mask] - outcomes[mask])))
    result = _aggregate(report)["groups"]["A"]
    assert result["n"] == len(predictions)
    assert result["ece"] == pytest.approx(expected_ece / len(predictions))
    assert result["brier"] == pytest.approx(np.mean((predictions - outcomes) ** 2))


def test_outcome_guard_checks_both_classes_at_the_boundary():
    values = (
        [0.05] * 10 + [0.25] * 5 + [0.45] * 5 + [0.65] * 10,
        [0] * 10 + [0, 0, 0, 0, 1] + [0, 0, 0, 1, 1] + [1] * 10,
        ["A"] * 30,
    )
    one = build_calibration_report(*values, min_outcome_count=1)
    two = build_calibration_report(*values, min_outcome_count=2)
    assert set(one["cells"]["A"]) == {"2", "4"}
    assert set(two["cells"]["A"]) == {"4"}
    one["min_outcome_count"] = 2
    with pytest.raises(ValueError):
        _aggregate(one)


def test_outcome_guard_is_applied_after_merging():
    report = build_calibration_report(
        [0.05] * 3 + [0.15] * 3,
        [0] * 3 + [1] * 3,
        ["A"] * 6,
        release_n_bins=5,
        min_outcome_count=3,
    )
    assert report["cells"]["A"]["0"]["n"] == 6
    assert report["cells"]["A"]["0"]["sum_y"] == 3


@pytest.mark.parametrize(
    "key,value", [("n_bins", 5), ("min_cell_count", 10), ("min_outcome_count", 1)]
)
@pytest.mark.parametrize("empty", [False, True])
def test_mixed_policies_are_rejected_including_empty_reports(key, value, empty):
    first = build_calibration_report([], [], []) if empty else _report()
    second = deepcopy(first)
    second[key] = value
    with pytest.raises(ValueError):
        aggregate_calibration_reports({"first": first, "second": second})


@pytest.mark.parametrize(
    "field,value",
    [
        ("n", 4),
        ("n", 5.0),
        ("n", True),
        ("n", -1),
        ("sum_p", np.nan),
        ("sum_p", np.inf),
        ("sum_p", -1),
        ("sum_p", 3.0),
        ("sum_p", "1.0"),
        ("sum_y", 0.5),
        ("sum_y", -1),
        ("sum_y", 6),
        ("sum_sq_err", np.nan),
        ("sum_sq_err", -1),
        ("sum_sq_err", 6),
    ],
)
def test_invalid_cell_statistics_are_rejected(field, value):
    report = _report()
    report["cells"]["A"]["2"][field] = value
    with pytest.raises(ValueError):
        _aggregate(report)


@pytest.mark.parametrize("index", ["-1", "10", "2.0", "02", 2])
def test_invalid_bin_keys_are_rejected(index):
    report = _report()
    report["cells"]["A"][index] = report["cells"]["A"].pop("2")
    with pytest.raises(ValueError):
        _aggregate(report)


@pytest.mark.parametrize("version", [True, 1.0, 2, None])
def test_invalid_schema_versions_are_rejected(version):
    report = _report()
    report["schema_version"] = version
    with pytest.raises(ValueError):
        _aggregate(report)


@pytest.mark.parametrize(
    "field", ["full_n", "suppressed_n", "client_id", "predictions"]
)
def test_undeclared_payload_fields_are_rejected(field):
    report = _report()
    report[field] = 9
    with pytest.raises(ValueError):
        _aggregate(report)


@pytest.mark.parametrize(
    "predictions,outcomes,groups",
    [
        ([np.nan], [0], ["A"]),
        ([np.inf], [0], ["A"]),
        ([-0.1], [0], ["A"]),
        ([1.1], [0], ["A"]),
        ([0.5], [2], ["A"]),
        ([0.5], [0.5], ["A"]),
        ([0.5], [np.nan], ["A"]),
        ([0.5, 0.6], [0], ["A"]),
        ([0.5], [0], []),
        ([[0.5]], [0], ["A"]),
        ([0.5], [[0]], ["A"]),
        ([0.5], [0], [["A"]]),
    ],
)
def test_invalid_patient_arrays_are_rejected(predictions, outcomes, groups):
    with pytest.raises(ValueError):
        build_calibration_report(predictions, outcomes, groups)


@pytest.mark.parametrize("group", [None, np.nan, np.inf, "", "  ", {}, []])
def test_invalid_group_labels_are_rejected(group):
    with pytest.raises(ValueError):
        build_calibration_report([0.5], [0], [group])


def test_group_labels_are_normalized():
    report = build_calibration_report(
        [0.1, 0.1, 0.1], [0, 0, 0], [" A ", "A", 2], min_cell_count=1
    )
    assert set(report["cells"]) == {"A", "2"}
    assert report["cells"]["A"]["1"]["n"] == 2


@pytest.mark.parametrize("group", [2, "", " A "])
def test_payload_group_keys_must_be_normalized_strings(group):
    report = _report()
    report["cells"][group] = report["cells"].pop("A")
    with pytest.raises(ValueError):
        _aggregate(report)


def test_warning_uses_retained_count():
    assert _aggregate(_report(), min_n_warn=6)["groups"]["A"]["warning"]
    assert _aggregate(_report(), min_n_warn=5)["groups"]["A"]["warning"] is None


@pytest.mark.parametrize("bins", [10, 5, 2])
@pytest.mark.parametrize("minimum", [1, 5, 10])
@pytest.mark.parametrize("outcome_minimum", [0, 1, 2])
def test_matches_independent_retained_patient_reference(bins, minimum, outcome_minimum):
    rng = np.random.default_rng(12)
    sites = []
    for n in (17, 61, 151):
        predictions = rng.random(n)
        outcomes = rng.binomial(1, predictions)
        groups = rng.choice(["A", "B"], n, p=[0.8, 0.2])
        sites.append((predictions, outcomes, groups))
    reports = {
        str(index): build_calibration_report(
            *site,
            min_cell_count=minimum,
            release_n_bins=bins,
            min_outcome_count=outcome_minimum,
        )
        for index, site in enumerate(sites)
    }
    actual = aggregate_calibration_reports(json.loads(json.dumps(reports)))
    expected_groups = set()
    for group in ("A", "B"):
        retained = []
        for predictions, outcomes, groups in sites:
            for bin_index in range(bins):
                mask = (groups == group) & (predictions >= bin_index / bins)
                mask &= (predictions < (bin_index + 1) / bins) | (
                    (bin_index == bins - 1) & (predictions == 1)
                )
                count, events = int(mask.sum()), int(outcomes[mask].sum())
                if count >= minimum and min(events, count - events) >= outcome_minimum:
                    retained.extend(
                        zip(predictions[mask], outcomes[mask], [bin_index] * count)
                    )
        if not retained:
            assert group not in actual["groups"]
            continue
        expected_groups.add(group)
        count = len(retained)
        expected_ece = (
            sum(
                abs(sum(p - y for p, y, index in retained if index == bin_index))
                for bin_index in range(bins)
            )
            / count
        )
        expected_brier = sum((p - y) ** 2 for p, y, _ in retained) / count
        assert actual["groups"][group]["n"] == count
        assert actual["groups"][group]["ece"] == pytest.approx(expected_ece, abs=1e-12)
        assert actual["groups"][group]["brier"] == pytest.approx(
            expected_brier, abs=1e-12
        )
    assert set(actual["groups"]) == expected_groups
