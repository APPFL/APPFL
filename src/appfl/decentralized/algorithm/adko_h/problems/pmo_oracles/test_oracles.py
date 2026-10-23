"""Offline parity tests for `oracles.py` against PyTDC's recorded scores.

Needs only rdkit, numpy, the stdlib and `goldens.json` (made by
`make_goldens.py`): no `tdc`, no network. Run with

    python -I test_oracles.py
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # so `-I` (isolated mode) still finds our modules

from oracles import ORACLES, canonical, get_oracle, murcko_scaffold  # noqa: E402

PMO_NAMES = [
    "qed",
    "celecoxib_rediscovery",
    "troglitazone_rediscovery",
    "thiothixene_rediscovery",
    "albuterol_similarity",
    "mestranol_similarity",
    "isomers_c7h8n2o2",
    "isomers_c9h10n2o2pf2cl",
    "median1",
    "median2",
    "osimertinib_mpo",
    "fexofenadine_mpo",
    "ranolazine_mpo",
    "perindopril_mpo",
    "amlodipine_mpo",
    "sitagliptin_mpo",
    "zaleplon_mpo",
    "valsartan_smarts",
    "deco_hop",
    "scaffold_hop",
]

INVALID = ["not_a_smiles", "", "C1CC", "C(C)(C)(C)(C)C", "   ", None, 42]
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"

# rdkit parses "" into a zero-atom Mol rather than None, so PyTDC scores the
# empty string (qed 0.339, deco_hop 0.500, ...). `oracles.py` scores it 0.0 on
# purpose, so this one probe is excluded from the parity sweep and pinned down
# by TestDivergence instead.
DIVERGENT = {""}

GOLDENS = json.loads((HERE / "goldens.json").read_text())
# Keys starting with "_" are bookkeeping ("_failed", "_meta"), not oracles.
GOLDEN_ORACLES = {k: v for k, v in GOLDENS.items() if not k.startswith("_")}


class TestParity(unittest.TestCase):
    """Every recorded (oracle, SMILES) pair must reproduce PyTDC's value."""

    def test_matches_tdc(self):
        compared = 0
        for name, pairs in GOLDEN_ORACLES.items():
            oracle = get_oracle(name)
            for smiles, expected in pairs.items():
                if smiles in DIVERGENT:
                    continue
                with self.subTest(oracle=name, smiles=smiles):
                    self.assertAlmostEqual(oracle(smiles), expected, delta=1e-6)
                compared += 1
        self.assertGreater(compared, 0, "goldens.json has no oracle values")
        print(f"\ncompared {compared} (oracle, SMILES) pairs "
              f"across {len(GOLDEN_ORACLES)} oracles")

    def test_goldens_cover_every_oracle(self):
        missing = set(PMO_NAMES) - set(GOLDEN_ORACLES)
        failed = GOLDENS.get("_failed", {})
        for name in sorted(missing):
            self.assertIn(name, failed, f"{name} has neither goldens nor a _failed note")


class TestRegistry(unittest.TestCase):

    def test_all_20_names_present(self):
        self.assertEqual(sorted(ORACLES), sorted(PMO_NAMES))

    def test_valid_molecule_scores_in_unit_interval(self):
        for name in PMO_NAMES:
            with self.subTest(oracle=name):
                value = get_oracle(name)(ASPIRIN)
                self.assertIsInstance(value, float)
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_ml_oracles_rejected_with_a_reason(self):
        for name in ("drd2", "gsk3b", "jnk3"):
            with self.subTest(oracle=name):
                with self.assertRaises(KeyError) as ctx:
                    get_oracle(name)
                self.assertIn("pickle", str(ctx.exception))

    def test_unknown_name_raises(self):
        with self.assertRaises(KeyError):
            get_oracle("no_such_oracle")


class TestInvalidInput(unittest.TestCase):

    def test_invalid_smiles_score_zero_without_raising(self):
        for name in PMO_NAMES:
            oracle = get_oracle(name)
            for smiles in INVALID:
                with self.subTest(oracle=name, smiles=smiles):
                    self.assertEqual(oracle(smiles), 0.0)


class TestDivergence(unittest.TestCase):
    """Pin down the single intentional difference from PyTDC."""

    def test_empty_string_scores_zero_here_but_not_in_tdc(self):
        for name in PMO_NAMES:
            with self.subTest(oracle=name):
                self.assertEqual(get_oracle(name)(""), 0.0)
        # Guard against the premise silently changing under a future rdkit:
        # if "" ever stops parsing, upstream would agree with us and the
        # carve-out in TestParity could go.
        self.assertGreater(GOLDEN_ORACLES["deco_hop"][""], 0.0)


class TestDeterminism(unittest.TestCase):

    def test_repeated_calls_are_identical(self):
        probes = [ASPIRIN, "CCO", "Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F"]
        for name in PMO_NAMES:
            oracle = get_oracle(name)
            for smiles in probes:
                with self.subTest(oracle=name, smiles=smiles):
                    self.assertEqual(oracle(smiles), oracle(smiles))


class TestHelpers(unittest.TestCase):

    def test_canonical(self):
        self.assertEqual(canonical("OC(=O)c1ccccc1OC(C)=O"), canonical(ASPIRIN))
        self.assertIsNone(canonical("not_a_smiles"))
        self.assertIsNone(canonical(None))

    def test_murcko_scaffold(self):
        self.assertEqual(murcko_scaffold(ASPIRIN), "c1ccccc1")
        self.assertEqual(murcko_scaffold("CCO"), "")  # acyclic
        self.assertEqual(murcko_scaffold("not_a_smiles"), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
