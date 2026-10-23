"""Record reference scores from the installed PyTDC, for parity tests.

Run this once in an environment that has `tdc` (and internet, in case an oracle
wants a download). It writes `goldens.json` next to this file:

    {oracle_name: {smiles: value}, "_failed": {...}, "_meta": {...}}

`test_oracles.py` then checks `oracles.py` against that file with no tdc and no
network. PyTDC 1.1.15 does not import under rdkit >= 2022.03 because
`rdkit.six` was removed, so this script injects a two-line `rdkit.six` shim
first; that is the fragility this vendoring exists to escape.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # so `-I` (isolated mode) still finds our modules


def _shim_rdkit_six() -> None:
    """PyTDC 1.1.15 imports `rdkit.six.iteritems`, gone since rdkit 2022.03."""
    import rdkit

    if "rdkit.six" in sys.modules:
        return
    module = types.ModuleType("rdkit.six")
    module.iteritems = lambda d: iter(d.items())
    sys.modules["rdkit.six"] = module
    rdkit.six = module


# A deliberately mixed probe set: approved drugs, every task's own target
# molecule, tiny molecules, charges, stereochemistry, halogens/phosphorus,
# formula matches for the two isomer tasks, and four unparseable strings.
SMILES = [
    # --- task target molecules -------------------------------------------
    "CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F",  # celecoxib
    "Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O",  # troglitazone
    "CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1",  # thiothixene
    "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",  # albuterol
    "COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1",  # mestranol
    "CC1(C)C2CCC1(C)C(=O)C2",  # camphor
    "CC(C)C1CCC(C)CC1O",  # menthol
    "O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C",  # tadalafil
    "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34",  # osimertinib
    "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4",  # fexofenadine
    "COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2",  # ranolazine
    "O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC",  # perindopril
    "Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F",  # sitagliptin
    "O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1",  # zaleplon
    "CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C",  # hop pharmacophore
    # --- other approved drugs --------------------------------------------
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C",  # sildenafil
    "CCCCC(=O)N(Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1)C(C(C)C)C(=O)O",  # valsartan
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "CC(=O)Nc1ccc(O)cc1",  # paracetamol
    "CN1CCC[C@H]1c1cccnc1",  # nicotine
    "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O",  # penicillin G
    "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",  # imatinib
    "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",  # morphine
    "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",  # cholesterol
    # --- isomer-task formula matches -------------------------------------
    "Cc1ccc(N)c([N+](=O)[O-])c1",  # C7H8N2O2
    "CCOC(=O)c1ccc(N)cc1N",  # C9H12N2O2, near-miss for the second isomer task
    "CCP(F)(F)C(=O)Nc1ccc(Cl)cc1[N+](=O)[O-]",  # C9H10ClF2N2O3P, P/F/Cl rich
    # --- small, charged, odd ---------------------------------------------
    "CCO",
    "C",
    "O",
    "c1ccccc1",
    "C1CCCCCCCCC1",
    "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",  # glucose
    "C[C@H]([NH3+])C(=O)[O-]",  # alanine zwitterion
    "CC(=O)[O-]",
    "CP(=O)(OC)OC",  # dimethyl methylphosphonate
    "FC(F)(F)c1ccc(Cl)c(Br)c1I",  # halogen soup
    "O=P(O)(O)OCC(F)(Cl)Br",
    "c1csc(n1)N",  # 2-aminothiazole
    # --- unparseable ------------------------------------------------------
    "not_a_smiles",
    "",
    "C1CC",  # unclosed ring
    "C(C)(C)(C)(C)C",  # pentavalent carbon
]

ORACLE_NAMES = [
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


def main() -> int:
    _shim_rdkit_six()
    import rdkit
    from tdc import Oracle

    goldens: dict = {}
    failed: dict = {}

    for name in ORACLE_NAMES:
        try:
            oracle = Oracle(name=name)
        except Exception as exc:  # download failure, missing dep, ...
            failed[name] = [{"smiles": None, "error": f"{type(exc).__name__}: {exc}"}]
            print(f"  FAILED to build {name}: {type(exc).__name__}: {exc}")
            continue

        values = {}
        errors = []
        for smiles in SMILES:
            try:
                values[smiles] = float(oracle(smiles))
            except Exception as exc:
                errors.append(
                    {"smiles": smiles, "error": f"{type(exc).__name__}: {exc}"}
                )
        if values:
            goldens[name] = values
        if errors:
            failed[name] = errors
        print(f"  {name}: {len(values)} values, {len(errors)} errors")

    goldens["_failed"] = failed
    goldens["_meta"] = {
        "source": "PyTDC tdc.Oracle",
        "rdkit": rdkit.__version__,
        "python": sys.version.split()[0],
        "n_smiles": len(SMILES),
    }

    out = HERE / "goldens.json"
    out.write_text(json.dumps(goldens, indent=1, sort_keys=True) + "\n")
    pairs = sum(len(v) for k, v in goldens.items() if not k.startswith("_"))
    print(f"wrote {out} -- {len(ORACLE_NAMES) - len(failed)} oracles, {pairs} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
