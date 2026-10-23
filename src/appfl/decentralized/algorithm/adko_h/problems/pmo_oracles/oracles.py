# -----------------------------------------------------------------------------
# Derived from PyTDC 1.1.15 (`tdc.chem_utils.oracle.oracle`, plus the
# single-SMILES dispatch of `tdc.oracles.Oracle.__call__`).
#   PyTDC -- MIT License, Copyright (c) 2020 TDC Team
#            https://github.com/mims-harvard/TDC
# The goal-directed oracle definitions themselves (rediscovery, similarity,
# isomers, median, MPO, hop, SMARTS) originate in GuacaMol:
#   GuacaMol -- MIT License, Copyright (c) 2018 BenevolentAI
#               https://github.com/BenevolentAI/guacamol
# Implementations are copied, not paraphrased, so that scores are numerically
# identical to PyTDC's (see test_oracles.py / goldens.json).
#
# Three deliberate substitutions, each verified exact:
#   1. `scipy.stats.mstats.gmean`      -> `_gmean` (bit-identical, see below)
#   2. deprecated `AllChem.GetMorganFingerprint` / `GetAtomPairFingerprint`
#                                      -> `rdFingerprintGenerator` equivalents
#                                         (identical sparse count vectors)
#   3. PyTDC's lazy `globals()` caching -> module-level constants
# -----------------------------------------------------------------------------
"""The 20 pure-rdkit PMO (Practical Molecular Optimization) oracles.

PMO -- Gao et al., "Sample Efficiency Matters: A Benchmark for Practical
Molecular Optimization", NeurIPS 2022 Datasets & Benchmarks -- is a 23-task
goal-directed molecular design benchmark. This module vendors the 20 tasks that
need nothing but rdkit, so a benchmark run has no network access, no PyTDC
import chain and no heavy dependency tree (the motivation: compute nodes with
no internet).

Use it as::

    from oracles import get_oracle, ORACLES
    score = get_oracle("qed")("CC(=O)Oc1ccccc1C(=O)O")   # float in [0, 1]

Every oracle maps one SMILES string to a float in [0, 1], higher is better, and
returns 0.0 for input rdkit cannot parse (PyTDC's `default_property`) or that
parses to an empty molecule (see `_guard` -- the one deliberate divergence from
PyTDC). Oracles are deterministic and stateless; unlike PyTDC's `Oracle` they
keep no call counter, so budget accounting belongs to the caller.

Excluded from PMO's 23 tasks: `drd2`, `gsk3b` and `jnk3`. Those three are
scikit-learn models whose weights PyTDC downloads as pickles at first call, so
they need both network access and a pinned sklearn to unpickle -- exactly the
two liabilities this module exists to avoid. PyTDC's other network-dependent
oracles (SA/`fpscores`, docking, retrosynthesis) are out of PMO and out of here.
"""

from __future__ import annotations

import re
from functools import partial

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem import QED
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

rdBase.DisableLog("rdApp.error")

__all__ = ["ORACLES", "get_oracle", "canonical", "murcko_scaffold"]


# --- helpers ----------------------------------------------------------------


def _gmean(values):
    """Geometric mean, replacing `scipy.stats.mstats.gmean`.

    scipy computes `exp(average(log(a)))`; this reproduces it bit-for-bit
    (checked over 2e4 random vectors) including the zero case, where
    `log(0) = -inf` makes the result 0.0.
    """
    with np.errstate(divide="ignore"):
        return float(np.exp(np.mean(np.log(np.asarray(values, dtype=float)))))


def canonical(smiles: str, include_stereocenters: bool = True):
    """rdkit canonical SMILES, or None if unparseable."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)


def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold SMILES; "" for acyclic or unparseable input."""
    if not isinstance(smiles, str):
        return ""
    try:
        return MurckoScaffoldSmiles(smiles)
    except Exception:
        return ""


def _mol(smiles: str):
    """PyTDC's `smiles_to_rdkit_mol`: parse plus an explicit sanitize check."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
    return mol


# Sparse count fingerprints, matching PyTDC's `fp2fpfunc` exactly.
_ECFP4_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2)
_ECFP6_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=3)
_FCFP4_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
)
_AP_GEN = rdFingerprintGenerator.GetAtomPairGenerator(maxDistance=10)

_FP_GENS = {
    "ECFP4": _ECFP4_GEN,
    "ECFP6": _ECFP6_GEN,
    "FCFP4": _FCFP4_GEN,
    "AP": _AP_GEN,
}


def _fp(smiles: str, kind: str):
    return _FP_GENS[kind].GetSparseCountFingerprint(_mol(smiles))


def _tanimoto(fp_a, fp_b) -> float:
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


# --- GuacaMol score modifiers -----------------------------------------------


class GaussianModifier:
    """Gaussian bell centred on `mu`."""

    def __init__(self, mu: float, sigma: float) -> None:
        self.mu, self.sigma = mu, sigma

    def __call__(self, x):
        return np.exp(-0.5 * np.power((x - self.mu) / self.sigma, 2.0))


class MinMaxGaussianModifier:
    """Half Gaussian: flat at 1.0 on the satisfied side of `mu`."""

    def __init__(self, mu: float, sigma: float, minimize: bool = False) -> None:
        self.mu, self.sigma, self.minimize = mu, sigma, minimize
        self._full_gaussian = GaussianModifier(mu=mu, sigma=sigma)

    def __call__(self, x):
        mod_x = np.maximum(x, self.mu) if self.minimize else np.minimum(x, self.mu)
        return self._full_gaussian(mod_x)


MinGaussianModifier = partial(MinMaxGaussianModifier, minimize=True)
MaxGaussianModifier = partial(MinMaxGaussianModifier, minimize=False)


class ClippedScoreModifier:
    """Linear ramp from `lower_x` to `upper_x`, clipped to [low, high]."""

    def __init__(
        self,
        upper_x: float,
        lower_x: float = 0.0,
        high_score: float = 1.0,
        low_score: float = 0.0,
    ) -> None:
        assert low_score < high_score
        self.low_score, self.high_score = low_score, high_score
        self.slope = (high_score - low_score) / (upper_x - lower_x)
        self.intercept = high_score - self.slope * upper_x

    def __call__(self, x):
        return np.clip(self.slope * x + self.intercept, self.low_score, self.high_score)


# --- molecular formula / isomer scoring -------------------------------------


class AtomCounter:
    """Count atoms of one element (adding explicit Hs when element == "H")."""

    def __init__(self, element: str) -> None:
        self.element = element

    def __call__(self, mol) -> int:
        if self.element == "H":
            mol = Chem.AddHs(mol)
        return sum(1 for a in mol.GetAtoms() if a.GetSymbol() == self.element)


def parse_molecular_formula(formula: str):
    """"C8H3F3Br" -> [("C", 8), ("H", 3), ("F", 3), ("Br", 1)]."""
    results = []
    for element, count in re.findall(r"([A-Z][a-z]*)(\d*)", formula):
        results.append((element, 1 if not count else int(count)))
    return results


class IsomerScoring:
    """Reward matching a target molecular formula (GuacaMol "isomers" tasks).

    Scores a Gaussian per element count plus one on the total atom count, then
    takes the geometric mean. Hydrogens are made explicit first, so the formula
    compared is rdkit's, not the raw SMILES text.
    """

    def __init__(self, target_formula: str, means: str = "geometric") -> None:
        assert means in ("geometric", "arithmetic")
        self.mean_func = _gmean if means == "geometric" else (lambda v: float(np.mean(v)))
        atom2cnt = parse_molecular_formula(target_formula)
        total = sum(cnt for _, cnt in atom2cnt)
        self.total_atom_modifier = GaussianModifier(mu=total, sigma=2.0)
        self.counter_modifiers = [
            (AtomCounter(atom), GaussianModifier(mu=cnt, sigma=1.0))
            for atom, cnt in atom2cnt
        ]

    def __call__(self, test_smiles: str) -> float:
        explicit_h = Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(canonical(test_smiles))))
        molecule = _mol(explicit_h)
        scores = [mod(counter(molecule)) for counter, mod in self.counter_modifiers]
        formula = CalcMolFormula(Chem.MolFromSmiles(explicit_h))
        total = sum(cnt for _, cnt in parse_molecular_formula(formula))
        scores.append(self.total_atom_modifier(total))
        return self.mean_func(scores)


# --- similarity-family oracles ----------------------------------------------


class _Rediscovery:
    """Plain Tanimoto to a target molecule."""

    def __init__(self, target_smiles: str, fp: str = "ECFP4") -> None:
        self.fp = fp
        self.target_fp = _fp(target_smiles, fp)

    def __call__(self, test_smiles: str) -> float:
        return _tanimoto(self.target_fp, _fp(test_smiles, self.fp))


class _Similarity:
    """Tanimoto to a target molecule, passed through a score modifier."""

    def __init__(self, target_smiles: str, fp: str, modifier_func=None) -> None:
        self.fp = fp
        self.target_fp = _fp(target_smiles, fp)
        self.modifier_func = modifier_func

    def __call__(self, test_smiles: str) -> float:
        value = _tanimoto(self.target_fp, _fp(test_smiles, self.fp))
        return value if self.modifier_func is None else self.modifier_func(value)


class _Median:
    """Geometric mean of the similarities to two targets at once."""

    def __init__(self, target_smiles_1: str, target_smiles_2: str, fp: str) -> None:
        self.fp = fp
        self.target_fp1 = _fp(target_smiles_1, fp)
        self.target_fp2 = _fp(target_smiles_2, fp)

    def __call__(self, test_smiles: str) -> float:
        test_fp = _fp(test_smiles, self.fp)
        return _gmean(
            [_tanimoto(self.target_fp1, test_fp), _tanimoto(self.target_fp2, test_fp)]
        )


_SIMILARITY_MODIFIER = ClippedScoreModifier(upper_x=0.75)

CELECOXIB = "CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F"
TROGLITAZONE = "Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O"
THIOTHIXENE = "CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1"
ALBUTEROL = "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1"
MESTRANOL = "COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1"
CAMPHOR = "CC1(C)C2CCC1(C)C(=O)C2"
MENTHOL = "CC(C)C1CCC(C)CC1O"
TADALAFIL = "O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C"
SILDENAFIL = "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C"
OSIMERTINIB = "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34"
FEXOFENADINE = "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4"
RANOLAZINE = "COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2"
PERINDOPRIL = "O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC"
AMLODIPINE = "Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\\C(=O)OC"
SITAGLIPTIN = "Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F"
ZALEPLON = "O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1"
PHARMACOPHORE = "CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C"

_celecoxib_rediscovery = _Rediscovery(CELECOXIB, fp="ECFP4")
_troglitazone_rediscovery = _Rediscovery(TROGLITAZONE, fp="ECFP4")
_thiothixene_rediscovery = _Rediscovery(THIOTHIXENE, fp="ECFP4")
_albuterol_similarity = _Similarity(ALBUTEROL, "FCFP4", _SIMILARITY_MODIFIER)
_mestranol_similarity = _Similarity(MESTRANOL, "AP", _SIMILARITY_MODIFIER)
_median1 = _Median(CAMPHOR, MENTHOL, fp="ECFP4")
_median2 = _Median(TADALAFIL, SILDENAFIL, fp="ECFP6")
_isomers_c7h8n2o2 = IsomerScoring("C7H8N2O2")
_isomers_c9h10n2o2pf2cl = IsomerScoring("C9H10N2O2PF2Cl")


# --- multi-property-objective (MPO) oracles ---------------------------------

_OSIMERTINIB_FCFP4 = _fp(OSIMERTINIB, "FCFP4")
_OSIMERTINIB_ECFP6 = _fp(OSIMERTINIB, "ECFP6")
_FEXOFENADINE_AP = _fp(FEXOFENADINE, "AP")
_RANOLAZINE_AP = _fp(RANOLAZINE, "AP")
_PERINDOPRIL_ECFP4 = _fp(PERINDOPRIL, "ECFP4")
_AMLODIPINE_ECFP4 = _fp(AMLODIPINE, "ECFP4")
_SITAGLIPTIN_ECFP4 = _fp(SITAGLIPTIN, "ECFP4")
_ZALEPLON_ECFP4 = _fp(ZALEPLON, "ECFP4")
_FLUORINE_COUNTER = AtomCounter("F")


def osimertinib_mpo(test_smiles: str) -> float:
    """Stay close to osimertinib (FCFP4) while moving away in ECFP6, TPSA, logP."""
    sim_v1_modifier = ClippedScoreModifier(upper_x=0.8)
    sim_v2_modifier = MinGaussianModifier(mu=0.85, sigma=0.1)
    tpsa_modifier = MaxGaussianModifier(mu=100, sigma=10)
    logp_modifier = MinGaussianModifier(mu=1, sigma=1)

    molecule = _mol(test_smiles)
    tpsa_score = tpsa_modifier(Descriptors.TPSA(molecule))
    logp_score = logp_modifier(Descriptors.MolLogP(molecule))
    similarity_v1 = sim_v1_modifier(
        _tanimoto(_OSIMERTINIB_FCFP4, _fp(test_smiles, "FCFP4"))
    )
    similarity_v2 = sim_v2_modifier(
        _tanimoto(_OSIMERTINIB_ECFP6, _fp(test_smiles, "ECFP6"))
    )
    return _gmean([tpsa_score, logp_score, similarity_v1, similarity_v2])


def fexofenadine_mpo(test_smiles: str) -> float:
    """Fexofenadine-like (atom pairs) with raised TPSA and lowered logP."""
    similar_modifier = ClippedScoreModifier(upper_x=0.8)
    tpsa_modifier = MaxGaussianModifier(mu=90, sigma=10)
    logp_modifier = MinGaussianModifier(mu=4, sigma=1)

    molecule = _mol(test_smiles)
    tpsa_score = tpsa_modifier(Descriptors.TPSA(molecule))
    logp_score = logp_modifier(Descriptors.MolLogP(molecule))
    similarity_value = similar_modifier(
        _tanimoto(_fp(test_smiles, "AP"), _FEXOFENADINE_AP)
    )
    return _gmean([tpsa_score, logp_score, similarity_value])


def ranolazine_mpo(test_smiles: str) -> float:
    """Ranolazine-like with raised TPSA and logP and one fluorine."""
    similar_modifier = ClippedScoreModifier(upper_x=0.7)
    tpsa_modifier = MaxGaussianModifier(mu=95, sigma=20)
    logp_modifier = MaxGaussianModifier(mu=7, sigma=1)
    fluorine_modifier = GaussianModifier(mu=1, sigma=1.0)

    molecule = _mol(test_smiles)
    tpsa_score = tpsa_modifier(Descriptors.TPSA(molecule))
    logp_score = logp_modifier(Descriptors.MolLogP(molecule))
    similarity_value = similar_modifier(_tanimoto(_fp(test_smiles, "AP"), _RANOLAZINE_AP))
    fluorine_value = fluorine_modifier(_FLUORINE_COUNTER(molecule))
    return _gmean([tpsa_score, logp_score, similarity_value, fluorine_value])


def perindopril_mpo(test_smiles: str) -> float:
    """Perindopril-like with exactly two aromatic rings."""
    arom_rings_modifier = GaussianModifier(mu=2, sigma=0.5)

    molecule = _mol(test_smiles)
    similarity_value = _tanimoto(_fp(test_smiles, "ECFP4"), _PERINDOPRIL_ECFP4)
    arom_rings_value = arom_rings_modifier(rdMolDescriptors.CalcNumAromaticRings(molecule))
    return _gmean([similarity_value, arom_rings_value])


def amlodipine_mpo(test_smiles: str) -> float:
    """Amlodipine-like with exactly three rings."""
    num_rings_modifier = GaussianModifier(mu=3, sigma=0.5)

    molecule = _mol(test_smiles)
    similarity_value = _tanimoto(_fp(test_smiles, "ECFP4"), _AMLODIPINE_ECFP4)
    num_rings_value = num_rings_modifier(rdMolDescriptors.CalcNumRings(molecule))
    return _gmean([similarity_value, num_rings_value])


_ISOMER_C16H15F6N5O = IsomerScoring("C16H15F6N5O")
_ISOMER_C19H17N3O2 = IsomerScoring("C19H17N3O2")
_SITAGLIPTIN_MOL = Chem.MolFromSmiles(SITAGLIPTIN)
_SITAGLIPTIN_LOGP_MODIFIER = GaussianModifier(
    mu=Descriptors.MolLogP(_SITAGLIPTIN_MOL), sigma=0.2
)
_SITAGLIPTIN_TPSA_MODIFIER = GaussianModifier(
    mu=Descriptors.TPSA(_SITAGLIPTIN_MOL), sigma=5
)
_SITAGLIPTIN_SIMILAR_MODIFIER = GaussianModifier(mu=0, sigma=0.1)


def sitagliptin_mpo(test_smiles: str) -> float:
    """A sitagliptin isomer with sitagliptin's logP and TPSA but unlike it."""
    molecule = Chem.MolFromSmiles(test_smiles)
    logp_score = _SITAGLIPTIN_LOGP_MODIFIER(Descriptors.MolLogP(molecule))
    tpsa_score = _SITAGLIPTIN_TPSA_MODIFIER(Descriptors.TPSA(molecule))
    isomer_score = _ISOMER_C16H15F6N5O(test_smiles)
    similarity_value = _SITAGLIPTIN_SIMILAR_MODIFIER(
        _tanimoto(_fp(test_smiles, "ECFP4"), _SITAGLIPTIN_ECFP4)
    )
    return _gmean([similarity_value, logp_score, tpsa_score, isomer_score])


def zaleplon_mpo(test_smiles: str) -> float:
    """Zaleplon-like and a C19H17N3O2 isomer."""
    similarity_value = _tanimoto(_fp(test_smiles, "ECFP4"), _ZALEPLON_ECFP4)
    isomer_value = _ISOMER_C19H17N3O2(test_smiles)
    return _gmean([similarity_value, isomer_value])


# --- SMARTS / pharmacophore oracles -----------------------------------------


class SMARTSScoring:
    """1.0 when the SMARTS pattern matches (or, if `inverse`, when it does not)."""

    def __init__(self, target_smarts: str, inverse: bool) -> None:
        self.target_mol = Chem.MolFromSmarts(target_smarts)
        self.inverse = inverse

    def __call__(self, mol) -> float:
        matched = len(mol.GetSubstructMatches(self.target_mol)) > 0
        return float(matched != self.inverse)


def _phco_fp(mol):
    """Gobbi 2D pharmacophore fingerprint."""
    return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)


_PHARMACOPHORE_FP = _phco_fp(_mol(PHARMACOPHORE))
_SCAFFOLD_SMARTS = "[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12"
_DECO1 = SMARTSScoring("CS([#6])(=O)=O", inverse=True)
_DECO2 = SMARTSScoring("[#7]-c1ccc2ncsc2c1", inverse=True)
_DECO_KEEP = SMARTSScoring(
    "[#6]-[#6]-[#6]-[#8]-[#6]~[#6]~[#6]~[#6]~[#6]-[#7]-c1ccc2ncsc2c1", inverse=False
)
_SCAFFOLD_KEEP = SMARTSScoring(_SCAFFOLD_SMARTS, inverse=False)
_SCAFFOLD_DROP = SMARTSScoring(_SCAFFOLD_SMARTS, inverse=True)


def deco_hop(test_smiles: str) -> float:
    """Keep the scaffold, replace both decorations."""
    molecule = _mol(test_smiles)
    similarity_modifier = ClippedScoreModifier(upper_x=0.85)
    similarity_value = similarity_modifier(_tanimoto(_phco_fp(molecule), _PHARMACOPHORE_FP))
    return float(
        np.mean(
            [
                similarity_value,
                _DECO1(molecule),
                _DECO2(molecule),
                _SCAFFOLD_KEEP(molecule),
            ]
        )
    )


def scaffold_hop(test_smiles: str) -> float:
    """Keep the decorations, replace the scaffold."""
    molecule = _mol(test_smiles)
    similarity_modifier = ClippedScoreModifier(upper_x=0.75)
    similarity_value = similarity_modifier(_tanimoto(_phco_fp(molecule), _PHARMACOPHORE_FP))
    return float(
        np.mean([similarity_value, _DECO_KEEP(molecule), _SCAFFOLD_DROP(molecule)])
    )


_VALSARTAN_SMARTS_MOL = Chem.MolFromSmarts("CN(C=O)Cc1ccc(c2ccccc2)cc1")
# The property targets are sitagliptin's, written here in its other SMILES form
# exactly as PyTDC/GuacaMol have it.
_VALSARTAN_TARGET_MOL = Chem.MolFromSmiles(
    "NC(CC(=O)N1CCn2c(nnc2C(F)(F)F)C1)Cc1cc(F)c(F)cc1F"
)
_VALSARTAN_LOGP_MODIFIER = GaussianModifier(
    mu=Descriptors.MolLogP(_VALSARTAN_TARGET_MOL), sigma=0.2
)
_VALSARTAN_TPSA_MODIFIER = GaussianModifier(
    mu=Descriptors.TPSA(_VALSARTAN_TARGET_MOL), sigma=5
)
_VALSARTAN_BERTZ_MODIFIER = GaussianModifier(
    mu=Descriptors.BertzCT(_VALSARTAN_TARGET_MOL), sigma=30
)


def valsartan_smarts(test_smiles: str) -> float:
    """Contain the valsartan SMARTS and hit sitagliptin's logP, TPSA, BertzCT."""
    molecule = _mol(test_smiles)
    smarts_score = float(len(molecule.GetSubstructMatches(_VALSARTAN_SMARTS_MOL)) > 0)
    logp_score = _VALSARTAN_LOGP_MODIFIER(Descriptors.MolLogP(molecule))
    tpsa_score = _VALSARTAN_TPSA_MODIFIER(Descriptors.TPSA(molecule))
    bertz_score = _VALSARTAN_BERTZ_MODIFIER(Descriptors.BertzCT(molecule))
    return _gmean([smarts_score, tpsa_score, logp_score, bertz_score])


def qed(smiles: str) -> float:
    """rdkit's quantitative estimate of drug-likeness."""
    return QED.qed(Chem.MolFromSmiles(smiles))


# --- public registry --------------------------------------------------------


def _guard(fn):
    """Return 0.0 instead of raising, matching PyTDC's `default_property`.

    PyTDC's `Oracle.__call__` short-circuits to 0.0 when rdkit cannot parse the
    SMILES; we do the same, plus two safeguards it lacks:

    * an exception from a parseable but unscoreable molecule scores 0.0, so a
      sweep over millions of generated strings cannot die on one of them;
    * a zero-atom molecule scores 0.0. This is the one deliberate numerical
      divergence from PyTDC: rdkit parses "" into an empty Mol rather than
      None, so upstream scores the empty string -- qed 0.339, deco_hop 0.500,
      scaffold_hop 0.333 -- which a generator could farm for free reward.
    """

    def scored(smiles: str) -> float:
        if not isinstance(smiles, str):
            return 0.0
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return 0.0
        try:
            return float(fn(smiles))
        except Exception:
            return 0.0

    scored.__name__ = getattr(fn, "__name__", "oracle")
    scored.__doc__ = fn.__doc__
    return scored


ORACLES = {
    name: _guard(fn)
    for name, fn in [
        ("qed", qed),
        ("celecoxib_rediscovery", _celecoxib_rediscovery),
        ("troglitazone_rediscovery", _troglitazone_rediscovery),
        ("thiothixene_rediscovery", _thiothixene_rediscovery),
        ("albuterol_similarity", _albuterol_similarity),
        ("mestranol_similarity", _mestranol_similarity),
        ("isomers_c7h8n2o2", _isomers_c7h8n2o2),
        ("isomers_c9h10n2o2pf2cl", _isomers_c9h10n2o2pf2cl),
        ("median1", _median1),
        ("median2", _median2),
        ("osimertinib_mpo", osimertinib_mpo),
        ("fexofenadine_mpo", fexofenadine_mpo),
        ("ranolazine_mpo", ranolazine_mpo),
        ("perindopril_mpo", perindopril_mpo),
        ("amlodipine_mpo", amlodipine_mpo),
        ("sitagliptin_mpo", sitagliptin_mpo),
        ("zaleplon_mpo", zaleplon_mpo),
        ("valsartan_smarts", valsartan_smarts),
        ("deco_hop", deco_hop),
        ("scaffold_hop", scaffold_hop),
    ]
}

EXCLUDED_ML_ORACLES = ("drd2", "gsk3b", "jnk3")


def get_oracle(name: str):
    """Look up one of the 20 PMO oracles by its PMO name (lowercase)."""
    key = name.strip().lower()
    if key not in ORACLES:
        if key in EXCLUDED_ML_ORACLES:
            raise KeyError(
                f"{name!r} is a scikit-learn PMO oracle; it needs a downloaded "
                "pickle and is deliberately excluded (see module docstring)."
            )
        raise KeyError(f"unknown oracle {name!r}; available: {sorted(ORACLES)}")
    return ORACLES[key]
