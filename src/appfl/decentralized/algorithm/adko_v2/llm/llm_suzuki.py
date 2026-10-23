"""The part of the reference implementation's Suzuki module that the L3 planner imports.

Copied from the reference implementation: the Suzuki option names and metadata (``DIM_NAMES``, ``REAGENT_METADATA``) that the
prompts render, and the tolerant JSON parser that reads the planner's answer. The rest of that
module is ADKO v1's LLM pipeline, which v2 does not use.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional


# ── Suzuki-specific labels & friendly names ──────────────────────────────────
DIM_NAMES = ("electrophile", "nucleophile", "base", "ligand", "solvent")

# Human-readable reagent metadata for the exact categorical options in
# run_suzuki.py. The names/classes are curated chemistry labels; dft_summary is
# a compact qualitative rendering of the Olympus descriptors_dft.csv fields
# rather than raw numeric rows, which are too noisy for prompt context.
REAGENT_METADATA = {
    "electrophile": {
        0: {
            "name": "heteroaryl chloride",
            "class": "least reactive aryl halide electrophile",
            "reaction_notes": "harder oxidative addition; often benefits from electron-rich bulky phosphines",
            "dft_summary": "compact electrophile; low dipole; small halide leaving group",
        },
        1: {
            "name": "heteroaryl bromide",
            "class": "moderately reactive aryl halide electrophile",
            "reaction_notes": "usually easier oxidative addition than chloride with similar substrate electronics",
            "dft_summary": "larger and slightly softer than chloride; similar frontier orbital energies",
        },
        2: {
            "name": "heteroaryl triflate",
            "class": "pseudohalide electrophile",
            "reaction_notes": "activated leaving group but bulky/polar triflate can change base and solvent sensitivity",
            "dft_summary": "largest, most polar electrophile; highest dipole and molar volume in this set",
        },
        3: {
            "name": "heteroaryl iodide",
            "class": "most reactive aryl halide electrophile",
            "reaction_notes": "fast oxidative addition; good exploitation target unless side reactions dominate",
            "dft_summary": "large soft halide; lowest hardness among electrophiles",
        },
    },
    "nucleophile": {
        0: {
            "name": "aryl boronic acid",
            "class": "free boronic acid nucleophile",
            "reaction_notes": "often reactive in transmetalation but can be sensitive to protodeboronation",
            "dft_summary": "smallest nucleophile; neutral; moderate dipole",
        },
        1: {
            "name": "aryl Bpin boronate ester",
            "class": "protected boronate nucleophile",
            "reaction_notes": "bulkier and more stable than boronic acid; may need base-assisted activation",
            "dft_summary": "largest nucleophile; neutral; slightly higher dipole than boronic acid",
        },
        2: {
            "name": "aryl trifluoroborate",
            "class": "anionic organotrifluoroborate nucleophile",
            "reaction_notes": "stable boron reagent; highly polar and often requires activation/hydrolysis before transmetalation",
            "dft_summary": "anionic and very polar; highest dipole and highest HOMO in nucleophile set",
        },
    },
    "base": {
        0: {
            "name": "sodium hydroxide",
            "class": "strong inorganic hydroxide base",
            "reaction_notes": "strongly basic, small, polar; can promote hydrolysis/protodeboronation",
            "dft_summary": "smallest base; high dipole; low steric bulk",
        },
        1: {
            "name": "sodium bicarbonate",
            "class": "mild inorganic carbonate base",
            "reaction_notes": "buffering, weaker base; often gentler but may underactivate difficult partners",
            "dft_summary": "moderate size and high dipole; comparatively low HOMO",
        },
        2: {
            "name": "cesium fluoride",
            "class": "fluoride base/activator",
            "reaction_notes": "can activate boron reagents, especially boronates/trifluoroborates; highly ionic",
            "dft_summary": "high dipole; small fluoride-centered steric profile",
        },
        3: {
            "name": "potassium phosphate",
            "class": "strong inorganic phosphate base",
            "reaction_notes": "common Suzuki base; robust for boron activation and less nucleophilic than hydroxide",
            "dft_summary": "largest inorganic base; high dipole; high buried-volume descriptor",
        },
        4: {
            "name": "potassium hydroxide",
            "class": "strong inorganic hydroxide base",
            "reaction_notes": "strongly basic; can accelerate coupling but may degrade sensitive boron reagents",
            "dft_summary": "small polar base; lower hardness than sodium hydroxide",
        },
        5: {
            "name": "lithium isopropoxide",
            "class": "alkoxide base",
            "reaction_notes": "strong organic alkoxide; more sterically hindered and less aqueous than hydroxide bases",
            "dft_summary": "larger organic base; moderate dipole; high steric descriptor",
        },
        6: {
            "name": "triethylamine",
            "class": "tertiary amine base",
            "reaction_notes": "organic, nonionic, comparatively weak base; useful when inorganic salts are problematic",
            "dft_summary": "lowest dipole base; high hardness; bulky organic base",
        },
    },
    "ligand": {
        0: {
            "name": "tri-tert-butylphosphine",
            "class": "very bulky electron-rich monophosphine",
            "reaction_notes": "strong donor; often helps difficult aryl chlorides and fast oxidative addition",
            "dft_summary": "compact but highly buried P environment; high steric descriptor",
        },
        1: {
            "name": "triphenylphosphine",
            "class": "less bulky neutral aryl monophosphine",
            "reaction_notes": "general ligand, but often weaker for challenging heteroaryl chlorides",
            "dft_summary": "moderate sterics; more electron-poor than alkyl phosphines",
        },
        2: {
            "name": "di-tert-butyl dimethylaminophenyl phosphine",
            "class": "electron-rich bulky aryl dialkylphosphine",
            "reaction_notes": "electron-rich and bulky; plausible for aryl chloride activation",
            "dft_summary": "highest ligand dipole; electron-rich HOMO; high steric descriptor",
        },
        3: {
            "name": "tricyclohexylphosphine",
            "class": "bulky electron-rich alkyl monophosphine",
            "reaction_notes": "strong donor; often supports oxidative addition and reductive elimination",
            "dft_summary": "large hydrophobic ligand; high buried-volume descriptor",
        },
        4: {
            "name": "tri-o-tolylphosphine",
            "class": "sterically hindered aryl monophosphine",
            "reaction_notes": "bulkier than triphenylphosphine but less electron-rich than alkyl phosphines",
            "dft_summary": "large aryl ligand; lower steric descriptor near one face",
        },
        5: {
            "name": "adamantyl butyl phosphine",
            "class": "very bulky alkyl phosphine",
            "reaction_notes": "large hydrophobic donor ligand; can favor oxidative addition to less reactive electrophiles",
            "dft_summary": "very large molar volume; high steric descriptor; electron-rich",
        },
        6: {
            "name": "SPhos-like biaryl phosphine",
            "class": "bulky electron-rich biaryl monophosphine",
            "reaction_notes": "privileged Suzuki ligand family; often strong for heteroaryl substrates",
            "dft_summary": "large molar volume; high steric descriptor; aryl-alkyl donor balance",
        },
        7: {
            "name": "Josiphos-like ferrocenyl bisphosphine",
            "class": "chelating ferrocenyl bisphosphine",
            "reaction_notes": "bidentate ferrocenyl ligand; can change catalyst geometry and selectivity",
            "dft_summary": "very large ligand; metal-containing; high steric descriptor",
        },
        8: {
            "name": "XPhos-like bulky biaryl phosphine",
            "class": "bulky biaryl dicyclohexylphosphine",
            "reaction_notes": "privileged ligand for Suzuki couplings; useful for aryl chlorides and heteroaryl partners",
            "dft_summary": "one of the largest ligands; highest steric descriptor in set",
        },
        9: {
            "name": "dppf",
            "class": "chelating ferrocenyl bisphosphine",
            "reaction_notes": "bidentate ligand; robust but may be less open than bulky monophosphines",
            "dft_summary": "very large metal-containing ligand; low hardness; high bidentate sterics",
        },
        10: {
            "name": "Xantphos-like wide-bite bisphosphine",
            "class": "wide-bite chelating bisphosphine",
            "reaction_notes": "large bidentate ligand; can favor reductive elimination but may be sterically restrictive",
            "dft_summary": "largest ligand by molar volume; bulky wide-bite scaffold",
        },
    },
    "solvent": {
        0: {
            "name": "MeCN (acetonitrile)",
            "class": "polar aprotic nitrile solvent",
            "reaction_notes": "highly polar and coordinating; can solubilize salts but may bind metal centers",
            "dft_summary": "high dipole; small aprotic solvent",
        },
        1: {
            "name": "THF (tetrahydrofuran)",
            "class": "ethereal aprotic solvent",
            "reaction_notes": "moderately polar coordinating ether; common for organoboron chemistry",
            "dft_summary": "lower dipole than MeCN/DMF; medium molar volume",
        },
        2: {
            "name": "DMF (dimethylformamide)",
            "class": "highly polar aprotic amide solvent",
            "reaction_notes": "strongly polar and coordinating; often improves salt solubility",
            "dft_summary": "highest solvent dipole in set; polar aprotic",
        },
        3: {
            "name": "MeOH (methanol)",
            "class": "polar protic alcohol solvent",
            "reaction_notes": "protic solvent; can assist boron activation but may alter base strength and side reactions",
            "dft_summary": "smallest solvent; protic; moderate dipole",
        },
    },
}

# Attach exact SMILES after the readable metadata so the LLM receives both
# semantic labels and the canonical dataset identifiers.
_SMILES_BY_DIM_IDX = {
    "electrophile": {
        0: "ClC1=CC=C(N=CC=C2)C2=C1",
        1: "BrC1=CC=C(N=CC=C2)C2=C1",
        2: "O=S(OC1=CC=C(N=CC=C2)C2=C1)(C(F)(F)F)=O",
        3: "IC1=CC=C(N=CC=C2)C2=C1",
    },
    "nucleophile": {
        0: "CC1=CC=C(N(C2CCCCO2)N=C3)C3=C1B(O)O",
        1: "CC1=CC=C(N(C2CCCCO2)N=C3)C3=C1B4OC(C)(C)C(C)(C)O4",
        2: "CC1=CC=C(N(C2CCCCO2)N=C3)C3=C1[B-](F)(F)F",
    },
    "base": {
        0: "[Na+].[OH-]",
        1: "OC([O-])=O.[Na+]",
        2: "[Cs+].[F-]",
        3: "O=P([O-])([O-])[O-].[K+].[K+].[K+]",
        4: "[K+].[OH-]",
        5: "CC([O-])C.[Li+]",
        6: "CCN(CC)CC",
    },
    "ligand": {
        0: "CC(P(C(C)(C)C)C(C)(C)C)(C)C",
        1: "P(C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3",
        2: "CC(C)(C)P(C(C)(C)C)C1=CC=C(N(C)C)C=C1",
        3: "P(C1CCCCC1)(C2CCCCC2)C3CCCCC3",
        4: "CC1=CC=CC=C1P(C2=CC=CC=C2C)C3=CC=CC=C3C",
        5: "CCCCP(C12C[C@@H]3C[C@@H](C[C@H](C2)C3)C1)C45C[C@H]6C[C@@H](C5)C[C@@H](C4)C6",
        6: "COC1=CC=CC(OC)=C1C2=C(P(C3CCCCC3)C4CCCCC4)C=CC=C2",
        7: "CC(C)(P(C(C)(C)C)[c-]1cccc1)C.CC(C)(P(C(C)(C)C)[c-]2cccc2)C.[Fe+2]",
        8: "CC(C1=C(C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)C(C(C)C)=CC(C(C)C)=C1)C",
        9: "[c-]1(P(C2=CC=CC=C2)C3=CC=CC=C3)cccc1.[c-]4(P(C5=CC=CC=C5)C6=CC=CC=C6)cccc4.[Fe+2]",
        10: "CC1(C)C2=C(OC3=C1C=CC=C3P(C4=CC=CC=C4)C5=CC=CC=C5)C(P(C6=CC=CC=C6)C7=CC=CC=C7)=CC=C2",
    },
    "solvent": {
        0: "N#CC",
        1: "C1COCC1",
        2: "O=CN(C)C",
        3: "CO",
    },
}

for _dim_name, _opts in REAGENT_METADATA.items():
    for _idx, _meta in _opts.items():
        _meta.setdefault("smiles", _SMILES_BY_DIM_IDX[_dim_name][_idx])

_SMILES_TO_METADATA = {
    (dim_name, meta["smiles"]): meta
    for dim_name, opts in REAGENT_METADATA.items()
    for meta in opts.values()
}



# ── JSON parsing (tolerant to markdown fences) ───────────────────────────────
def _parse_json_response(text: str) -> Optional[Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = text.find(opener)
        if start < 0:
            continue
        depth = 0
        in_str = False
        escape = False
        for end in range(start, len(text)):
            ch = text[end]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:end + 1])
                    except Exception:
                        break
    # Try the whole text as-is (handles leading whitespace / no-fence cases).
    try:
        return json.loads(text)
    except Exception:
        pass
    # Salvage: model emitted a top-level object whose `candidates` array got
    # truncated (max_tokens hit mid-element). Walk the array element-wise and
    # accept the longest contiguous prefix of complete `{...}` entries.
    salvaged = _salvage_truncated_object(text)
    if salvaged is not None:
        return salvaged
    return None


def _salvage_truncated_object(text: str) -> Optional[dict]:
    """Best-effort recovery of `{"<key>": [<elem>, <elem>, …<truncated>]}`
    when the response was cut off mid-array. Returns a dict containing only
    the keys whose values fully closed (scalars, completed objects, completed
    arrays) plus, if found, an array key with its longest valid prefix of
    complete object/scalar elements.
    """
    obj_start = text.find("{")
    if obj_start < 0:
        return None
    arr_start = text.find("[", obj_start)
    if arr_start < 0:
        return None
    # Key immediately preceding the array: "<key>" : [
    key_match = re.search(r'"([^"]+)"\s*:\s*\[', text[obj_start:arr_start + 1])
    if not key_match:
        return None
    key = key_match.group(1)
    # Walk the array character-by-character collecting complete top-level
    # elements (objects, arrays, strings, or numbers).
    i = arr_start + 1
    elements: list = []
    n = len(text)
    while i < n:
        # skip whitespace and commas between elements
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n or text[i] == "]":
            break
        # Scan one element with brace/bracket/string awareness; bail on truncation.
        depth = 0
        in_str = False
        escape = False
        j = i
        terminated = False
        while j < n:
            cj = text[j]
            if in_str:
                if escape:
                    escape = False
                elif cj == "\\":
                    escape = True
                elif cj == '"':
                    in_str = False
                j += 1
                continue
            if cj == '"':
                in_str = True
            elif cj in "{[":
                depth += 1
            elif cj in "}]":
                depth -= 1
                if depth == 0:
                    j += 1
                    terminated = True
                    break
                if depth < 0:  # we've stepped past the array close
                    break
            elif depth == 0 and cj == ",":
                terminated = True  # bare scalar (number/literal)
                break
            j += 1
        if not terminated:
            break  # truncated mid-element — stop here
        elem_text = text[i:j].strip().rstrip(",").strip()
        if not elem_text:
            break
        try:
            elements.append(json.loads(elem_text))
        except Exception:
            break  # malformed element — stop here, keep prior good ones
        i = j
    if not elements:
        return None
    return {key: elements}
