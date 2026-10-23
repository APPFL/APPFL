# PMO oracles, vendored

PMO (Practical Molecular Optimization, Gao et al., NeurIPS 2022 Datasets &
Benchmarks) as a self-contained black-box benchmark: **rdkit and numpy are the
only third-party runtime dependencies**. Nothing here touches the network at
call time, so it runs on compute nodes with no internet.

The oracle implementations are copied from PyTDC 1.1.15
(`tdc.chem_utils.oracle.oracle`, MIT, Copyright (c) 2020 TDC Team) rather than
reimplemented, so scores are numerically identical to PyTDC's; the goal-directed
definitions originate in GuacaMol (MIT, BenevolentAI). `test_oracles.py` holds
that parity to 1e-6 against recorded PyTDC values.

## Files

| File | Lines | What |
|---|---|---|
| `oracles.py` | 581 | The 20 oracles. `get_oracle(name)`, `ORACLES`, plus `canonical()` and `murcko_scaffold()` helpers. |
| `pool.py` | 283 | `Pool.load()` (packed ECFP4 bit matrix + cache), `tanimoto()`, `descriptors()`. |
| `fetch_pool.py` | 192 | Downloads, canonicalizes and de-duplicates ZINC 250k. Login node only. |
| `make_goldens.py` | 164 | Regenerates `goldens.json` from an installed PyTDC. Login node only. |
| `test_oracles.py` | 155 | Offline parity / range / determinism tests. |
| `goldens.json` | 51 KB | Reference PyTDC scores: 20 oracles x 45 SMILES = 900 pairs. Checked in as data. |

## Install

A fresh environment needs exactly this (rdkit 2026.9.1 is the version these
numbers were measured on; `rdkit` pulls in numpy, pinned here for the record):

```sh
python -m pip install "rdkit==2026.9.1" "numpy==2.2.6"
```

Regenerating `goldens.json` additionally needs PyTDC and the packages its import
chain reaches for. `goldens.json` is checked in, so this is only for refreshing
it:

```sh
python -m pip install "PyTDC==1.1.15" setuptools "pandas<3" huggingface_hub seaborn scikit-learn
```

## Commands

On a **login node with internet** — fetch the candidate pool (required once),
and optionally refresh the goldens:

```sh
python -I fetch_pool.py "$POOL_DIR"      # writes $POOL_DIR/zinc250k.smi + pool.meta.json
python -I make_goldens.py                # optional; overwrites goldens.json
```

On a **compute node, offline** — run the tests:

```sh
python -I test_oracles.py
```

Build or warm the fingerprint cache (also offline; `--limit` subsamples):

```sh
python -I pool.py "$POOL_DIR/zinc250k.smi" --limit 20000
```

`-I` (isolated mode) keeps the interpreter from importing anything out of the
data directory; the scripts put their own directory on `sys.path` explicitly so
this still works.

## Oracles

20 tasks, all pure rdkit, PMO's lowercase names:

```
qed                     celecoxib_rediscovery   troglitazone_rediscovery
thiothixene_rediscovery albuterol_similarity    mestranol_similarity
isomers_c7h8n2o2        isomers_c9h10n2o2pf2cl  median1
median2                 osimertinib_mpo         fexofenadine_mpo
ranolazine_mpo          perindopril_mpo         amlodipine_mpo
sitagliptin_mpo         zaleplon_mpo            valsartan_smarts
deco_hop                scaffold_hop
```

Each maps one SMILES to a float in [0, 1], higher is better, 0.0 for input rdkit
cannot parse.

**Excluded: `drd2`, `gsk3b`, `jnk3`** — the three remaining PMO tasks (23 total,
20 here). They are scikit-learn classifiers whose weights PyTDC downloads as
pickles on first call, so they need both network access and an sklearn new
enough to unpickle but old enough not to have changed the estimator layout.
Those are the two liabilities this module exists to remove, so they are left
out; `get_oracle("drd2")` raises a `KeyError` that says why. PyTDC's other
network-dependent oracles (SA via `fpscores`, docking, retrosynthesis) are
outside PMO and absent here too.

### One deliberate divergence from PyTDC

rdkit parses `""` into a zero-atom molecule rather than `None`, so PyTDC scores
the empty string: `qed("") = 0.339`, `deco_hop("") = 0.500`,
`scaffold_hop("") = 0.333`. A generator could farm that for free reward, so
`oracles.py` treats any zero-atom molecule as invalid and returns 0.0.
`TestDivergence` pins this down and `TestParity` skips that one probe; the other
880 pairs match PyTDC exactly.

## Measured

Single core, rdkit 2026.9.1 / numpy 2.2.6 / Python 3.11.17.

| | |
|---|---|
| `test_oracles.py` | **11 tests, OK**, 880 pairs compared, 0.8 s |
| `import oracles` | 0.27 s (builds all 20 target fingerprints eagerly) |
| cheapest oracle | `celecoxib_rediscovery` **3,360 calls/s** |
| most expensive oracle | `scaffold_hop` **272 calls/s** (Gobbi 2D pharmacophore fingerprint) |
| `qed` | 965 calls/s |
| all 20 oracles, 1 molecule | ~19 ms (1,040 single-oracle calls/s aggregate) |
| `fetch_pool.py`, full pool | 38 s (3.0 MB download + canonicalizing 249,455 SMILES) |
| `zinc250k.smi` | 11.3 MB, 249,455 molecules |
| fingerprinting | **6,800 molecules/s**; 20,000 took 2.9 s, peak RSS 81 MB |
| `tanimoto` 1 vs 20,000 | 15.6 ms (**1.3M pairs/s**) |
| `descriptors` | 840 molecules/s (QED dominates) |
| cache, 20,000 | 1.3 MB `.npz` on disk, 5.1 MB in RAM; cache-hit load 0.03 s |

**Full pool, extrapolated**: ~37 s to
fingerprint all 249,455, giving a 249,455 x 256 B = **63.9 MB** uint8 matrix and
roughly a 17 MB `.npz`. Peak RSS should land near 250 MB — the matrix plus the
SMILES text. `--limit N` builds a subsample and is keyed separately in the
cache.

## Pool

`fetch_pool.py` tries three sources in order and uses the first that downloads.
The one that worked:

- **`mol_opt`** — `https://raw.githubusercontent.com/wenhao-gao/mol_opt/main/data/zinc.txt.gz`,
  PMO's own copy of the pool. sha256
  `cf1f05bfed0c334a38aa26f27e4a5016dd09f5e225b86dcad490bb5f203fb099` (pinned and
  verified). Despite the `.gz` name it is a gzipped **tar** holding one
  `zinc.txt`; the script handles tar.gz, plain gz and plain text alike.
  249,455 raw SMILES, 0 unparseable, 0 duplicates after rdkit canonicalization
  → **249,455 molecules**.

Fallbacks, reachable but **not** sha256-pinned (the script warns and records the
digest it saw): `chemical_vae`'s `250k_rndm_zinc_drugs_clean_3.csv` (same pool,
CSV, quoted SMILES with embedded newlines, so it is read with `csv.reader`) and
guacamol's v1 ChEMBL training set (a different, ~1.27M-molecule distribution —
use `--source guacamol` deliberately, not as a silent substitute).

## Notes on `pool.py`

Fingerprints are ECFP4 (Morgan, radius 2, 2048 bits) stored **packed**, 8 bits
per uint8, so the pool is 256 B per molecule. `tanimoto()` popcounts through a
256-entry uint8 lookup table, avoiding `np.unpackbits` and any extra dependency;
it was checked against rdkit's `BulkTanimotoSimilarity` over 300x300 pairs, max
absolute difference 1.6e-08 (float32 rounding — the return dtype is float32),
and `popcount()` matches `GetNumOnBits()` exactly. The `np.savez_compressed`
cache is keyed by the `.smi` file's sha256, radius, nbits and limit, and
re-validates the digest on load. SMILES go into the cache as one newline-joined
byte blob: `np.array(list_of_str)` would pad every entry to the longest SMILES
and turn 11 MB of text into ~110 MB.
