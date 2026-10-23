"""The candidate pool: SMILES, packed ECFP4 bits, Tanimoto, and descriptors.

A PMO search space is a fixed pool of purchasable molecules (ZINC 250k, written
by `fetch_pool.py`). This module turns that `.smi` file into the two arrays a
search needs -- a bit matrix for similarity and a few descriptors for region /
property windows -- with rdkit and numpy as the only dependencies.

Fingerprints are ECFP4 (Morgan, radius 2, 2048 bits) kept **packed**, 8 bits to
a uint8, so the whole pool is 250k x 256 B = 64 MB and fits comfortably in
memory on any node. Tanimoto popcounts through a 256-entry uint8 lookup table, which avoids
both `np.unpackbits` (8x the memory traffic) and any extra dependency.

    python -I pool.py POOL.smi [--limit N] [--radius 2] [--nbits 2048]

builds (or reuses) the `np.savez_compressed` cache next to the `.smi` file and
prints timings. In code::

    pool = Pool.load("zinc250k.smi", limit=20000)
    sims = tanimoto(pool.fps[0], pool.fps)        # (N,) float32 in [0, 1]
    props = descriptors(pool.smiles[:1000])       # {"mw": ..., "qed": ...}
"""

from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, QED, rdFingerprintGenerator, rdMolDescriptors

rdBase.DisableLog("rdApp.error")

__all__ = ["Pool", "tanimoto", "descriptors", "fingerprint_matrix", "popcount"]

DEFAULT_RADIUS = 2
DEFAULT_NBITS = 2048

# popcount(byte) for every uint8 value: the whole trick behind `tanimoto`.
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount(packed: np.ndarray) -> np.ndarray:
    """Bits set per row of a packed uint8 array (int32, so 2048 bits cannot overflow)."""
    return _POPCOUNT[packed].sum(axis=-1, dtype=np.int32)


def tanimoto(packed_a: np.ndarray, packed_b: np.ndarray) -> np.ndarray:
    """Tanimoto of one packed fingerprint against a packed matrix (or another row).

    `packed_a` is (n_bytes,) and `packed_b` is (N, n_bytes) or (n_bytes,).
    Returns float32; a pair with no bits set at all scores 0.0.
    """
    packed_b = np.atleast_2d(packed_b)
    intersection = popcount(np.bitwise_and(packed_a, packed_b))
    union = popcount(packed_a) + popcount(packed_b) - intersection
    out = np.zeros(intersection.shape, dtype=np.float32)
    np.divide(intersection, union, out=out, where=union > 0, casting="unsafe")
    return out


def _generator(radius: int, nbits: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)


def fingerprint_matrix(
    smiles: list[str], radius: int = DEFAULT_RADIUS, nbits: int = DEFAULT_NBITS
) -> np.ndarray:
    """Packed ECFP bit matrix, shape (len(smiles), nbits // 8), dtype uint8.

    An unparseable SMILES gets an all-zero row rather than being dropped, so row
    i always lines up with `smiles[i]`.
    """
    generator = _generator(radius, nbits)
    matrix = np.zeros((len(smiles), nbits // 8), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        matrix[i] = np.packbits(generator.GetFingerprintAsNumPy(mol))
    return matrix


def descriptors(smiles_list: list[str]) -> dict[str, np.ndarray]:
    """MW, logP, TPSA, ring count and QED per molecule, as parallel arrays.

    These are the quantities region and property windows get carved out of.
    Unparseable or unscoreable molecules get NaN (and -1 rings), never an
    exception.
    """
    n = len(smiles_list)
    out = {
        key: np.full(n, np.nan, dtype=np.float32)
        for key in ("mw", "logp", "tpsa", "qed")
    }
    out["num_rings"] = np.full(n, -1, dtype=np.int16)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        out["mw"][i] = Descriptors.MolWt(mol)
        out["logp"][i] = Descriptors.MolLogP(mol)
        out["tpsa"][i] = Descriptors.TPSA(mol)
        out["num_rings"][i] = rdMolDescriptors.CalcNumRings(mol)
        try:
            out["qed"][i] = QED.qed(mol)
        except Exception:
            pass
    return out


def _blob(smiles: list[str]) -> np.ndarray:
    """SMILES as one uint8 byte array.

    `np.array(list_of_str)` would pad every entry to the longest SMILES (~110
    chars, 4 bytes each), turning 11 MB of text into ~110 MB in RAM and in the
    cache. A newline-joined blob costs exactly the text.
    """
    return np.frombuffer("\n".join(smiles).encode("utf-8"), dtype=np.uint8)


def _unblob(blob: np.ndarray) -> list[str]:
    return blob.tobytes().decode("utf-8").split("\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Pool:
    """A fixed pool of molecules with its packed fingerprint matrix."""

    smiles: list[str]
    fps: np.ndarray
    radius: int
    nbits: int
    source: Path

    def __len__(self) -> int:
        return len(self.smiles)

    @staticmethod
    def read_smi(path: Path, limit: int | None = None) -> list[str]:
        """First whitespace-delimited field of each non-blank line."""
        smiles = []
        with Path(path).open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                smiles.append(line.split()[0])
                if limit is not None and len(smiles) >= limit:
                    break
        return smiles

    @classmethod
    def cache_path(
        cls,
        path: Path,
        radius: int = DEFAULT_RADIUS,
        nbits: int = DEFAULT_NBITS,
        limit: int | None = None,
        sha256: str | None = None,
    ) -> Path:
        """Cache name keyed by the `.smi` content digest, radius, nbits and limit."""
        path = Path(path)
        sha256 = sha256 or file_sha256(path)
        tag = "all" if limit is None else str(limit)
        return path.with_name(f"{path.stem}.fp-{sha256[:12]}-r{radius}-n{nbits}-{tag}.npz")

    @classmethod
    def load(
        cls,
        path: Path,
        radius: int = DEFAULT_RADIUS,
        nbits: int = DEFAULT_NBITS,
        limit: int | None = None,
        use_cache: bool = True,
        verbose: bool = False,
    ) -> "Pool":
        """Load a `.smi` pool, building the fingerprint matrix or reusing the cache."""
        path = Path(path)
        sha256 = file_sha256(path)
        cache = cls.cache_path(path, radius, nbits, limit, sha256)

        if use_cache and cache.exists():
            with np.load(cache, allow_pickle=False) as data:
                if (
                    str(data["sha256"]) == sha256
                    and int(data["radius"]) == radius
                    and int(data["nbits"]) == nbits
                ):
                    if verbose:
                        print(f"cache hit  {cache.name}")
                    return cls(
                        smiles=_unblob(data["smiles"]),
                        fps=data["fps"],
                        radius=radius,
                        nbits=nbits,
                        source=path,
                    )

        smiles = cls.read_smi(path, limit)
        start = time.perf_counter()
        fps = fingerprint_matrix(smiles, radius, nbits)
        elapsed = time.perf_counter() - start
        if verbose:
            print(
                f"fingerprinted {len(smiles)} molecules in {elapsed:.1f}s "
                f"({len(smiles) / elapsed:,.0f}/s), matrix {fps.nbytes / 1e6:.1f} MB"
            )

        if use_cache:
            np.savez_compressed(
                cache,
                fps=fps,
                smiles=_blob(smiles),
                sha256=sha256,
                radius=radius,
                nbits=nbits,
                limit=-1 if limit is None else limit,
            )
            if verbose:
                print(f"cache -> {cache} ({cache.stat().st_size / 1e6:.1f} MB)")

        return cls(smiles=smiles, fps=fps, radius=radius, nbits=nbits, source=path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("smi", type=Path, help="pool file, one SMILES per line")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N")
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS)
    parser.add_argument("--nbits", type=int, default=DEFAULT_NBITS)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    pool = Pool.load(
        args.smi,
        radius=args.radius,
        nbits=args.nbits,
        limit=args.limit,
        use_cache=not args.no_cache,
        verbose=True,
    )
    print(f"pool   {len(pool)} molecules from {pool.source}")
    print(f"fps    {pool.fps.shape} uint8, {pool.fps.nbytes / 1e6:.1f} MB")

    start_sim = time.perf_counter()
    sims = tanimoto(pool.fps[0], pool.fps)
    sim_elapsed = time.perf_counter() - start_sim
    print(
        f"tanimoto 1 vs {len(pool)} in {sim_elapsed * 1e3:.1f} ms "
        f"({len(pool) / sim_elapsed / 1e6:.1f}M pairs/s); "
        f"self={sims[0]:.3f} mean={sims.mean():.4f} max_other={np.sort(sims)[-2]:.3f}"
    )

    probe = pool.smiles[: min(2000, len(pool))]
    start_desc = time.perf_counter()
    props = descriptors(probe)
    desc_elapsed = time.perf_counter() - start_desc
    print(
        f"descriptors {len(probe)} molecules in {desc_elapsed:.1f}s "
        f"({len(probe) / desc_elapsed:,.0f}/s)"
    )
    for key, values in props.items():
        finite = values[np.isfinite(values)] if values.dtype.kind == "f" else values
        print(f"  {key:9s} min {finite.min():8.2f}  mean {finite.mean():8.2f}  max {finite.max():8.2f}")
    print(f"total  {time.perf_counter() - start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
