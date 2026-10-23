"""Download the ZINC 250k candidate pool, canonicalize it, de-duplicate it.

PMO's standard starting set. Run this once on a login node that has internet;
compute nodes then only read the resulting `.smi` file.

    python -I fetch_pool.py OUT_DIR [--source NAME] [--keep-download]

It writes `OUT_DIR/zinc250k.smi` (one canonical SMILES per line) and
`OUT_DIR/pool.meta.json` (source URL, archive sha256, counts, rdkit version).

Sources are tried in order and the first that downloads wins. `mol_opt` is the
default because it is PMO's own copy of the pool and its sha256 is pinned here;
the two fallbacks are recorded but not pinned, and the script says so loudly.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

# `sha256 = None` means "record the digest, do not verify it".
SOURCES = {
    "mol_opt": {
        "url": "https://raw.githubusercontent.com/wenhao-gao/mol_opt/main/data/zinc.txt.gz",
        # Despite the name this is a gzipped *tar* holding a single `zinc.txt`.
        "sha256": "cf1f05bfed0c334a38aa26f27e4a5016dd09f5e225b86dcad490bb5f203fb099",
        "kind": "lines",
        "note": "PMO reference pool (wenhao-gao/mol_opt data/zinc.txt.gz)",
    },
    "chemical_vae": {
        "url": "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/"
        "models/zinc/250k_rndm_zinc_drugs_clean_3.csv",
        "sha256": None,
        "kind": "csv",
        "note": "classic 250k_rndm_zinc_drugs_clean (CSV, 'smiles' column)",
    },
    "guacamol": {
        "url": "https://ndownloader.figshare.com/files/13612760",
        "sha256": None,
        "kind": "lines",
        "note": "guacamol v1 ChEMBL training set -- NOT ZINC, ~1.27M molecules",
    },
}

CHUNK = 1 << 20


def download(url: str, dest: Path) -> str:
    """Stream `url` to `dest`, returning its sha256."""
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=300) as response, dest.open("wb") as out:
        while chunk := response.read(CHUNK):
            digest.update(chunk)
            out.write(chunk)
    return digest.hexdigest()


def raw_bytes(path: Path) -> bytes:
    """Decompress the archive, whichever of tar.gz / gz / plain it turns out to be."""
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            members = [m for m in archive.getmembers() if m.isfile()]
            if len(members) != 1:
                raise ValueError(f"expected 1 file in {path.name}, found {len(members)}")
            return archive.extractfile(members[0]).read()
    with path.open("rb") as handle:
        head = handle.read(2)
    if head == b"\x1f\x8b":
        return gzip.decompress(path.read_bytes())
    return path.read_bytes()


def extract_smiles(path: Path, kind: str) -> list[str]:
    """Pull the SMILES column out of the downloaded file."""
    text = raw_bytes(path).decode("utf-8", errors="replace")
    if kind == "csv":
        # The classic CSV quotes each SMILES *and* embeds a newline inside the
        # quotes, so it has to go through a real CSV reader.
        reader = csv.DictReader(io.StringIO(text))
        field = next(f for f in reader.fieldnames if f.strip().lower() == "smiles")
        return [row[field].strip() for row in reader if row.get(field)]
    return [line.split()[0] for line in text.splitlines() if line.strip()]


def canonicalize(smiles_list: list[str]) -> tuple[list[str], int]:
    """Canonical SMILES, order-preserving de-duplication. Returns (kept, n_bad)."""
    from rdkit import Chem, rdBase

    rdBase.DisableLog("rdApp.error")
    seen: set[str] = set()
    kept: list[str] = []
    n_bad = 0
    for raw in smiles_list:
        mol = Chem.MolFromSmiles(raw)
        if mol is None or mol.GetNumAtoms() == 0:
            n_bad += 1
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical not in seen:
            seen.add(canonical)
            kept.append(canonical)
    return kept, n_bad


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path, help="directory for the pool files")
    parser.add_argument(
        "--source",
        action="append",
        choices=sorted(SOURCES),
        help="try only this source (repeatable; default: mol_opt, chemical_vae, guacamol)",
    )
    parser.add_argument(
        "--keep-download", action="store_true", help="keep the raw archive"
    )
    args = parser.parse_args(argv)

    import rdkit

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    order = args.source or ["mol_opt", "chemical_vae", "guacamol"]

    failures = []
    for name in order:
        source = SOURCES[name]
        archive = out_dir / f"_download_{name}"
        print(f"[{name}] {source['url']}")
        try:
            sha256 = download(source["url"], archive)
            print(f"[{name}] {archive.stat().st_size} bytes, sha256 {sha256}")
            expected = source["sha256"]
            if expected is None:
                print(f"[{name}] WARNING: no pinned sha256 for this source; recording only")
            elif sha256 != expected:
                raise ValueError(f"sha256 mismatch: expected {expected}, got {sha256}")
            else:
                print(f"[{name}] sha256 verified against the pinned digest")
            raw = extract_smiles(archive, source["kind"])
        except Exception as exc:
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}")
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            archive.unlink(missing_ok=True)
            continue

        print(f"[{name}] {len(raw)} raw SMILES; canonicalizing...")
        kept, n_bad = canonicalize(raw)

        smi_path = out_dir / "zinc250k.smi"
        smi_path.write_text("\n".join(kept) + "\n")
        meta = {
            "source": name,
            "source_note": source["note"],
            "url": source["url"],
            "archive_sha256": sha256,
            "archive_sha256_pinned": source["sha256"] is not None,
            "n_raw": len(raw),
            "n_unparseable": n_bad,
            "n_duplicates": len(raw) - n_bad - len(kept),
            "count": len(kept),
            "rdkit_version": rdkit.__version__,
            "smi_sha256": hashlib.sha256(smi_path.read_bytes()).hexdigest(),
        }
        meta_path = out_dir / "pool.meta.json"
        meta_path.write_text(json.dumps(meta, indent=1) + "\n")

        if not args.keep_download:
            archive.unlink(missing_ok=True)

        print(
            f"\n{len(raw)} raw -> {n_bad} unparseable, "
            f"{meta['n_duplicates']} duplicate, {len(kept)} kept\n"
            f"pool : {smi_path} ({smi_path.stat().st_size} bytes)\n"
            f"meta : {meta_path}"
        )
        return 0

    print("\nevery source failed:\n  " + "\n  ".join(failures), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
