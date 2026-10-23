"""ADKO v2 run records, in the reference's own schema.

A launcher given ``--results_path`` gathers every agent's trace and writes one JSON file with
the keys the v2 reference implementation writes: ``steps`` (one row per lab and round), ``calibration``, ``trust``,
``token_log``, ``best_int`` and ``system_best``. Its analytics therefore read an APPFL run
unchanged, and a run here can be compared with a run there key by key.

A gRPC site holds only its own trace, so its record covers that one lab.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _token_row(token: Dict[str, Any]) -> Dict[str, Any]:
    """A sent token as the reference logs it: source, round, input, q, text, and the bit."""
    row = dict(
        source=token["provenance"]["source"],
        round=token["provenance"]["round"],
        x=list(token["x"]),
        q=token["q"],
        insight=token["insight"],
    )
    if token["tag"]:
        row["tag"] = token["tag"]
    return row


def run_record(
    traces: Sequence[Dict[str, Any]],
    *,
    seed: int,
    setting: Optional[str],
    n_rounds: int,
    config: Dict[str, Any],
    wall_seconds: float,
) -> Dict[str, Any]:
    """Assemble the agents' traces (from ``ADKOv2Agent.trace``) into one run record."""
    traces = sorted(traces, key=lambda t: t["agent"])
    agents = [t["agent"] for t in traces]
    steps: List[Dict[str, Any]] = []
    for t in traces:
        for s in t["steps"]:
            s = dict(s)
            s["theta_int"] = s.pop("theta")
            steps.append(s)
    steps.sort(key=lambda s: (s["round"], s["agent"]))
    calibration = sorted(
        (c for t in traces for c in t["calibration"]),
        key=lambda c: (c["round"], c["agent"]),
    )
    token_log = sorted(
        (_token_row(k) for t in traces for k in t["tokens"]),
        key=lambda row: (row["round"], row["source"]),
    )
    trust = []
    if any(t["trust"] for t in traces):
        by_round: Dict[int, Dict[int, Dict]] = {}
        for t in traces:
            for rnd, snapshot in t["trust"]:
                by_round.setdefault(int(rnd), {})[t["agent"]] = snapshot
        trust = [{"round": r, "t": by_round[r]} for r in sorted(by_round)]

    best = np.full((n_rounds, len(agents)), -np.inf)
    column = {a: i for i, a in enumerate(agents)}
    for s in steps:
        best[s["round"], column[s["agent"]]] = s["best_so_far"]
    llm = [t["llm"] for t in traces if t["llm"]]
    record = {
        "steps": steps,
        "calibration": calibration,
        "trust": trust,
        "token_log": token_log,
        "method": "ADKO-v2",
        "setting": setting,
        "seed": int(seed),
        "config": config,
        "best_int": best.tolist(),
        "system_best": best.max(axis=1).tolist(),
        "wall_seconds": wall_seconds,
    }
    record["final_system_best"] = record["system_best"][-1]
    if llm:
        record["llm_summary"] = {k: sum(row.get(k, 0) for row in llm) for k in llm[0]}
    return record


def write_run_record(path: str, record: Dict[str, Any]) -> None:
    """Write atomically, so a half-written file is never read as a result."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(record, default=_jsonable))
    os.replace(tmp, target)
    print(f"run record          : {target}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)
