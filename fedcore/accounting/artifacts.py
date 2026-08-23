"""Compact, deterministic persistence of the actual sampled IDs + multiplicities.

Every count in the accounting CSVs must be auditable back to the exact sample IDs
that produced it, so those IDs are written out per draw.

Format: parquet when ``pyarrow`` is importable, else a compressed ``.npz`` with an
identical schema. ``pyarrow`` is not in ``requirements.lock``; making it a hard
requirement would break the "clean checkout reproduces the reports" criterion on the
project's pinned environment. Both writers are deterministic (sorted by sample_id)
and the chosen format is recorded in the manifest.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

try:  # pragma: no cover - environment dependent
    import pyarrow as _pa
    import pyarrow.parquet as _pq

    HAVE_PARQUET = True
except ImportError:  # pragma: no cover
    _pa = _pq = None
    HAVE_PARQUET = False


def artifact_extension() -> str:
    return ".parquet" if HAVE_PARQUET else ".npz"


def write_draw_ids(path_stem: str, records: List[Dict]) -> str:
    """Persist ``[{stratum_type, stratum_id, sample_id, multiplicity}, ...]``.

    Rows are sorted by (stratum_id, sample_id) so the artifact is byte-stable across
    runs and machines.
    """
    stratum_type, stratum_id, sample_id, multiplicity = [], [], [], []
    for rec in records:
        ids, counts = np.unique(np.asarray(rec["sample_ids"]), return_counts=True)
        order = np.argsort(ids.astype(str), kind="stable")
        for i, c in zip(ids[order].tolist(), counts[order].tolist()):
            stratum_type.append(rec["stratum_type"])
            stratum_id.append(int(rec["stratum_id"]))
            sample_id.append(str(i))
            multiplicity.append(int(c))

    os.makedirs(os.path.dirname(os.path.abspath(path_stem)), exist_ok=True)
    if HAVE_PARQUET:
        path = path_stem + ".parquet"
        table = _pa.table(
            {
                "stratum_type": _pa.array(stratum_type, type=_pa.string()),
                "stratum_id": _pa.array(stratum_id, type=_pa.int32()),
                "sample_id": _pa.array(sample_id, type=_pa.string()),
                "multiplicity": _pa.array(multiplicity, type=_pa.int32()),
            }
        )
        _pq.write_table(table, path, compression="snappy")
        return path
    path = path_stem + ".npz"
    np.savez_compressed(
        path,
        stratum_type=np.array(stratum_type, dtype="U16"),
        stratum_id=np.array(stratum_id, dtype=np.int32),
        sample_id=np.array(sample_id, dtype="U32"),
        multiplicity=np.array(multiplicity, dtype=np.int32),
    )
    return path


def read_draw_ids(path: str) -> Dict[str, np.ndarray]:
    """Read back an artifact written by :func:`write_draw_ids` (either format)."""
    if path.endswith(".parquet"):
        t = _pq.read_table(path)
        return {c: np.asarray(t[c]) for c in t.column_names}
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
