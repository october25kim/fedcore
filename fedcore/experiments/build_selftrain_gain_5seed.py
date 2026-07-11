"""Task E: assemble the 5-seed certified self-training gain table (FedPD base, 4x budget).

Reads the raw one-shot self-training run rows for the FedPD-PROSER base at the
4x-audit cell (alpha=0.20, beta=1.0) and, per seed, computes:

  certified_gain    = known_acc(certified) - known_acc(none)
  oracle_gain       = known_acc(oracle)    - known_acc(none)
  admitted          = admitted_count(certified)   (pseudo-labels admitted)
  max_contamination = realized_contam(certified)  (label-noise among admitted)

Seeds 0-2 come from runs/selftrain_pkg.csv (already produced); seeds 3-4 from
runs/selftrain_pkg_5seed.csv (Task E extension). Emits runs/selftrain_gain_5seed.csv
with schema: seed,certified_gain,oracle_gain,admitted,max_contamination
and prints the mean+/-std summary.

Run: python experiments/fedcore/build_selftrain_gain_5seed.py   (CPU, no torch)
"""

from __future__ import annotations

import csv
import math
import os

import numpy as np

from fedcore.io_utils import atomic_write_csv

RAW_SOURCES = ["runs/selftrain_pkg.csv", "runs/selftrain_pkg_5seed.csv"]
BASE, ALPHA, AUDIT, BETA = "FedPD-PROSER", 0.2, "4", "1.0"


def _load_rows():
    """{(seed, mode): row} for the FedPD-PROSER 4x, beta=1.0 cell."""
    out = {}
    for src in RAW_SOURCES:
        if not os.path.exists(src):
            print(f"[warn] missing raw source {src}")
            continue
        with open(src) as f:
            for r in csv.DictReader(f):
                if r["base_model"] != BASE:
                    continue
                if abs(float(r["alpha"]) - ALPHA) > 1e-9:
                    continue
                if str(r["audit_mult"]) != AUDIT or str(float(r["beta"])) != BETA:
                    continue
                out[(int(r["seed"]), r["mode"])] = r
    return out


def _acc(rows, seed, mode):
    r = rows.get((seed, mode))
    if r is None:
        return None
    v = float(r["known_acc"])
    return None if math.isnan(v) else v


def main() -> None:
    rows = _load_rows()
    seeds = sorted({s for (s, _m) in rows})
    fields = ["seed", "certified_gain", "oracle_gain", "admitted", "max_contamination"]
    out_rows = []
    for s in seeds:
        acc_none = _acc(rows, s, "none")
        acc_cert = _acc(rows, s, "certified")
        acc_orac = _acc(rows, s, "oracle")
        cert_row = rows.get((s, "certified"), {})
        # certified halting => admitted 0 and known_acc==none (no gain); treat missing acc as none.
        cg = (acc_cert - acc_none) if (acc_cert is not None and acc_none is not None) else \
             (0.0 if acc_none is not None else float("nan"))
        og = (acc_orac - acc_none) if (acc_orac is not None and acc_none is not None) else float("nan")
        admitted = int(float(cert_row.get("admitted_count", 0)))
        contam = cert_row.get("realized_contam", "nan")
        contam = float(contam) if contam not in ("", "nan", None) else float("nan")
        out_rows.append({"seed": s, "certified_gain": round(cg, 4),
                         "oracle_gain": round(og, 4), "admitted": admitted,
                         "max_contamination": round(contam, 4) if not math.isnan(contam) else float("nan")})

    atomic_write_csv("runs/selftrain_gain_5seed.csv", fields, out_rows)
    print(f"saved runs/selftrain_gain_5seed.csv  ({len(out_rows)} seeds: {seeds})\n")

    print(f"{'seed':>4} {'cert_gain':>10} {'oracle_gain':>12} {'admitted':>9} {'max_contam':>11}")
    for r in out_rows:
        mc = f"{r['max_contamination']:.4f}" if not (isinstance(r['max_contamination'], float)
                                                     and math.isnan(r['max_contamination'])) else "nan"
        print(f"{r['seed']:>4} {r['certified_gain']:>10.4f} {r['oracle_gain']:>12.4f} "
              f"{r['admitted']:>9} {mc:>11}")

    def _ms(key):
        vals = [r[key] for r in out_rows if not (isinstance(r[key], float) and math.isnan(r[key]))]
        return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (float("nan"), float("nan"), 0)

    print("\n=== mean +/- std over seeds ===")
    for key in ("certified_gain", "oracle_gain", "admitted", "max_contamination"):
        m, sd, n = _ms(key)
        print(f"{key:>18}: {m:.4f} +/- {sd:.4f}  (n={n})")


if __name__ == "__main__":
    main()
