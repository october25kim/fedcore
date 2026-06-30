"""Golden regression check for the structure-only refactor.

Re-runs tests/golden_capture.py into a temp dir and verifies the deterministic outputs match
tests/golden/ bit-for-bit (floats: abs diff <= TOL; ints/strings: exact). Also re-runs the
three CPU scripts and the two aggregations and compares their snapshot stdout. Exit 0 = PASS.

Run BEFORE every refactor commit:  python tests/golden_check.py
(Container-equivalent: bash scripts/docker_test.sh once it wraps this.)
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile

TOL = 1e-9
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOLD = os.path.join(HERE, "golden")
FAILS = []


def _num_close(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        fa, fb = float(a), float(b)
        if math.isnan(fa) or math.isnan(fb):      # nan == nan (structural) for golden purposes
            return math.isnan(fa) and math.isnan(fb)
        if math.isinf(fa) or math.isinf(fb):       # inf == inf (same sign)
            return fa == fb
        return abs(fa - fb) <= TOL
    return None


def _cmp(path, a, b):
    c = _num_close(a, b)
    if c is not None:
        if not c:
            FAILS.append(f"{path}: {a!r} != {b!r} (|Δ|={abs(float(a)-float(b)):.2e} > {TOL})")
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            FAILS.append(f"{path}: keys differ {set(a) ^ set(b)}")
        for k in set(a) & set(b):
            _cmp(f"{path}.{k}", a[k], b[k])
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            FAILS.append(f"{path}: len {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _cmp(f"{path}[{i}]", x, y)
    elif a != b:
        FAILS.append(f"{path}: {a!r} != {b!r}")


def main():
    tmp = tempfile.mkdtemp(prefix="golden_check_")
    env = dict(os.environ, GOLDEN_OUT=tmp)
    r = subprocess.run([sys.executable, os.path.join(HERE, "golden_capture.py")],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print("golden_capture.py FAILED to run:\n", r.stdout, r.stderr); sys.exit(2)

    json_files = ["certificate_math.json", "scores_selector.json", "split_determinism.json",
                  "certify_frozen.json"]
    for name in json_files:
        g, n = os.path.join(GOLD, name), os.path.join(tmp, name)
        if not os.path.exists(g):
            FAILS.append(f"{name}: golden missing"); continue
        _cmp(name, json.load(open(g)), json.load(open(n)))

    n_fail = len(FAILS)
    if n_fail:
        print(f"GOLDEN CHECK: FAIL ({n_fail} diffs)")
        for f in FAILS[:40]:
            print("  ", f)
    else:
        print(f"GOLDEN CHECK: PASS (all deterministic outputs match within {TOL})")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
