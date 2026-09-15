"""Reproduce the corrected T3 replay (Table 5 of the manuscript) from released artifacts.

Loads t3_corrected_fixed_traffic.py unmodified except for four path substitutions, each
asserted to apply exactly once, that point it at this repository instead of the sandbox
it was written in. No certification logic is touched.

Run from the repository root inside the project image:

    docker run --rm -v "$PWD":/w -w /w fedcore-c400r:latest \
        python3 results/fk_t3_uncertainty_e6/reproduce_corrected_t3.py

Expected: simplex/bounded = 0/0, 21/21, 81/104, 177/199, 256/288, 343/358 at
alpha = 0.05 .. 0.30, then GATE PASS. Exit status 2 on any mismatch.
"""
import collections
import hashlib
import pathlib
import sys
import types

SRC = pathlib.Path("/w/results/fk_t3_uncertainty_e6/t3_corrected_fixed_traffic.py")
SRC_SHA = "bbf5542eedb1ed1d7947265773662c6fbeb04dd2615b854ea547c5d5f4614096"
EXPECTED = {0.05: (0, 0), 0.10: (21, 21), 0.15: (81, 104),
            0.20: (177, 199), 0.25: (256, 288), 0.30: (343, 358)}


def first_present(*candidates):
    for c in candidates:
        if pathlib.Path(c).exists():
            return c
    raise SystemExit("none of these inputs was found: " + ", ".join(candidates))


# The count tensor sits at different paths in the distribution and in the working
# repository; the archived traffic realisation sits at one path in both.
COUNTS_PATH = first_present(
    "/w/paper/wr-v3/artifacts/primary/primary_candidate_counts.csv.gz",
    "/w/results/theorem_aligned_wr_450_v3/primary_run/primary_candidate_counts.csv.gz")
TRAFFIC_PATH = first_present("/w/results/t3_bounded_lambda/primary_run/t3_cells.csv")
print("[paths] counts  =", COUNTS_PATH)
print("[paths] traffic =", TRAFFIC_PATH)

raw = SRC.read_bytes()
got = hashlib.sha256(raw).hexdigest()
if got != SRC_SHA:
    raise SystemExit("replay script hash mismatch: %s" % got)
src = raw.decode()

SUBS = [
    ('sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inputs"))',
     'sys.path.insert(0, "/w")'),
    ('COUNTS = str(Path(__file__).resolve().parents[1] / "inputs/counts.csv.gz")',
     'COUNTS = %r' % COUNTS_PATH),
    ('from mixture import (',
     'from fedcore.mixture import ('),
    ('path = Path(__file__).resolve().parents[1] / "inputs/t3_cells_original.csv"',
     'path = Path(%r)' % TRAFFIC_PATH),
]
for old, new in SUBS:
    if src.count(old) != 1:
        raise SystemExit("substitution target not unique (%d): %s" % (src.count(old), old))
    src = src.replace(old, new)
print("[gate] module loaded, %d path substitutions applied" % len(SUBS))

mod = types.ModuleType("t3c")
mod.__file__ = str(SRC)
exec(compile(src, str(SRC), "exec"), mod.__dict__)

cells = mod.load_counts(mod.COUNTS)
boxes = {}
agg = collections.defaultdict(lambda: {"s": 0, "b": 0, "n": 0})
for (sid, alpha), cell in sorted(cells.items()):
    members = cell["members"]
    J = len(next(iter(members.values()))["A"])
    if sid not in boxes:
        boxes[sid] = mod.multinomial_mixture_confidence_box(
            mod.traffic_counts(sid, J), mod.DELTA_LAMBDA)
    res = mod.evaluate_cell(members, alpha, boxes[sid])
    a = agg[alpha]
    a["n"] += 1
    a["s"] += res["simplex"]["certified"]
    a["b"] += res["bounded"]["certified"]

ok = True
for alpha in sorted(EXPECTED):
    es, eb = EXPECTED[alpha]
    a = agg[alpha]
    good = (a["s"] == es and a["b"] == eb)
    ok = ok and good
    print("  alpha=%.2f  simplex %3d (exp %3d)  bounded %3d (exp %3d)  %s"
          % (alpha, a["s"], es, a["b"], eb, "OK" if good else "MISMATCH"))
print("GATE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 2)
