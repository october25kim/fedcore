"""Golden regression check for the structure-only refactor.

Re-runs tests/golden_capture.py into a temp dir and verifies the deterministic JSON outputs
match tests/golden/ bit-for-bit (floats: abs diff <= TOL; ints/strings: exact). ALSO re-runs
the three CPU scripts (exp_lemma_L, exp_pooling_fail, run_smoke) and tolerantly compares their
stdout against the pinned tests/golden/*.stdout.txt snapshots: whitespace is normalized, floats
are compared within STDOUT_TOL (1e-6), remaining text must match exactly, and absolute
filesystem paths are canonicalized to <ROOT>/<repo-relative-suffix> so the check is portable
across clone locations (the run_smoke "saved ..." line prints an absolute CWD path). The two
aggregator goldens are enforced separately by the Makefile's agg-* targets (repro-check), not
here. Artifact-dependent checks (certify_frozen) LOUD-SKIP when runs/*_logits.npz are absent.
Exit 0 = PASS.

Run BEFORE every refactor commit:  python tests/golden_check.py
(Container-equivalent: bash scripts/docker_test.sh once it wraps this.)
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile

TOL = 1e-9         # JSON deterministic goldens (certificate math / scores / splits)
STDOUT_TOL = 1e-6  # floats embedded in the three CPU scripts' stdout
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOLD = os.path.join(HERE, "golden")
FAILS = []
FROZEN_LOGITS = [
    "runs/cifar10_d5_resnet18_seed0_logits.npz",
    "runs/cifar100_d5_none0.0_seed0_logits.npz",
]
# CPU scripts whose stdout is pinned and must stay reproducible (no artifacts needed).
CPU_STDOUT = [
    ("exp_lemma_L", "exp_lemma_L.stdout.txt"),
    ("exp_pooling_fail", "exp_pooling_fail.stdout.txt"),
    ("run_smoke", "run_smoke.stdout.txt"),
]


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


# --------------------------------------------------------------------------- #
# tolerant stdout comparison (whitespace + floats-within-1e-6 + path canonicalization)
# --------------------------------------------------------------------------- #
_ABS_RE = re.compile(r"/(?:[^/\s]+/)*[^/\s]+")     # candidate absolute paths (leading slash)
_FLOAT_RE = re.compile(
    r"[-+]?(?:\d+\.\d*(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+|\.\d+(?:[eE][-+]?\d+)?|\d+)"
)
_SENT = "\x00"                                     # placeholder for extracted floats
_TOP = None


def _canon_paths(text):
    """Replace the volatile absolute prefix of any repo path with <ROOT>, keeping the
    repo-relative suffix. Generic (not tied to one line): an absolute token is rewritten only
    when one of its segments is a repo top-level entry AND the resulting repo-relative path (or
    its parent dir) exists under ROOT -- so foreign roots normalize to the same <ROOT>/... form
    on both the golden and the freshly-captured side, and non-repo paths are left untouched.
    Leftmost resolving split wins => longest repo-relative suffix is preserved.
    """
    global _TOP
    if _TOP is None:
        _TOP = set(os.listdir(ROOT))

    def repl(m):
        p = m.group(0)
        segs = p.split("/")
        for k in range(1, len(segs)):
            if segs[k] in _TOP:
                cand = os.path.join(ROOT, *segs[k:])
                if os.path.exists(cand) or os.path.isdir(os.path.dirname(cand)):
                    return "<ROOT>/" + "/".join(segs[k:])
        return p

    return _ABS_RE.sub(repl, text)


def _norm_lines(text):
    lines = [re.sub(r"[ \t]+", " ", ln).rstrip() for ln in _canon_paths(text).splitlines()]
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def _cmp_stdout(name, gold, got):
    gl, tl = _norm_lines(gold), _norm_lines(got)
    if len(gl) != len(tl):
        FAILS.append(f"{name} stdout: line count {len(gl)} != {len(tl)}")
    for i, (g, t) in enumerate(zip(gl, tl)):
        if g == t:
            continue
        gtxt, ttxt = _FLOAT_RE.sub(_SENT, g), _FLOAT_RE.sub(_SENT, t)
        if gtxt != ttxt:
            FAILS.append(f"{name} stdout L{i+1} text: |{g}| != |{t}|")
            continue
        gn, tn = _FLOAT_RE.findall(g), _FLOAT_RE.findall(t)
        if len(gn) != len(tn):
            FAILS.append(f"{name} stdout L{i+1} numcount: |{g}| != |{t}|")
            continue
        for a, b in zip(gn, tn):
            fa, fb = float(a), float(b)
            if abs(fa - fb) > STDOUT_TOL and abs(fa - fb) > STDOUT_TOL * max(abs(fa), abs(fb)):
                FAILS.append(f"{name} stdout L{i+1} float {a} != {b} (|Δ|={abs(fa-fb):.2e} > {STDOUT_TOL})")


def main():
    tmp = tempfile.mkdtemp(prefix="golden_check_")
    env = dict(os.environ, GOLDEN_OUT=tmp)
    r = subprocess.run([sys.executable, os.path.join(HERE, "golden_capture.py")],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print("golden_capture.py FAILED to run:\n", r.stdout, r.stderr); sys.exit(2)

    json_files = ["certificate_math.json", "scores_selector.json", "split_determinism.json"]
    for name in json_files:
        g, n = os.path.join(GOLD, name), os.path.join(tmp, name)
        if not os.path.exists(g):
            FAILS.append(f"{name}: golden missing"); continue
        _cmp(name, json.load(open(g)), json.load(open(n)))
    print(f"GOLDEN CHECK: RAN 3 JSON goldens ({', '.join(json_files)})")

    # three CPU scripts: run and tolerantly compare stdout to the pinned snapshots
    for mod, gf in CPU_STDOUT:
        gpath = os.path.join(GOLD, gf)
        if not os.path.exists(gpath):
            FAILS.append(f"{gf}: golden missing"); continue
        before = len(FAILS)
        run = subprocess.run([sys.executable, "-m", f"fedcore.experiments.{mod}"],
                             cwd=ROOT, env=dict(os.environ), capture_output=True, text=True)
        if run.returncode != 0:
            FAILS.append(f"{mod}: exited {run.returncode}\n{run.stderr.strip()[:800]}")
        else:
            _cmp_stdout(mod, open(gpath).read(), run.stdout)
        status = "PASS" if len(FAILS) == before else "FAIL"
        print(f"GOLDEN CHECK: RAN stdout {mod} -> {status}")

    frozen_present = all(os.path.exists(os.path.join(ROOT, p)) for p in FROZEN_LOGITS)
    if frozen_present:
        name = "certify_frozen.json"
        g, n = os.path.join(GOLD, name), os.path.join(tmp, name)
        if not os.path.exists(g):
            FAILS.append(f"{name}: golden missing")
        else:
            _cmp(name, json.load(open(g)), json.load(open(n)))
    else:
        print("GOLDEN CHECK: SKIP certify_frozen.json (runs/*_logits.npz absent)")

    n_fail = len(FAILS)
    if n_fail:
        print(f"GOLDEN CHECK: FAIL ({n_fail} diffs)")
        for f in FAILS[:40]:
            print("  ", f)
    else:
        print(f"GOLDEN CHECK: PASS (JSON <= {TOL}; stdout floats <= {STDOUT_TOL}, paths canonicalized)")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
