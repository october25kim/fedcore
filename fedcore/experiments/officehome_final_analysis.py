"""Office-Home 50-cell FINAL post-hoc certificate analysis (no training).

This is a POST-HOC analysis over the 50 FROZEN Office-Home training-logit
artifacts (``runs/oneshot/officehome/logits/*_logits.npz``).  It trains nothing,
launches nothing, and modifies no frozen artifact.  Every number is computed
from the canonical artifacts using the exact fedcore certificate core
(``fedcore.certificate.joint``, ``fedcore.certificate.cp``); no CP is
reimplemented and no favourable seed/alpha/score is selected.

Design (owner-fixed, immutable -- see docs/agent/officehome_experiment_plan.md):

* J = 4 domain clients {Art, Clipart, Product, Real_World}.
* Selector: proposal-fold ONLY, one global (score in {msp,energy,margin},
  gamma in {0.3,0.5,0.7,1.0}, kappa 0.01) threshold, largest proposal accepted
  coverage (``fedcore.officehome_selector.select_from_artifact``).  Re-selected
  per alpha because the risk buffer is ``gamma * alpha``.
* PRIMARY certificate: client-level FULL SIMPLEX (Delta^3), uniform allocation
  ``rbar_j = U+(K_j, A_j; delta_r/J)``, ``U = max_j rbar_j``; coverage LCB per
  Corollary 1 ``min_j U-(A_j, n_j; delta_c/J)``.  delta_r = delta_c = 0.05.
  PRIMARY alpha = 0.20; secondary grid {0.10,0.15,0.20,0.25,0.30}.
* Certification draw: n_j with-replacement per domain from the cert reservoir
  using the frozen per-cell ``audit_draw_seed`` (outcome-independent; a single
  1000-length common stream per domain, prefixes give {250,500,1000}). Primary
  n_j = 500.
* Traffic-derived Lambda_hat (SECONDARY): domain identities only from the
  traffic fold, m draws with the frozen ``traffic_draw_seed``, two-sided CP tails
  delta_lambda/(2J); delta_lambda/delta_r/delta_c = 0.02/0.04/0.04, m primary
  1000 (grid {250,500,1000,2000}).
* Fixed-rho SECONDARY: rho in {0.05,0.10,0.15,0.25,0.50} around the uniform
  center, identical selector+draw across rho.

Budget composition (documented, valid union bound):

* Full-simplex primary  -> FailureBudget(conditional_risk=delta_r=0.05,
  acceptance_lower=delta_c=0.05).  Two per-client events, delta/J each.
* Traffic box           -> FailureBudget(mixture=delta_lambda=0.02,
  conditional_risk=0.02, acceptance_upper=0.02, acceptance_lower=delta_c=0.04).
  delta_r=0.04 is split evenly over the two upper risk events (rbar, ahigh);
  acceptance_lower (alow) is the shared coverage/denominator event = delta_c.
* Fixed-rho box         -> same risk/coverage events as the traffic box but no
  mixture budget (the rho box is a DECLARED, not estimated, set): total 0.08.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binom as _sp_binom

from fedcore.budget import FailureBudget, allocate_failure_budget
from fedcore.certificate.allocation import zero_error_floor
from fedcore.certificate.cp import cp_upper
from fedcore.certificate.joint import joint_conditional_certificate
from fedcore.mixture import rho_mixture_box
from fedcore.officehome_selector import (
    DEFAULT_GAMMAS,
    DEFAULT_KAPPA,
    DEFAULT_SCORES,
    select_from_artifact,
)
from fedcore.officehome_traffic_lambda import build_traffic_lambda, draw_traffic_client_counts
from fedcore.scores import compute_score
from fedcore.selector import open_set_error

# --------------------------------------------------------------------------- #
# Immutable design constants
# --------------------------------------------------------------------------- #
J = 4
ALPHAS: Tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30)
ALPHA_PRIMARY = 0.20
NJ_GRID: Tuple[int, ...] = (250, 500, 1000)
NJ_PRIMARY = 500
NJ_MAX = 1000

DELTA_R = 0.05          # full-simplex primary risk budget
DELTA_C = 0.05          # full-simplex primary coverage budget

# Traffic / rho box budgets
T_DELTA_LAMBDA = 0.02
T_DELTA_R = 0.04        # split evenly over rbar + ahigh
T_DELTA_C = 0.04        # acceptance_lower (shared)
M_GRID: Tuple[int, ...] = (250, 500, 1000, 2000)
M_PRIMARY = 1000
RHO_GRID: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.25, 0.50)

SCORES = DEFAULT_SCORES
GAMMAS = DEFAULT_GAMMAS
KAPPA = DEFAULT_KAPPA

# Computational (non-semantic) seed for bootstrap resampling only.
BOOTSTRAP_SEED = 20260715
N_BOOTSTRAP = 20000

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOGITS_DIR = os.path.join(REPO, "runs", "oneshot", "officehome", "logits")
CHECKSUMS = os.path.join(
    REPO, "results", "officehome", "launch", "officehome_logits_checksums.sha256"
)
OUTDIR = os.path.join(REPO, "results", "officehome")
DOCDIR = os.path.join(REPO, "docs", "agent")


# --------------------------------------------------------------------------- #
# Artifact loading + fail-closed integrity
# --------------------------------------------------------------------------- #
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frozen_checksums() -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(CHECKSUMS, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            digest, name = line.split()
            out[name] = digest
    return out


@dataclass
class Cell:
    name: str
    path: str
    pipeline: str
    split_id: str
    train_rep: int
    experiment_id: str
    audit_draw_seed: int
    traffic_draw_seed: int
    arrays: Dict[str, np.ndarray]

    @property
    def split_index(self) -> int:
        # "officehome_split_3" -> 3
        return int(self.split_id.rsplit("_", 1)[-1])

    @property
    def block_id(self) -> str:
        return f"{self.split_id}__rep{self.train_rep}"


def load_cells(frozen: Dict[str, str]) -> List[Cell]:
    cells: List[Cell] = []
    names = sorted(frozen)
    if len(names) != 50:
        raise RuntimeError(f"expected 50 frozen logit files, found {len(names)}")
    for name in names:
        path = os.path.join(LOGITS_DIR, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"FAIL CLOSED: missing frozen artifact {path}")
        actual = _sha256(path)
        if actual != frozen[name]:
            raise RuntimeError(
                f"FAIL CLOSED: sha256 mismatch for {name}: {actual} != {frozen[name]}"
            )
        with np.load(path, allow_pickle=True) as archive:
            arrays = {k: np.array(archive[k], copy=True) for k in archive.files}
        cells.append(
            Cell(
                name=name,
                path=path,
                pipeline=str(arrays["pipeline"]),
                split_id=str(arrays["split_id"]),
                train_rep=int(arrays["train_rep"]),
                experiment_id=str(arrays["experiment_id"]),
                audit_draw_seed=int(arrays["audit_draw_seed"]),
                traffic_draw_seed=int(arrays["traffic_draw_seed"]),
                arrays=arrays,
            )
        )
    return cells


def fold_overlap_row(cell: Cell) -> Dict[str, object]:
    """Every pairwise sample_id overlap among prop/cert/eval/traffic (assert 0)."""
    ids = {
        "prop": set(cell.arrays["prop_sample_id"].tolist()),
        "cert": set(cell.arrays["cert_sample_id"].tolist()),
        "eval": set(cell.arrays["eval_sample_id"].tolist()),
        "traffic": set(cell.arrays["traffic_sample_id"].tolist()),
    }
    roles = ["prop", "cert", "eval", "traffic"]
    row: Dict[str, object] = {
        "cell": cell.name,
        "pipeline": cell.pipeline,
        "split_id": cell.split_id,
        "train_rep": cell.train_rep,
    }
    total = 0
    for i, a in enumerate(roles):
        for b in roles[i + 1 :]:
            n = len(ids[a] & ids[b])
            row[f"overlap_{a}_{b}"] = n
            total += n
    row["overlap_total"] = total
    row["disjoint_ok"] = bool(total == 0)
    return row


# --------------------------------------------------------------------------- #
# Certification draw (outcome-independent, common-random-stream with prefixes)
# --------------------------------------------------------------------------- #
@dataclass
class DrawnCounts:
    A: np.ndarray            # accepted (with multiplicity)
    K: np.ndarray            # accepted & error (with multiplicity)
    n: np.ndarray            # n_j (nominal)
    # accounting (per domain)
    reservoir: np.ndarray
    unique_draws: np.ndarray
    max_multiplicity: np.ndarray
    unique_accepted: np.ndarray
    unique_accepted_error: np.ndarray


def draw_positions(cell: Cell) -> Dict[int, np.ndarray]:
    """1000 with-replacement reservoir positions per domain (common stream)."""
    cert_client = cell.arrays["cert_client"]
    rng = np.random.default_rng(cell.audit_draw_seed)
    draws: Dict[int, np.ndarray] = {}
    for j in range(J):
        res = np.where(cert_client == j)[0]
        if res.size == 0:
            raise RuntimeError(f"FAIL CLOSED: empty cert reservoir domain {j} {cell.name}")
        draws[j] = res[rng.integers(0, res.size, size=NJ_MAX)]
    return draws


def counts_from_draw(
    cell: Cell,
    draws: Dict[int, np.ndarray],
    n_j: int,
    score_name: str,
    threshold: float,
    feasible: bool,
) -> DrawnCounts:
    logits = cell.arrays["cert_logits"]
    y_open = cell.arrays["cert_y_open"]
    cert_client = cell.arrays["cert_client"]
    pred = logits.argmax(axis=-1)
    err = open_set_error(pred, y_open)
    score = compute_score(score_name, logits)
    accept_full = feasible & (score >= threshold)

    A = np.zeros(J, dtype=int)
    K = np.zeros(J, dtype=int)
    reservoir = np.zeros(J, dtype=int)
    unique_draws = np.zeros(J, dtype=int)
    max_mult = np.zeros(J, dtype=int)
    unique_acc = np.zeros(J, dtype=int)
    unique_acc_err = np.zeros(J, dtype=int)
    for j in range(J):
        pos = draws[j][:n_j]
        reservoir[j] = int(np.sum(cert_client == j))
        uniq, cnt = np.unique(pos, return_counts=True)
        unique_draws[j] = uniq.size
        max_mult[j] = int(cnt.max()) if cnt.size else 0
        acc = accept_full[pos]
        A[j] = int(acc.sum())
        K[j] = int((acc & err[pos]).sum())
        acc_pos = pos[acc]
        unique_acc[j] = int(np.unique(acc_pos).size)
        acc_err_pos = pos[acc & err[pos]]
        unique_acc_err[j] = int(np.unique(acc_err_pos).size)
    return DrawnCounts(
        A=A, K=K, n=np.full(J, n_j, dtype=int),
        reservoir=reservoir, unique_draws=unique_draws, max_multiplicity=max_mult,
        unique_accepted=unique_acc, unique_accepted_error=unique_acc_err,
    )


@dataclass
class FrontierDiag:
    min_ucb: float = float("inf")
    any_certifies: bool = False
    n_feasible: int = 0
    # realizable proposal-only "most-buffered" alternative selector (min gamma,
    # then max prop coverage): a legitimate alternative pre-registration, reported
    # as an honest sensitivity only -- NOT the frozen primary.
    altbuf_ucb: float = float("nan")
    altbuf_certifies: bool = False
    altbuf_gamma: float = float("nan")
    altbuf_score: str = ""


def frontier_diagnostic(
    cell: Cell, draws: Dict[int, np.ndarray], alpha: float, candidates, budget: FailureBudget
) -> FrontierDiag:
    """Diagnostic (NOT the primary): over ALL proposal-feasible (score,gamma)
    candidate thresholds, evaluate the frozen full-simplex certificate on the SAME
    n_j=500 draw. ``any_certifies`` is an ACHIEVABILITY upper bound (it inspects the
    cert outcome). ``altbuf_*`` is a realizable proposal-only alternative selector
    (most-buffered: smallest gamma, then largest proposal coverage) whose choice
    uses proposal statistics only. This isolates whether the empirical-risk barrier
    is fundamental or an artifact of the frozen coverage-max tie-break."""
    diag = FrontierDiag()
    feas = [c for c in candidates if c.proposal_feasible]
    diag.n_feasible = len(feas)
    for cand in feas:
        dc = counts_from_draw(cell, draws, NJ_PRIMARY, cand.score_name, cand.threshold, True)
        cert = certify_full_simplex(dc, alpha, budget)
        if math.isfinite(cert.risk_ucb):
            diag.min_ucb = min(diag.min_ucb, float(cert.risk_ucb))
        if cert.certified:
            diag.any_certifies = True
    if feas:
        # proposal-only rule: most buffered first (deterministic, outcome-free).
        alt = min(feas, key=lambda c: (c.gamma, -c.prop_coverage, c.prop_risk, c.score_name))
        dc = counts_from_draw(cell, draws, NJ_PRIMARY, alt.score_name, alt.threshold, True)
        cert = certify_full_simplex(dc, alpha, budget)
        diag.altbuf_ucb = float(cert.risk_ucb)
        diag.altbuf_certifies = bool(cert.certified)
        diag.altbuf_gamma = float(alt.gamma)
        diag.altbuf_score = alt.score_name
    return diag


def eval_client_simplex_risk(
    cell: Cell, score_name: str, threshold: float, feasible: bool
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """Realized held-out risk on the FULL eval fold (no draw).

    Returns (client_simplex_risk = max_j r_j over domains with A_j>0, per-domain
    r_j, per-domain eval accepted A, pooled eval risk)."""
    logits = cell.arrays["eval_logits"]
    y_open = cell.arrays["eval_y_open"]
    client = cell.arrays["eval_client"]
    pred = logits.argmax(axis=-1)
    err = open_set_error(pred, y_open)
    score = compute_score(score_name, logits)
    accept = feasible & (score >= threshold)
    r = np.full(J, np.nan)
    Ae = np.zeros(J, dtype=int)
    for j in range(J):
        m = (client == j) & accept
        Ae[j] = int(m.sum())
        if Ae[j] > 0:
            r[j] = float(err[m].mean())
    live = Ae > 0
    csr = float(np.nanmax(r[live])) if np.any(live) else float("nan")
    pooled = float(err[accept].mean()) if np.any(accept) else float("nan")
    return csr, r, Ae, pooled


# --------------------------------------------------------------------------- #
# Certificates (exact fedcore core reuse)
# --------------------------------------------------------------------------- #
def full_simplex_budget() -> FailureBudget:
    return FailureBudget(
        total=DELTA_R + DELTA_C,
        conditional_risk=DELTA_R,
        acceptance_lower=DELTA_C,
    )


def box_budget(with_mixture: bool) -> FailureBudget:
    mixture = T_DELTA_LAMBDA if with_mixture else 0.0
    total = mixture + T_DELTA_R + T_DELTA_C
    return FailureBudget(
        total=total,
        mixture=mixture,
        conditional_risk=T_DELTA_R / 2.0,     # rbar
        acceptance_upper=T_DELTA_R / 2.0,     # ahigh
        acceptance_lower=T_DELTA_C,           # alow (shared coverage/denominator)
    )


def _uniform_eps(dc: DrawnCounts, budget: FailureBudget, bounded: bool):
    alloc = allocate_failure_budget(budget, dc.A, dc.K, policy="uniform")
    risk_eps = alloc["conditional_risk"]
    lower_eps = alloc["acceptance_lower"]
    upper_eps = alloc.get("acceptance_upper") if bounded else None
    return risk_eps, lower_eps, upper_eps


def certify_full_simplex(dc: DrawnCounts, alpha: float, budget: FailureBudget):
    risk_eps, lower_eps, _ = _uniform_eps(dc, budget, bounded=False)
    return joint_conditional_certificate(
        dc.A, dc.K, dc.n, alpha=alpha,
        risk_eps=risk_eps, acceptance_lower_eps=lower_eps,
    )


def certify_box(dc: DrawnCounts, alpha: float, budget: FailureBudget,
                lam_lower: Sequence[float], lam_upper: Sequence[float]):
    risk_eps, lower_eps, upper_eps = _uniform_eps(dc, budget, bounded=True)
    return joint_conditional_certificate(
        dc.A, dc.K, dc.n, alpha=alpha,
        risk_eps=risk_eps, acceptance_lower_eps=lower_eps,
        acceptance_upper_eps=upper_eps,
        lambda_lower=lam_lower, lambda_upper=lam_upper,
    )


def classify_failure(
    proposal_support: bool, A: np.ndarray, count_floor: int, certified: bool, U: float, alpha: float
) -> Tuple[str, bool, bool, bool]:
    """Return (failure_class, cert_feasible(all A>0), count_feasible, nonvacuous).

    failure_class: '' if certified else first barrier hit in the order
    support -> zero-acceptance -> count -> empirical-risk."""
    cert_feasible = bool(np.all(A > 0))
    count_feasible = bool(np.all(A >= count_floor))
    if certified:
        return "", cert_feasible, count_feasible, True
    if not proposal_support:
        return "support", cert_feasible, count_feasible, False
    if not cert_feasible:
        return "zero-acceptance", cert_feasible, count_feasible, False
    if not count_feasible:
        return "count", cert_feasible, count_feasible, False
    return "empirical-risk", cert_feasible, count_feasible, False


# --------------------------------------------------------------------------- #
# Per-cell evaluation
# --------------------------------------------------------------------------- #
@dataclass
class CellAlphaResult:
    cell: Cell
    alpha: float
    n_j: int
    policy: object
    dc: DrawnCounts
    cert: object
    failure_class: str
    cert_feasible: bool
    count_feasible: bool
    nonvacuous: bool
    proposal_support: bool
    eff_cov: float
    cond_cov: float
    eval_csr: float
    eval_pooled: float
    frontier: FrontierDiag = field(default_factory=lambda: FrontierDiag())


def evaluate_cell(cell: Cell) -> Dict[Tuple[float, int], CellAlphaResult]:
    draws = draw_positions(cell)
    fs_budget = full_simplex_budget()
    results: Dict[Tuple[float, int], CellAlphaResult] = {}
    for alpha in ALPHAS:
        policy, cands = select_from_artifact(
            cell.arrays, alpha=alpha, scores=SCORES, gammas=GAMMAS, kappa=KAPPA
        )
        support = bool(policy.feasible)
        count_floor = zero_error_floor(DELTA_R / J, alpha)
        frdiag = frontier_diagnostic(cell, draws, alpha, cands, fs_budget)
        for n_j in NJ_GRID:
            dc = counts_from_draw(
                cell, draws, n_j, policy.score_name if support else "msp",
                policy.threshold, support,
            )
            cert = certify_full_simplex(dc, alpha, fs_budget)
            certified = bool(support and cert.certified)
            fclass, cfeas, countfeas, nonvac = classify_failure(
                support, dc.A, count_floor, certified, cert.risk_ucb, alpha
            )
            csr, _r, _Ae, pooled = (
                eval_client_simplex_risk(cell, policy.score_name, policy.threshold, support)
                if support else (float("nan"), np.full(J, np.nan), np.zeros(J, int), float("nan"))
            )
            eff = float(cert.coverage_lcb) if certified else 0.0
            cond = float(cert.coverage_lcb) if certified else float("nan")
            results[(alpha, n_j)] = CellAlphaResult(
                cell=cell, alpha=alpha, n_j=n_j, policy=policy, dc=dc, cert=cert,
                failure_class=fclass, cert_feasible=cfeas, count_feasible=countfeas,
                nonvacuous=nonvac, proposal_support=support,
                eff_cov=eff, cond_cov=cond, eval_csr=csr, eval_pooled=pooled,
                frontier=frdiag,
            )
    return results


# --------------------------------------------------------------------------- #
# Traffic-derived Lambda_hat + fixed-rho
# --------------------------------------------------------------------------- #
def traffic_rows(cell: Cell, cell_results: Dict[Tuple[float, int], CellAlphaResult]) -> List[dict]:
    draws = draw_positions(cell)
    tbudget = box_budget(with_mixture=True)
    rows = []
    for alpha in ALPHAS:
        res = cell_results[(alpha, NJ_PRIMARY)]
        policy = res.policy
        support = res.proposal_support
        dc = res.dc  # primary n_j=500 draw, identical selector+draw
        fs = res.cert
        fs_ucb = float(fs.risk_ucb)
        fs_cert = bool(res.nonvacuous)
        for m in M_GRID:
            counts = draw_traffic_client_counts(
                cell.arrays["traffic_client"], m, seed=cell.traffic_draw_seed, n_clients=J
            )
            tl = build_traffic_lambda(counts, delta_lambda=T_DELTA_LAMBDA)
            lo = tl.box.mixture.lower
            hi = tl.box.mixture.upper
            worst = None
            if support:
                cert = certify_box(dc, alpha, tbudget, lo, hi)
                t_ucb = float(cert.risk_ucb)
                t_cov = float(cert.coverage_lcb)
                t_cert = bool(cert.certified)
                # matched-budget full simplex (same rbar as the box) isolates the set effect
                matched = float(np.max(cert.rbar))
                worst = cert.lambda_upper.tolist()
            else:
                t_ucb = float("inf"); t_cov = 0.0; t_cert = False; matched = float("inf")
            rows.append({
                "cell": cell.name, "pipeline": cell.pipeline, "split_id": cell.split_id,
                "train_rep": cell.train_rep, "alpha": alpha, "m": m,
                "primary_m": bool(m == M_PRIMARY),
                "traffic_counts": counts.tolist(),
                "box_lower": [float(x) for x in lo], "box_upper": [float(x) for x in hi],
                "raw_lower": [float(x) for x in tl.box.raw_lower],
                "raw_upper": [float(x) for x in tl.box.raw_upper],
                "box_total_width": float(np.sum(tl.box.raw_upper - tl.box.raw_lower)),
                "proposal_support": support,
                "traffic_risk_ucb": t_ucb, "traffic_coverage_lcb": t_cov,
                "traffic_certified": t_cert,
                "simplex_matched_ucb": matched,
                "gain_from_set_ucb": (matched - t_ucb) if math.isfinite(matched) and math.isfinite(t_ucb) else float("nan"),
                "full_simplex_primary_ucb": fs_ucb,
                "full_simplex_primary_certified": fs_cert,
                "deploy_gain_vs_full_simplex": bool(t_cert and not fs_cert),
                "worst_case_mixture_upper": worst,
            })
    return rows


def rho_rows(cell: Cell, cell_results: Dict[Tuple[float, int], CellAlphaResult]) -> List[dict]:
    rbudget = box_budget(with_mixture=False)
    center = np.full(J, 1.0 / J)
    rows = []
    for alpha in ALPHAS:
        res = cell_results[(alpha, NJ_PRIMARY)]
        support = res.proposal_support
        dc = res.dc
        for rho in RHO_GRID:
            box = rho_mixture_box(center, rho)
            if support:
                cert = certify_box(dc, alpha, rbudget, box.lower, box.upper)
                ucb = float(cert.risk_ucb); cov = float(cert.coverage_lcb)
                certd = bool(cert.certified)
            else:
                ucb = float("inf"); cov = 0.0; certd = False
            rows.append({
                "cell": cell.name, "pipeline": cell.pipeline, "split_id": cell.split_id,
                "train_rep": cell.train_rep, "alpha": alpha, "rho": rho,
                "box_lower": [float(x) for x in box.lower],
                "box_upper": [float(x) for x in box.upper],
                "proposal_support": support,
                "rho_risk_ucb": ucb, "rho_coverage_lcb": cov, "rho_certified": certd,
                "full_simplex_primary_ucb": float(res.cert.risk_ucb),
                "full_simplex_primary_certified": bool(res.nonvacuous),
            })
    return rows


# --------------------------------------------------------------------------- #
# CSV helpers
# --------------------------------------------------------------------------- #
def _j(value) -> str:
    return json.dumps(value, separators=(",", ":"))


def write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        ).stdout.strip()
    except Exception:
        return "UNAVAILABLE"


# --------------------------------------------------------------------------- #
# Paired A-vs-B contrasts
# --------------------------------------------------------------------------- #
def paired_contrasts(all_results: Dict[str, Dict[Tuple[float, int], CellAlphaResult]]) -> List[dict]:
    """25 paired (split_id, train_rep) blocks: A minus B, per alpha, n_j=500."""
    # index by (split_id, train_rep, pipeline)
    idx: Dict[Tuple[str, int, str], Dict[Tuple[float, int], CellAlphaResult]] = {}
    for name, res in all_results.items():
        any_r = next(iter(res.values()))
        idx[(any_r.cell.split_id, any_r.cell.train_rep, any_r.cell.pipeline)] = res
    blocks = sorted({(s, r) for (s, r, _p) in idx})
    if len(blocks) != 25:
        raise RuntimeError(f"expected 25 paired blocks, found {len(blocks)}")
    clusters = sorted({s for (s, _r) in blocks})  # 5 class splits
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metrics = ["eff_cov", "risk_ucb", "cov_lcb", "deploy"]
    rows: List[dict] = []
    for alpha in ALPHAS:
        diffs: Dict[str, np.ndarray] = {mm: np.zeros(len(blocks)) for mm in metrics}
        block_cluster = np.array([clusters.index(s) for (s, _r) in blocks])
        for bi, (s, r) in enumerate(blocks):
            ra = idx[(s, r, "A")][(alpha, NJ_PRIMARY)]
            rb = idx[(s, r, "B")][(alpha, NJ_PRIMARY)]
            diffs["eff_cov"][bi] = ra.eff_cov - rb.eff_cov
            diffs["risk_ucb"][bi] = (
                (ra.cert.risk_ucb if math.isfinite(ra.cert.risk_ucb) else 1.0)
                - (rb.cert.risk_ucb if math.isfinite(rb.cert.risk_ucb) else 1.0)
            )
            diffs["cov_lcb"][bi] = ra.cert.coverage_lcb - rb.cert.coverage_lcb
            diffs["deploy"][bi] = float(ra.nonvacuous) - float(rb.nonvacuous)
        for mm in metrics:
            d = diffs[mm]
            mean = float(d.mean())
            # cluster bootstrap over the 5 class splits
            boot = np.empty(N_BOOTSTRAP)
            for b in range(N_BOOTSTRAP):
                pick = rng.integers(0, len(clusters), size=len(clusters))
                mask = np.concatenate([np.where(block_cluster == c)[0] for c in pick])
                boot[b] = d[mask].mean()
            lo, hi = np.percentile(boot, [2.5, 97.5])
            # exact sign test (two-sided binomial on nonzero signs)
            n_pos = int(np.sum(d > 0)); n_neg = int(np.sum(d < 0))
            n_eff = n_pos + n_neg
            if n_eff == 0:
                sign_p = 1.0
            else:
                k = min(n_pos, n_neg)
                sign_p = float(min(1.0, 2.0 * _sp_binom.cdf(k, n_eff, 0.5)))
            # exact clustered sign-flip permutation (2^5 = 32 patterns)
            obs = abs(mean)
            ge = 0
            for pattern in range(1 << len(clusters)):
                signs = np.array([1.0 if (pattern >> c) & 1 else -1.0 for c in range(len(clusters))])
                flipped = float((d * signs[block_cluster]).mean())
                if abs(flipped) >= obs - 1e-12:
                    ge += 1
            perm_p = ge / float(1 << len(clusters))
            rows.append({
                "alpha": alpha, "metric": mm, "n_blocks": len(blocks),
                "n_clusters": len(clusters), "mean_diff_A_minus_B": mean,
                "boot_ci_lo": float(lo), "boot_ci_hi": float(hi),
                "n_pos": n_pos, "n_neg": n_neg, "n_zero": int(np.sum(d == 0)),
                "exact_sign_p": sign_p, "clustered_perm_p": perm_p,
                "bootstrap_reps": N_BOOTSTRAP,
            })
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    frozen = load_frozen_checksums()
    cells = load_cells(frozen)
    print(f"[integrity] 50/50 frozen logit sha256 verified; numpy {np.__version__}")

    # fold overlap (assert 0)
    fold_rows = [fold_overlap_row(c) for c in cells]
    if any(not r["disjoint_ok"] for r in fold_rows):
        raise RuntimeError("FAIL CLOSED: non-disjoint fold detected")

    all_results: Dict[str, Dict[Tuple[float, int], CellAlphaResult]] = {}
    for c in cells:
        all_results[c.name] = evaluate_cell(c)
    print("[eval] 50 cells evaluated over 5 alphas x 3 n_j")

    # ----- final_cell_results.csv (primary n_j=500, all alphas) -----------
    cell_fields = [
        "cell", "pipeline", "split_id", "train_rep", "experiment_id", "alpha", "n_j",
        "score_name", "gamma", "threshold", "prop_coverage", "prop_risk",
        "proposal_support", "proposal_infeasible",
        "certified", "cert_risk_ucb", "cert_coverage_lcb",
        "EffectiveCertCov", "CondCertCov",
        "empirical_client_simplex_risk", "empirical_pooled_eval_risk",
        "failure_class", "certification_feasible", "count_feasible", "actual_nonvacuous",
        "per_client_A", "per_client_K", "per_client_n", "per_client_rbar",
        "count_floor", "cert_reason",
    ]
    cell_rows = []
    for c in cells:
        for alpha in ALPHAS:
            r = all_results[c.name][(alpha, NJ_PRIMARY)]
            p = r.policy
            cell_rows.append({
                "cell": c.name, "pipeline": c.pipeline, "split_id": c.split_id,
                "train_rep": c.train_rep, "experiment_id": c.experiment_id,
                "alpha": alpha, "n_j": NJ_PRIMARY,
                "score_name": p.score_name, "gamma": p.gamma, "threshold": p.threshold,
                "prop_coverage": p.prop_coverage, "prop_risk": p.prop_risk,
                "proposal_support": r.proposal_support,
                "proposal_infeasible": (not r.proposal_support),
                "certified": r.nonvacuous,
                "cert_risk_ucb": float(r.cert.risk_ucb),
                "cert_coverage_lcb": float(r.cert.coverage_lcb),
                "EffectiveCertCov": r.eff_cov, "CondCertCov": r.cond_cov,
                "empirical_client_simplex_risk": r.eval_csr,
                "empirical_pooled_eval_risk": r.eval_pooled,
                "failure_class": r.failure_class,
                "certification_feasible": r.cert_feasible,
                "count_feasible": r.count_feasible,
                "actual_nonvacuous": r.nonvacuous,
                "per_client_A": _j(r.dc.A.tolist()), "per_client_K": _j(r.dc.K.tolist()),
                "per_client_n": _j(r.dc.n.tolist()),
                "per_client_rbar": _j([float(x) for x in r.cert.rbar]),
                "count_floor": zero_error_floor(DELTA_R / J, alpha),
                "cert_reason": r.cert.reason,
            })
    write_csv(os.path.join(OUTDIR, "final_cell_results.csv"), cell_rows, cell_fields)

    # ----- full_simplex_summary.csv (per alpha x n_j x pipeline) ----------
    sum_fields = [
        "alpha", "n_j", "pipeline", "primary_alpha", "primary_n_j", "n_cells",
        "n_proposal_infeasible", "n_zero_acceptance", "n_count_barrier",
        "n_empirical_risk_barrier", "n_certified_nonvacuous",
        "mean_EffectiveCertCov", "mean_CondCertCov_deployed", "median_cert_risk_ucb",
        "mean_empirical_client_simplex_risk",
    ]
    sum_rows = []
    for alpha in ALPHAS:
        for n_j in NJ_GRID:
            for pipe in ("A", "B", "all"):
                sel = [
                    all_results[c.name][(alpha, n_j)] for c in cells
                    if pipe == "all" or c.pipeline == pipe
                ]
                fclass = [r.failure_class for r in sel]
                cond = [r.cert.coverage_lcb for r in sel if r.nonvacuous]
                risks = [r.cert.risk_ucb for r in sel if math.isfinite(r.cert.risk_ucb)]
                csr = [r.eval_csr for r in sel if math.isfinite(r.eval_csr)]
                sum_rows.append({
                    "alpha": alpha, "n_j": n_j, "pipeline": pipe,
                    "primary_alpha": bool(alpha == ALPHA_PRIMARY),
                    "primary_n_j": bool(n_j == NJ_PRIMARY),
                    "n_cells": len(sel),
                    "n_proposal_infeasible": sum(1 for f in fclass if f == "support"),
                    "n_zero_acceptance": sum(1 for f in fclass if f == "zero-acceptance"),
                    "n_count_barrier": sum(1 for f in fclass if f == "count"),
                    "n_empirical_risk_barrier": sum(1 for f in fclass if f == "empirical-risk"),
                    "n_certified_nonvacuous": sum(1 for r in sel if r.nonvacuous),
                    "mean_EffectiveCertCov": float(np.mean([r.eff_cov for r in sel])),
                    "mean_CondCertCov_deployed": float(np.mean(cond)) if cond else float("nan"),
                    "median_cert_risk_ucb": float(np.median(risks)) if risks else float("nan"),
                    "mean_empirical_client_simplex_risk": float(np.mean(csr)) if csr else float("nan"),
                })
    write_csv(os.path.join(OUTDIR, "full_simplex_summary.csv"), sum_rows, sum_fields)

    # ----- failure_cascade.csv (per cell x alpha, n_j=500) ----------------
    casc_fields = [
        "cell", "pipeline", "split_id", "train_rep", "alpha", "n_j",
        "proposal_support", "certification_feasible", "count_feasible", "actual_nonvacuous",
        "failure_class", "min_A", "count_floor", "n_zero_accept_domains",
        "cert_risk_ucb", "cert_coverage_lcb",
        "frontier_min_ucb_diag", "frontier_any_certifies_diag", "n_feasible_candidates_diag",
        "altbuf_ucb_diag", "altbuf_certifies_diag", "altbuf_gamma_diag", "altbuf_score_diag",
    ]
    casc_rows = []
    for c in cells:
        for alpha in ALPHAS:
            r = all_results[c.name][(alpha, NJ_PRIMARY)]
            casc_rows.append({
                "cell": c.name, "pipeline": c.pipeline, "split_id": c.split_id,
                "train_rep": c.train_rep, "alpha": alpha, "n_j": NJ_PRIMARY,
                "proposal_support": r.proposal_support,
                "certification_feasible": r.cert_feasible,
                "count_feasible": r.count_feasible,
                "actual_nonvacuous": r.nonvacuous,
                "failure_class": r.failure_class,
                "min_A": int(r.dc.A.min()),
                "count_floor": zero_error_floor(DELTA_R / J, alpha),
                "n_zero_accept_domains": int(np.sum(r.dc.A == 0)),
                "cert_risk_ucb": float(r.cert.risk_ucb),
                "cert_coverage_lcb": float(r.cert.coverage_lcb),
                "frontier_min_ucb_diag": r.frontier.min_ucb,
                "frontier_any_certifies_diag": r.frontier.any_certifies,
                "n_feasible_candidates_diag": r.frontier.n_feasible,
                "altbuf_ucb_diag": r.frontier.altbuf_ucb,
                "altbuf_certifies_diag": r.frontier.altbuf_certifies,
                "altbuf_gamma_diag": r.frontier.altbuf_gamma,
                "altbuf_score_diag": r.frontier.altbuf_score,
            })
    write_csv(os.path.join(OUTDIR, "failure_cascade.csv"), casc_rows, casc_fields)

    # ----- reservoir_accounting.csv (per cell x domain, n_j=500) ----------
    res_fields = [
        "cell", "pipeline", "split_id", "train_rep", "domain", "domain_name",
        "reservoir_size", "nominal_draw_count", "unique_draws", "duplication_rate",
        "max_multiplicity", "accepted_count", "accepted_errors", "unique_accepted_evidence",
        "unique_accepted_error_evidence", "evidence_inflation_ratio",
        "fold_overlap_total", "trusted_labels_reservoir",
    ]
    res_rows = []
    for c in cells:
        overlap_total = fold_overlap_row(c)["overlap_total"]
        # use primary alpha selector for the accepted counts (predeclared primary)
        r = all_results[c.name][(ALPHA_PRIMARY, NJ_PRIMARY)]
        dnames = [str(x) for x in c.arrays["domains"]]
        for j in range(J):
            nominal = NJ_PRIMARY
            uniq = int(r.dc.unique_draws[j])
            res_rows.append({
                "cell": c.name, "pipeline": c.pipeline, "split_id": c.split_id,
                "train_rep": c.train_rep, "domain": j, "domain_name": dnames[j],
                "reservoir_size": int(r.dc.reservoir[j]),
                "nominal_draw_count": nominal,
                "unique_draws": uniq,
                "duplication_rate": float(1.0 - uniq / nominal),
                "max_multiplicity": int(r.dc.max_multiplicity[j]),
                "accepted_count": int(r.dc.A[j]),
                "accepted_errors": int(r.dc.K[j]),
                "unique_accepted_evidence": int(r.dc.unique_accepted[j]),
                "unique_accepted_error_evidence": int(r.dc.unique_accepted_error[j]),
                "evidence_inflation_ratio": (
                    float(r.dc.A[j] / r.dc.unique_accepted[j]) if r.dc.unique_accepted[j] > 0 else float("nan")
                ),
                "fold_overlap_total": int(overlap_total),
                "trusted_labels_reservoir": int(r.dc.reservoir[j]),
            })
    write_csv(os.path.join(OUTDIR, "reservoir_accounting.csv"), res_rows, res_fields)

    # ----- fold_overlap.csv -----------------------------------------------
    fo_fields = [
        "cell", "pipeline", "split_id", "train_rep",
        "overlap_prop_cert", "overlap_prop_eval", "overlap_prop_traffic",
        "overlap_cert_eval", "overlap_cert_traffic", "overlap_eval_traffic",
        "overlap_total", "disjoint_ok",
    ]
    write_csv(os.path.join(OUTDIR, "fold_overlap.csv"), fold_rows, fo_fields)

    # ----- data_derived_lambda_summary.csv --------------------------------
    tr_rows: List[dict] = []
    for c in cells:
        tr_rows.extend(traffic_rows(c, all_results[c.name]))
    tr_fields = [
        "cell", "pipeline", "split_id", "train_rep", "alpha", "m", "primary_m",
        "traffic_counts", "box_lower", "box_upper", "raw_lower", "raw_upper",
        "box_total_width", "proposal_support", "traffic_risk_ucb", "traffic_coverage_lcb",
        "traffic_certified", "simplex_matched_ucb", "gain_from_set_ucb",
        "full_simplex_primary_ucb", "full_simplex_primary_certified",
        "deploy_gain_vs_full_simplex", "worst_case_mixture_upper",
    ]
    # json-encode list columns
    for row in tr_rows:
        for key in ("traffic_counts", "box_lower", "box_upper", "raw_lower", "raw_upper", "worst_case_mixture_upper"):
            row[key] = _j(row[key])
    write_csv(os.path.join(OUTDIR, "data_derived_lambda_summary.csv"), tr_rows, tr_fields)

    # ----- rho_sensitivity.csv --------------------------------------------
    rr_rows: List[dict] = []
    for c in cells:
        rr_rows.extend(rho_rows(c, all_results[c.name]))
    for row in rr_rows:
        for key in ("box_lower", "box_upper"):
            row[key] = _j(row[key])
    rr_fields = [
        "cell", "pipeline", "split_id", "train_rep", "alpha", "rho",
        "box_lower", "box_upper", "proposal_support",
        "rho_risk_ucb", "rho_coverage_lcb", "rho_certified",
        "full_simplex_primary_ucb", "full_simplex_primary_certified",
    ]
    write_csv(os.path.join(OUTDIR, "rho_sensitivity.csv"), rr_rows, rr_fields)

    # ----- paired_pipeline_contrasts.csv ----------------------------------
    pc_rows = paired_contrasts(all_results)
    pc_fields = [
        "alpha", "metric", "n_blocks", "n_clusters", "mean_diff_A_minus_B",
        "boot_ci_lo", "boot_ci_hi", "n_pos", "n_neg", "n_zero",
        "exact_sign_p", "clustered_perm_p", "bootstrap_reps",
    ]
    write_csv(os.path.join(OUTDIR, "paired_pipeline_contrasts.csv"), pc_rows, pc_fields)

    # ----- FINAL_MANIFEST.json + FINAL_CHECKSUMS.sha256 -------------------
    outputs = [
        "final_cell_results.csv", "full_simplex_summary.csv",
        "data_derived_lambda_summary.csv", "rho_sensitivity.csv",
        "paired_pipeline_contrasts.csv", "failure_cascade.csv",
        "reservoir_accounting.csv", "fold_overlap.csv",
    ]
    checks = {name: _sha256(os.path.join(OUTDIR, name)) for name in outputs}
    manifest = {
        "artifact_type": "fedcore.officehome.final_posthoc_analysis",
        "generated_by": "fedcore.experiments.officehome_final_analysis",
        "git_commit": git_commit(),
        "numpy_version": np.__version__,
        "n_cells": len(cells),
        "J": J,
        "alphas": list(ALPHAS), "alpha_primary": ALPHA_PRIMARY,
        "n_j_grid": list(NJ_GRID), "n_j_primary": NJ_PRIMARY,
        "full_simplex_budget": {"delta_r": DELTA_R, "delta_c": DELTA_C, "allocation": "uniform"},
        "traffic_budget": {"delta_lambda": T_DELTA_LAMBDA, "delta_r": T_DELTA_R, "delta_c": T_DELTA_C,
                           "split": "delta_r evenly over rbar+ahigh; alow=delta_c shared"},
        "rho_grid": list(RHO_GRID), "m_grid": list(M_GRID), "m_primary": M_PRIMARY,
        "scores": list(SCORES), "gammas": list(GAMMAS), "kappa": KAPPA,
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_reps": N_BOOTSTRAP,
        "frozen_logits_checksums": CHECKSUMS,
        "output_sha256": checks,
        "certificate_core": "fedcore.certificate.joint.joint_conditional_certificate (no CP reimplemented)",
    }
    with open(os.path.join(OUTDIR, "FINAL_MANIFEST.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(os.path.join(OUTDIR, "FINAL_CHECKSUMS.sha256"), "w", encoding="utf-8") as handle:
        for name in outputs + ["FINAL_MANIFEST.json"]:
            handle.write(f"{_sha256(os.path.join(OUTDIR, name))}  {name}\n")

    # ----- console headline ----------------------------------------------
    print("\n=== PRIMARY HEADLINE: client-level full-simplex certificate (frozen coverage-max selector) ===")
    for alpha in ALPHAS:
        sel = [all_results[c.name][(alpha, NJ_PRIMARY)] for c in cells]
        nA = sum(1 for r in sel if r.nonvacuous and r.cell.pipeline == "A")
        nB = sum(1 for r in sel if r.nonvacuous and r.cell.pipeline == "B")
        tag = " <== PRIMARY" if alpha == ALPHA_PRIMARY else ""
        print(f"  alpha={alpha:.2f}: non-vacuous {sum(1 for r in sel if r.nonvacuous):2d}/50 "
              f"(A {nA}/25, B {nB}/25){tag}")
    print("\n=== DIAGNOSTIC (NOT primary): achievability + proposal-only most-buffered alt selector ===")
    for alpha in ALPHAS:
        sel = [all_results[c.name][(alpha, NJ_PRIMARY)] for c in cells]
        anyc = sum(1 for r in sel if r.frontier.any_certifies)
        altc = sum(1 for r in sel if r.frontier.altbuf_certifies)
        print(f"  alpha={alpha:.2f}: any-frozen-grid-candidate certifies {anyc:2d}/50 | "
              f"most-buffered proposal-only alt certifies {altc:2d}/50")
    print("\nOutputs written to", OUTDIR)


if __name__ == "__main__":
    main()
