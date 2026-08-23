"""F3b stage 1 -- fidelity gate + embedding score extraction (Fed-ISIC, NO training).

Runs inside the authorized root Docker container (repo mounted READ-ONLY at
/workspace, output at /out).  Two stages, invoked separately so the CHEAP fidelity
gate always runs and is inspected before any embedding work:

  --stage gate
      Re-infer prop/cert/test class logits for every reproducible cell (46) from the
      canonical root-owned checkpoint using fedcore's EXACT eval pipeline
      (load_fed_isic_job + eval_transform + MedicalImageDataset + the same head as
      export_logits) and compare to the canonical (mode-0600, root-only) *_logits
      npz.  Writes f3b_reinference_fidelity.csv.  Fails closed (exit 1) if ANY cell
      violates: identical source-id order, exact argmax match, max|logit diff|<=1e-4,
      matching held-out task.

  --stage embed
      ONLY run after the gate passes.  For every cell, extract penultimate 1280-d
      efficientnet-b0 embeddings (extract_features -> avg-pool -> flatten; the _fc
      input) for the TRAIN (known-class reference), PROPOSAL and CERTIFICATION folds,
      aggregated per unit as mean-image-embedding (matching mean-image-logits-v1).
      Builds the four embedding scores from TRAIN known-class references ONLY
      (cos_proto, cos_margin, neg_maha, neg_knn; higher=more-known=accept), scores
      the prop/cert units, and caches per-unit scores + reproduced logits + labels
      to /out/_emb/<cell>.npz for the pure-numpy certification stage (Script 2).

No canonical input is ever modified; no fedcore module is edited.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

REPO = os.environ.get("F3B_REPO", "/workspace")
OUT = os.environ.get("F3B_OUT", "/out")
EMB_DIR = os.path.join(OUT, "_emb")

LOGIT_GLOB = os.path.join(REPO, "runs/oneshot/full/logits/*isic*.npz")
CKPT_DIR = os.path.join(REPO, "runs/oneshot/full/ckpt")
RAW_DIR = os.path.join(REPO, "results/raw")
METADATA_CSV = os.path.join(REPO, "results/source_data/fed_isic2019_metadata.csv")
SOURCE_DIR = os.path.join(REPO, "results/source_data")
IMAGE_ROOT = os.path.join(REPO, "data/isic2019/ISIC_2019_Training_Input_preprocessed")

BACKBONE = "efficientnet-b0"
IMAGE_SIZE = 200
BATCH = 128
NUM_WORKERS = 8
LOGIT_TOL = 1e-4
KNN_K = 5
MAHA_RIDGE = 1e-3
EMB_SCORES = ("cos_proto", "cos_margin", "neg_maha", "neg_knn")

OUTPUT_FOLD = {"proposal": "prop", "certification": "cert", "test": "test"}


# --------------------------------------------------------------------------- #
# Cell manifest: map every reproducible logit cell to its held-out set + folds.
# --------------------------------------------------------------------------- #
def cell_manifest():
    cells = []
    for path in sorted(glob.glob(LOGIT_GLOB)):
        cell = os.path.basename(path)[:-4]
        raw = os.path.join(RAW_DIR, cell, "run_0.json")
        cmd = json.load(open(raw))["command"]
        heldout = re.search(r"--held-out-diagnoses\s+(\S+)", cmd).group(1)
        folds = re.search(r"--folds-csv\s+(\S+)", cmd).group(1)
        ckpt = os.path.join(CKPT_DIR, cell + ".pt")
        cells.append(
            dict(
                cell=cell,
                logit_npz=path,
                heldout=heldout,
                folds_csv=os.path.join(REPO, folds),
                ckpt=ckpt,
            )
        )
    return cells


def load_job(entry):
    from fedcore.medical.data import MedicalDataConfig, load_fed_isic_job

    cfg = MedicalDataConfig(
        metadata_csv=METADATA_CSV,
        folds_csv=entry["folds_csv"],
        center_col="center",
        diagnosis_col="diagnosis",
        patient_col="patient_id",
        lesion_col="lesion_id",
        image_col="image_id",
        unit_col="lesion_id",
        image_path_col=None,
        image_root=IMAGE_ROOT,
        image_extension=".jpg",
        dataset_name="fed-isic2019",
    )
    return load_fed_isic_job(cfg, entry["heldout"], check_image_files=True)


def build_model(job, ckpt_path, device):
    import torch

    from fedcore.models.models import make_model

    # pretrained=False: from_name gives the IDENTICAL architecture/keys as the
    # canonical pretrained=True path; the checkpoint state_dict fully overwrites all
    # weights AND BatchNorm running buffers, so the final model is numerically
    # identical while avoiding a network download.
    model = make_model(job.n_known, backbone=BACKBONE, norm="bn", pretrained=False)
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(ckpt_path, map_location="cpu")
    meta = dict(payload.get("metadata", {}))
    rc = int(payload.get("round_completed", -1))
    rr = int(payload.get("rounds_requested", -1))
    if rc != rr - 1:
        raise RuntimeError(f"non-final checkpoint round_completed={rc} rounds={rr}")
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, meta, (rc, rr)


def forward_logits_embeddings(model, dataset, device, want_logits=True):
    """One eval pass. Returns (image_logits, image_embeddings).

    A forward pre-hook on ``_fc`` captures its input, i.e. the 1280-d penultimate
    embedding (avg-pool -> flatten -> dropout, dropout being identity in eval), while
    ``model(x)`` returns exactly the canonical logits (the hook never perturbs the
    forward).  Deterministic transform => worker count / batch size do not change
    values (BatchNorm uses fixed running stats in eval).
    """
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS
    )
    embs = []
    logits = []

    def pre_hook(_module, inp):
        embs.append(inp[0].detach().float().cpu().numpy())

    handle = model._fc.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            for xb, _ in loader:
                out = model(xb.to(device))
                if want_logits:
                    logits.append(out.detach().float().cpu().numpy())
    finally:
        handle.remove()
    emb = np.concatenate(embs, axis=0) if embs else np.zeros((0, 1280), np.float32)
    lg = (
        np.concatenate(logits, axis=0)
        if logits
        else np.zeros((0, model._fc.out_features), np.float32)
    )
    return lg, emb


def reproduce_fold(job, model, device, source_fold, want_embed=False):
    """Reproduce canonical per-unit arrays for one audit fold; optional unit embed."""
    from fedcore.medical.data import (
        MedicalImageDataset,
        aggregate_unit_logits,
        audit_artifact_arrays,
        flatten_unit_images,
    )
    from fedcore.medical.flamby import eval_transform

    units = job.fold_units(source_fold)
    images, parent_ids = flatten_unit_images(units)
    ds = MedicalImageDataset(images, transform=eval_transform(IMAGE_SIZE))
    img_logits, img_emb = forward_logits_embeddings(model, ds, device, want_logits=True)
    arrays = audit_artifact_arrays(units, img_logits, job.heldout_diagnoses)
    unit_emb = None
    if want_embed:
        order = [u.sample_id for u in units]
        unit_emb, _mult = aggregate_unit_logits(img_emb, parent_ids, order)
    return arrays, unit_emb


# --------------------------------------------------------------------------- #
# Stage: fidelity gate
# --------------------------------------------------------------------------- #
def stage_gate():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cells = cell_manifest()
    rows = []
    all_pass = True
    for i, entry in enumerate(cells):
        cell = entry["cell"]
        canon = np.load(entry["logit_npz"], allow_pickle=True)
        job = load_job(entry)
        model, meta, (rc, rr) = build_model(job, entry["ckpt"], device)
        job_heldout = set(job.heldout_diagnoses)
        canon_heldout = set(canon["heldout_diagnoses"].tolist())
        heldout_match = job_heldout == canon_heldout

        max_diff = 0.0
        argmax_hits = 0
        n_units = 0
        order_match = True
        for source_fold, out in OUTPUT_FOLD.items():
            arrays, _ = reproduce_fold(job, model, device, source_fold, want_embed=False)
            rep_logits = np.asarray(arrays["logits"], dtype=np.float64)
            can_logits = np.asarray(canon[f"{out}_logits"], dtype=np.float64)
            rep_ids = np.asarray(arrays["sample_id"], dtype=str)
            can_ids = np.asarray(canon[f"{out}_sample_id"], dtype=str)
            if rep_ids.shape != can_ids.shape or not np.array_equal(rep_ids, can_ids):
                order_match = False
                # align defensively for diff reporting if same set
            if rep_logits.shape == can_logits.shape:
                d = float(np.max(np.abs(rep_logits - can_logits)))
                max_diff = max(max_diff, d)
                argmax_hits += int(
                    np.sum(rep_logits.argmax(1) == can_logits.argmax(1))
                )
                n_units += rep_logits.shape[0]
            else:
                order_match = False
                n_units += rep_logits.shape[0]
        argmax_frac = (argmax_hits / n_units) if n_units else 0.0
        cell_pass = bool(
            order_match
            and heldout_match
            and argmax_frac == 1.0
            and max_diff <= LOGIT_TOL
        )
        all_pass = all_pass and cell_pass
        rows.append(
            dict(
                cell=cell,
                n_units=n_units,
                max_abs_logit_diff=f"{max_diff:.3e}",
                argmax_match_frac=f"{argmax_frac:.6f}",
                source_id_order_match=order_match,
                heldout_task_match=heldout_match,
                round_completed=rc,
                rounds_requested=rr,
                pass_=cell_pass,
            )
        )
        print(
            f"[gate {i+1:2d}/46] {cell}  n={n_units}  maxdiff={max_diff:.2e}  "
            f"argmax={argmax_frac:.4f}  order={order_match}  ho={heldout_match}  "
            f"pass={cell_pass}",
            flush=True,
        )
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    os.makedirs(OUT, exist_ok=True)
    fpath = os.path.join(OUT, "f3b_reinference_fidelity.csv")
    with open(fpath, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "cell",
                "n_units",
                "max_abs_logit_diff",
                "argmax_match_frac",
                "source_id_order_match",
                "heldout_task_match",
                "round_completed",
                "rounds_requested",
                "pass",
            ],
        )
        w.writeheader()
        for r in rows:
            row = dict(r)
            row["pass"] = row.pop("pass_")
            w.writerow(row)
    overall_maxdiff = max(float(r["max_abs_logit_diff"]) for r in rows)
    print(f"\n=== GATE SUMMARY ===  cells={len(rows)}  all_pass={all_pass}  "
          f"overall_max_abs_logit_diff={overall_maxdiff:.3e}", flush=True)
    if not all_pass:
        failed = [r["cell"] for r in rows if not r["pass_"]]
        print(f"GATE FAILED for {len(failed)} cells: {failed[:5]}", flush=True)
        with open(os.path.join(OUT, "f3b_GATE_FAILED.txt"), "w") as f:
            f.write("FAILED_CLOSED_REINFERENCE_FIDELITY\n")
            f.write(f"failed_cells={failed}\n")
        sys.exit(1)
    print("GATE PASSED: all 46/46 reproducible cells reconciled.", flush=True)


# --------------------------------------------------------------------------- #
# Stage: embedding score extraction
# --------------------------------------------------------------------------- #
def _pooled_within_cov(emb, label, n_known):
    D = emb.shape[1]
    cov = np.zeros((D, D), dtype=np.float64)
    dof = 0
    for c in range(n_known):
        Xc = emb[label == c]
        if len(Xc) < 2:
            continue
        dc = Xc - Xc.mean(0, keepdims=True)
        cov += dc.T @ dc
        dof += len(Xc) - 1
    cov /= max(dof, 1)
    ridge = MAHA_RIDGE * (np.trace(cov) / D)
    cov += ridge * np.eye(D)
    return np.linalg.inv(cov)


def _embedding_scores(train_emb, train_label, n_known, query_emb):
    """Four higher=more-known scores from TRAIN known-class references only."""
    train_emb = np.asarray(train_emb, dtype=np.float64)
    query_emb = np.asarray(query_emb, dtype=np.float64)
    protos = np.stack(
        [train_emb[train_label == c].mean(0) for c in range(n_known)]
    )  # (C, D)

    qn = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-12)
    pn = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12)
    cos = qn @ pn.T  # (Nq, C)
    cos_proto = cos.max(1)
    part = np.sort(cos, axis=1)
    cos_margin = part[:, -1] - part[:, -2]

    invc = _pooled_within_cov(train_emb, train_label, n_known)
    diffs = query_emb[:, None, :] - protos[None, :, :]  # (Nq, C, D)
    md2 = np.einsum("qcd,de,qce->qc", diffs, invc, diffs)
    neg_maha = -np.sqrt(np.clip(md2.min(1), 0.0, None))

    # brute-force euclidean kNN (D high => KD-tree is not helpful)
    tb2 = np.sum(train_emb**2, axis=1)  # (Ntr,)
    qb2 = np.sum(query_emb**2, axis=1)  # (Nq,)
    d2 = qb2[:, None] + tb2[None, :] - 2.0 * (query_emb @ train_emb.T)
    d2 = np.clip(d2, 0.0, None)
    k = min(KNN_K, train_emb.shape[0])
    part_idx = np.partition(d2, k - 1, axis=1)[:, :k]
    neg_knn = -np.sqrt(part_idx).mean(1)

    return {
        "cos_proto": cos_proto,
        "cos_margin": cos_margin,
        "neg_maha": neg_maha,
        "neg_knn": neg_knn,
    }


def extract_train_embeddings(job, model, device):
    from fedcore.medical.data import MedicalImageDataset
    from fedcore.medical.flamby import eval_transform

    records = []
    for client_records in job.training_by_client:
        records.extend(client_records)
    labels = np.asarray([r.label for r in records], dtype=np.int64)
    ds = MedicalImageDataset(records, transform=eval_transform(IMAGE_SIZE))
    _lg, emb = forward_logits_embeddings(model, ds, device, want_logits=False)
    return emb, labels


def stage_embed():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(EMB_DIR, exist_ok=True)
    cells = cell_manifest()
    for i, entry in enumerate(cells):
        cell = entry["cell"]
        job = load_job(entry)
        model, _meta, _ = build_model(job, entry["ckpt"], device)
        n_known = job.n_known

        train_emb, train_label = extract_train_embeddings(job, model, device)

        prop_arr, prop_emb = reproduce_fold(job, model, device, "proposal", want_embed=True)
        cert_arr, cert_emb = reproduce_fold(job, model, device, "certification", want_embed=True)

        prop_scores = _embedding_scores(train_emb, train_label, n_known, prop_emb)
        cert_scores = _embedding_scores(train_emb, train_label, n_known, cert_emb)

        cache = dict(
            cell=cell,
            n_known=n_known,
            heldout="+".join(job.heldout_diagnoses),
            prop_logits=np.asarray(prop_arr["logits"], np.float64),
            prop_y=np.asarray(prop_arr["y_open"], np.int64),
            prop_client=np.asarray(prop_arr["client"], np.int64),
            prop_sample_id=np.asarray(prop_arr["sample_id"], str),
            cert_logits=np.asarray(cert_arr["logits"], np.float64),
            cert_y=np.asarray(cert_arr["y_open"], np.int64),
            cert_client=np.asarray(cert_arr["client"], np.int64),
            cert_sample_id=np.asarray(cert_arr["sample_id"], str),
            train_n=len(train_label),
        )
        for sc in EMB_SCORES:
            cache[f"prop_score__{sc}"] = np.asarray(prop_scores[sc], np.float64)
            cache[f"cert_score__{sc}"] = np.asarray(cert_scores[sc], np.float64)
        np.savez_compressed(os.path.join(EMB_DIR, cell + ".npz"), **cache)
        print(
            f"[embed {i+1:2d}/46] {cell}  train={len(train_label)}  "
            f"prop={len(cache['prop_y'])} cert={len(cache['cert_y'])}  cached",
            flush=True,
        )
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"\n=== EMBED DONE ===  cache dir: {EMB_DIR}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("gate", "embed"), required=True)
    args = ap.parse_args()
    if args.stage == "gate":
        stage_gate()
    else:
        stage_embed()


if __name__ == "__main__":
    main()
