"""F3b provenance metadata writer (runs in-container as root).

Emits f3b_run_meta.json: docker image digest + exact command (from env), GPU UUID
(nvidia-smi -L), pip versions, a sha256 sample of canonical inputs (including the
root-only 0600 logit npz and the integrity anchor), and sha256 of every generated
F3b output.  Reads only; modifies no canonical file.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess

REPO = os.environ.get("F3B_REPO", "/workspace")
OUT = os.environ.get("F3B_OUT", "/out")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pip_versions():
    import importlib.metadata as im

    out = {}
    for pkg in ("torch", "numpy", "scipy", "efficientnet_pytorch",
                "efficientnet-pytorch", "albumentations", "pillow", "Pillow"):
        try:
            out[pkg] = im.version(pkg)
        except Exception:
            pass
    return out


def gpu_uuid():
    try:
        return subprocess.check_output(["nvidia-smi", "-L"], text=True).strip()
    except Exception as exc:
        return f"nvidia-smi failed: {exc}"


def main():
    canonical_samples = {}
    for rel in [
        "results/final/fedisic_eligible_32_risk_decomposition.csv",
        "results/source_data/fed_isic2019_metadata.csv",
        "results/source_data/fed_isic2019_folds_split00_seed0.csv",
        "runs/oneshot/full/ckpt/fed_isic2019__fed-isic__split00__seed0__dna__alphaANY__variantANY.pt",
        "runs/oneshot/full/logits/fed_isic2019__fed-isic__split00__seed0__dna__alphaANY__variantANY.npz",
    ]:
        p = os.path.join(REPO, rel)
        if os.path.exists(p):
            canonical_samples[rel] = sha256(p)

    outputs = {}
    for p in sorted(glob.glob(os.path.join(OUT, "f3b_*.csv"))
                    + glob.glob(os.path.join(OUT, "f3b_*.json"))
                    + glob.glob(os.path.join(OUT, "f3b_*.txt"))):
        outputs[os.path.basename(p)] = sha256(p)

    meta = {
        "phase": "F3b_embedding_selector_rescue",
        "docker_image": os.environ.get("F3B_IMAGE", ""),
        "docker_image_digest": os.environ.get("F3B_IMAGE_DIGEST", ""),
        "docker_run_command": os.environ.get("F3B_DOCKER_CMD", ""),
        "gpu": gpu_uuid(),
        "pip_versions": pip_versions(),
        "canonical_input_sha256_sample": canonical_samples,
        "output_sha256": outputs,
        "n_embedding_cells": len(glob.glob(os.path.join(OUT, "_emb", "*isic*.npz"))),
        "notes": (
            "Fidelity gate PASSED 46/46 reproducible cells (see "
            "f3b_reinference_fidelity.csv). 4 canonical cells (split04 seed0-3) have "
            "no logit npz and are non-reproducible; gate scope is 46/46 available. "
            "Reference=TRAIN only; thresholds=PROPOSAL only; no cert/eval label in "
            "candidate construction. Per-client FULL SIMPLEX, J<=6."
        ),
    }
    with open(os.path.join(OUT, "f3b_run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote f3b_run_meta.json")
    print(json.dumps({k: meta[k] for k in ("gpu", "pip_versions")}, indent=2))


if __name__ == "__main__":
    main()
