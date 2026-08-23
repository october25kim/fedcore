"""Resolve a frozen ``runs/*_logits.npz`` back to the config that produced it.

Two sources, in order of authority:

1. ``scripts/ws4090/manifest_*.txt`` -- ``LABEL <TAB> EXPECTED_NPZ <TAB> COMMAND``.
   The COMMAND is the verbatim ``run_cifar`` CLI, so it carries every knob that
   drives the split, including ``--unknown_classes`` (which the filename does not).
   This is the authoritative source.
2. Filename convention -- legacy exploratory runs predating the manifests. Only the
   seed-driven open-set split is recoverable this way; a run that actually used
   ``--unknown_classes`` will FAIL verification in ``ids.py`` rather than resolve
   to a wrong split.

Neither source is trusted on its own: ``ids.py`` verifies the resulting config by
recomputing the split and requiring exact agreement with the stored arrays.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

MANIFEST_GLOB = "scripts/ws4090/manifest_*.txt"

# Datasets whose calibration points live in a recoverable index space. The tabular
# covtype pipeline (run_tabular.py) does not share the CIFAR test-set index space
# and is reported as unresolved rather than guessed.
SUPPORTED_DATASETS = ("cifar10", "cifar100")


@dataclass(frozen=True)
class RunSpec:
    """Everything needed to recompute a run's calibration split, plus provenance."""

    npz_path: str
    run_id: str
    dataset: str
    backbone: str
    n_known: int
    n_clients: int
    dirichlet_alpha: float
    noise_type: str
    noise_rate: float
    seed: int
    unknown_classes: Optional[Tuple[int, ...]]
    provenance_source: str  # 'manifest' | 'filename'
    manifest_file: str = ""
    manifest_label: str = ""

    @property
    def heterogeneity_d(self) -> float:
        return self.dirichlet_alpha

    @property
    def model(self) -> str:
        return self.backbone

    # Seeds are aliased in run_cifar.py: a single --seed drives the open-set split,
    # the Dirichlet partition, calibration construction, and torch init. They are
    # exposed as distinct columns because the schema requires them, and reported as
    # aliased because that is the truth. See docs/agent_plan_phase1.md F3.
    @property
    def split_seed(self) -> int:
        return self.seed

    @property
    def train_seed(self) -> int:
        return self.seed

    @property
    def partition_seed(self) -> int:
        return self.seed


def _run_cifar_parser() -> argparse.ArgumentParser:
    """A parser mirroring run_cifar.py's CLI, used to read manifest commands.

    Kept deliberately permissive (``parse_known_args``) so an unrelated future flag
    does not break provenance resolution.
    """
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--n_known", type=int, default=6)
    ap.add_argument("--n_clients", type=int, default=5)
    ap.add_argument("--dirichlet_alpha", type=float, default=0.1)
    ap.add_argument("--noise_type", default="none")
    ap.add_argument("--noise_rate", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone", default="simplecnn")
    ap.add_argument("--norm", default="bn")
    ap.add_argument("--unknown_classes", default=None)
    ap.add_argument("--out", default="runs/cifar_results.csv")
    return ap


def _backbone_label(backbone: str, norm: str) -> str:
    """Match the naming used by gen_manifest.py / aggregate.main.parse_tag."""
    if backbone == "resnet18":
        return "resnet18gn" if norm == "gn" else "resnet18"
    return backbone


def load_manifest_index(root: str = ".") -> Dict[str, Dict[str, str]]:
    """Map basename(npz) -> {command, manifest_file, label} over all manifests."""
    index: Dict[str, Dict[str, str]] = {}
    for path in sorted(glob.glob(os.path.join(root, MANIFEST_GLOB))):
        with open(path) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                label, npz, cmd = parts[0], parts[1], parts[2]
                key = os.path.basename(npz)
                if key in index and index[key]["command"] != cmd:
                    raise ValueError(
                        f"{key} declared with conflicting commands in "
                        f"{index[key]['manifest_file']} and {os.path.basename(path)}"
                    )
                index[key] = {
                    "command": cmd,
                    "manifest_file": os.path.basename(path),
                    "label": label,
                }
    return index


def _from_manifest(npz_path: str, entry: Dict[str, str]) -> RunSpec:
    argv = shlex.split(entry["command"])
    args, _unknown = _run_cifar_parser().parse_known_args(argv[1:])
    unk = (
        tuple(int(c) for c in args.unknown_classes.split(","))
        if args.unknown_classes
        else None
    )
    return RunSpec(
        npz_path=npz_path,
        run_id=os.path.basename(npz_path).replace("_logits.npz", ""),
        dataset=args.dataset,
        backbone=_backbone_label(args.backbone, args.norm),
        n_known=args.n_known,
        n_clients=args.n_clients,
        dirichlet_alpha=args.dirichlet_alpha,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        seed=args.seed,
        unknown_classes=unk,
        provenance_source="manifest",
        manifest_file=entry["manifest_file"],
        manifest_label=entry["label"],
    )


def _from_filename(npz_path: str, n_known: int, n_clients: int) -> Optional[RunSpec]:
    """Legacy fallback. ``n_known``/``n_clients`` come from the npz itself.

    Mirrors ``fedcore.aggregate.main.parse_tag`` so the accounting layer groups runs
    into exactly the same cells the headline aggregator does.
    """
    fn = os.path.basename(npz_path)
    dataset = (
        "cifar100" if "cifar100" in fn else ("cifar10" if "cifar10" in fn else None)
    )
    if dataset is None:
        return None
    backbone = (
        "resnet18gn"
        if "resnet18gn" in fn
        else "resnet18" if "resnet18" in fn else "simplecnn"
    )
    md = re.search(r"_d([0-9.]+)", fn)
    ms = re.search(r"seed(\d+)", fn)
    if not (md and ms):
        return None
    mn = re.search(r"(none|symmetric|asymmetric)([0-9.]+)", fn)
    noise_type = mn.group(1) if mn else "none"
    noise_rate = float(mn.group(2)) if mn else 0.0
    return RunSpec(
        npz_path=npz_path,
        run_id=fn.replace("_logits.npz", ""),
        dataset=dataset,
        backbone=backbone,
        n_known=n_known,
        n_clients=n_clients,
        dirichlet_alpha=float(md.group(1)),
        noise_type=noise_type,
        noise_rate=noise_rate,
        seed=int(ms.group(1)),
        unknown_classes=None,
        provenance_source="filename",
    )


def resolve_run(
    npz_path: str,
    manifest_index: Optional[Dict[str, Dict[str, str]]] = None,
) -> Optional[RunSpec]:
    """Resolve one npz to a ``RunSpec``, or ``None`` if it is out of scope.

    ``None`` means "no CIFAR index space to recover" (e.g. covtype) -- the caller
    reports it as unresolved. It never means "guessed".
    """
    fn = os.path.basename(npz_path)
    if manifest_index is None:
        manifest_index = load_manifest_index()
    if fn in manifest_index:
        spec = _from_manifest(npz_path, manifest_index[fn])
        if spec.dataset not in SUPPORTED_DATASETS:
            return None
        return spec

    with np.load(npz_path) as z:
        if "prop_logits" not in z.files or "cert_client" not in z.files:
            return None
        n_known = int(z["prop_logits"].shape[1])
        n_clients = int(z["cert_client"].max()) + 1
    return _from_filename(npz_path, n_known, n_clients)


def discover_runs(root: str = ".") -> List[Tuple[str, Optional[RunSpec]]]:
    """All ``runs/*_logits.npz`` paired with their RunSpec (None = out of scope)."""
    index = load_manifest_index(root)
    out = []
    for p in sorted(glob.glob(os.path.join(root, "runs", "*_logits.npz"))):
        out.append((p, resolve_run(p, index)))
    return out
