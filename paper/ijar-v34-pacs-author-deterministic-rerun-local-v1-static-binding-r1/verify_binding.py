#!/usr/bin/env python3
"""Verify the local PACS static control binding without third-party packages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_STATUS = "PASS_LOCAL_STATIC_BINDING_HOLD_RUNTIME_VALIDATION"
PARENT_ROOT = "362f4e9dc19b9276db54fdef2748887296bc0c01f8e40af56b90a2613a5db752"
PERMITTED_CLAIM = "author-executed deterministic full-denominator rerun"
FORBIDDEN_CLAIMS = [
    "independent replication",
    "prospective confirmation",
    "untouched evaluation",
    "outcome-blind confirmation",
]
LOCAL_ROOT = "/home/sanghoon/Desktop/Workspace/Fedcore_HSB_local_v1"
LOCAL_INPUT_DIR = f"{LOCAL_ROOT}/inputs"
LOCAL_OUTPUT_DIR = (
    f"{LOCAL_ROOT}/runs/scientific/FEDCORE-IJAR-PACS-AUTHOR-DETERMINISTIC-RERUN-HSB-local-v1"
)
LEGACY_ROOTS = ("Fedcore_HSB_v3_exec", "Fedcore_HSB_v4_exec")
IMAGE_ID = "sha256:943f4ca6d930c839489a8de50b0acef434c0d5d9b5775280c6d757584aaaef94"
IMAGE_REVISION = "4fb2d580e96a99f25f6cc5967ff10484ef0f7bd1"
PROBE_LOG = "PASS_INPUTS_READ_ONLY\nPASS_ROOT_READ_ONLY\nPASS_TMP_RW\nPASS_OUTPUT_RW\n"
EXPECTED_INPUTS = (
    ("PACS_mirror.zip", 184417365, "42bf567f1ed8a01d522e47e4a677e2a3149577bbd6fcbb38bedfdd73cb59e147",
     "/home/sanghoon/Desktop/Workspace/Fedcore/data/v1_pacs/PACS_mirror.zip"),
    ("IMAGE_MANIFEST.csv", 1537910, "283d47bd8cfb7017d6e7fde6d65e07ff6479ab25127afcba135da3a06775afae",
     "/home/sanghoon/Desktop/Workspace/Fedcore/outputs/fedcore_v1_ubuntu/IMAGE_MANIFEST.csv"),
    ("FOLD_MANIFEST.csv", 7853382, "fe5b64d237f47f503b8895aa5e566727bc818e05ee597f84c440df70e91e5344",
     "/home/sanghoon/Desktop/Workspace/Fedcore/outputs/fedcore_v1_ubuntu/FOLD_MANIFEST.csv"),
    ("resnet18-f37072fd.pth", 46830571, "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
     "/home/sanghoon/Desktop/Workspace/Fedcore/data/v1_pacs/torch_cache/hub/checkpoints/resnet18-f37072fd.pth"),
    ("convnext_tiny-983f1562.pth", 114419221, "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d",
     "/home/sanghoon/Desktop/Workspace/Fedcore/data/v1_pacs/torch_cache/hub/checkpoints/convnext_tiny-983f1562.pth"),
)
GOVERNING_FILES = {
    "HOST_LAUNCH_PLAN.json",
    "IMAGE_INSPECT.json",
    "INPUT_BINDING.json",
    "MOUNT_PROBE.inspect.json",
    "MOUNT_PROBE.stdout.txt",
    "MOUNT_RECEIPT.json",
    "README.md",
    "verify_binding.py",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def load(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def fail(code: str) -> None:
    raise SystemExit(f"INVALID_{code}")


def main() -> int:
    # 1. file hashes, exact file set, bundle root
    listed: dict[str, str] = {}
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        if name in listed:
            fail(f"DUPLICATE_MANIFEST_ROW {name}")
        if digest(ROOT / name) != expected:
            fail(f"BUNDLE_HASH {name}")
        listed[name] = expected
    if set(listed) != GOVERNING_FILES | {"SEAL.json"}:
        fail("MANIFEST_FILE_SET")
    present = {p.name for p in ROOT.iterdir() if p.name != "__pycache__"}
    if present != set(listed) | {"SHA256SUMS.txt"}:
        fail(f"BUNDLE_FILE_SET {sorted(present ^ (set(listed) | {'SHA256SUMS.txt'}))}")
    rows = [f"{listed[name]}  {name}\n" for name in sorted(GOVERNING_FILES)]
    root_hash = hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()
    seal = load("SEAL.json")
    if root_hash != seal["bundle_root_sha256"]:
        fail("BUNDLE_ROOT")

    # 2. seal lineage, status, claims
    if seal["status"] != EXPECTED_STATUS:
        fail("SEAL_STATUS")
    if seal["parent_static_binding_root_sha256"] != PARENT_ROOT or seal["parent_is_active_local_binding"] is not False:
        fail("SEAL_PARENT")
    if seal["permitted_claim"] != PERMITTED_CLAIM or seal["forbidden_claims"] != FORBIDDEN_CLAIMS:
        fail("SEAL_CLAIMS")
    for field in ("training_authorized", "scientific_execution_authorized", "run_authorized", "submission_authorized",
                  "pacs_training_started", "pacs_inference_started", "GPU_used_for_science", "CUDA_initialized"):
        if seal[field] is not False:
            fail(f"SEAL_FIELD {field}")

    # 3. input binding: denominator, identities, local staged paths
    binding_hash = listed["INPUT_BINDING.json"]
    binding = load("INPUT_BINDING.json")
    if (binding["status"] != "PASS_FIVE_INPUT_BINDING" or binding["isolated_input_file_count"] != 5
            or len(binding["files"]) != 5 or binding["isolated_input_directory"] != LOCAL_INPUT_DIR):
        fail("INPUT_BINDING")
    for field in ("exact_directory_contents", "all_regular_nonsymlink", "all_read_only"):
        if binding[field] is not True:
            fail(f"INPUT_BINDING_FIELD {field}")
    for field in ("scientific_content_interpreted", "pacs_training_started", "pacs_inference_started"):
        if binding[field] is not False:
            fail(f"INPUT_BINDING_FIELD {field}")
    for record, (name, size, sha256, source) in zip(binding["files"], EXPECTED_INPUTS, strict=True):
        if (record["name"] != name or record["size_bytes"] != size or record["sha256"] != sha256
                or record["source_path"] != source or record["staged_path"] != f"{LOCAL_INPUT_DIR}/{name}"
                or record["container_path"] != f"/inputs/{name}" or record["staged_mode_octal"] != "0444"
                or record["regular_file"] is not True or record["symlink"] is not False
                or record["read_only_mount_required"] is not True):
            fail(f"INPUT_RECORD {name}")

    # 4. launch plan: canonical bytes, internal hash, local mounts, no execution
    plan_bytes = (ROOT / "HOST_LAUNCH_PLAN.json").read_bytes()
    plan = json.loads(plan_bytes)
    if canonical(plan) != plan_bytes:
        fail("LAUNCH_PLAN_NOT_CANONICAL")
    body = {k: v for k, v in plan.items() if k not in {"launch_plan_sha256", "launch_plan_hash_definition"}}
    if hashlib.sha256(canonical(body)).hexdigest() != plan["launch_plan_sha256"]:
        fail("LAUNCH_PLAN_INTERNAL_HASH")
    if (plan["status"] != "PASS_CANDIDATE_HOST_LAUNCH_PLAN_HOLD_EXECUTION_READINESS"
            or plan["execution_allowed"] is not False or plan["run_authorization_required"] is not True
            or plan["execution_readiness_required"] is not True):
        fail("LAUNCH_PLAN_STATUS")
    if plan["input_binding_sha256"] != binding_hash or plan["image"]["id"] != IMAGE_ID:
        fail("LAUNCH_PLAN_BINDING")
    for field in ("PACS_records_decoded", "PACS_labels_inspected", "PACS_scientific_content_opened",
                  "torch_imported", "CUDA_touched"):
        if plan[field] is not False:
            fail(f"LAUNCH_PLAN_FIELD {field}")
    if len(plan["cell_order"]) != 30 or len(plan["cell_launches"]) != 30 or len(plan["input_inventory"]) != 5:
        fail("LAUNCH_PLAN_DENOMINATOR")
    expected_mounts = [
        {"type": "bind", "source": f"{LOCAL_INPUT_DIR}/{name}", "destination": f"/inputs/{name}", "read_only": True}
        for name, *_ in EXPECTED_INPUTS
    ] + [
        {"type": "bind", "source": LOCAL_OUTPUT_DIR, "destination": "/output", "read_only": False},
        {"type": "tmpfs", "source": "", "destination": "/tmp", "read_only": False, "options": "rw,noexec,nosuid,size=1g"},
    ]
    if plan["mounts"] != expected_mounts or plan["output_directory"] != LOCAL_OUTPUT_DIR:
        fail("LAUNCH_PLAN_MOUNTS")
    if any(root in plan_bytes.decode("utf-8") for root in LEGACY_ROOTS):
        fail("LAUNCH_PLAN_LEGACY_ROOT")

    # 5. raw probe evidence
    inspect_list = load("MOUNT_PROBE.inspect.json")
    if len(inspect_list) != 1:
        fail("PROBE_INSPECT_COUNT")
    inspect = inspect_list[0]
    host, config, state = inspect["HostConfig"], inspect["Config"], inspect["State"]
    if host["DeviceRequests"] not in (None, []) or host["Devices"] not in (None, []):
        fail("GPU_DEVICE_SCOPE")
    if (host["NetworkMode"] != "none" or host["ReadonlyRootfs"] is not True or host["Privileged"] is not False
            or host["Runtime"] != "runc" or host["CapDrop"] != ["ALL"]
            or "no-new-privileges" not in (host["SecurityOpt"] or [])
            or host["Tmpfs"] != {"/tmp": "rw,noexec,nosuid,size=1g"}):
        fail("CONTAINER_ISOLATION")
    if config["User"] != "1000:1000" or "NVIDIA_VISIBLE_DEVICES=void" not in config["Env"]:
        fail("CONTAINER_IDENTITY")
    if config["Entrypoint"] != ["python"] or config["Cmd"] != ["-m", "fedcore.experiments.v3_mount_probe"]:
        fail("PROBE_COMMAND")
    if state["Status"] != "exited" or state["ExitCode"] != 0 or state["OOMKilled"] is not False:
        fail("PROBE_STATE")
    if inspect["Image"] != IMAGE_ID:
        fail("PROBE_IMAGE")
    observed = sorted((m["Type"], m["Source"], m["Destination"], m["RW"]) for m in inspect["Mounts"])
    wanted = sorted(("bind", m["source"], m["destination"], not m["read_only"]) for m in expected_mounts if m["type"] == "bind")
    if observed != wanted:
        fail("PROBE_MOUNTS")
    if (ROOT / "MOUNT_PROBE.stdout.txt").read_text(encoding="utf-8") != PROBE_LOG:
        fail("PROBE_OUTPUT")

    # 6. mount receipt
    receipt = load("MOUNT_RECEIPT.json")
    if receipt["status"] != "PASS_EXACT_FIVE_RO_MOUNT_PROBE_NO_GPU":
        fail("MOUNT_RECEIPT_STATUS")
    if receipt["launch_plan_sha256"] != plan["launch_plan_sha256"] or receipt["input_binding_sha256"] != binding_hash:
        fail("MOUNT_RECEIPT_BINDING")
    if receipt["container_id"] != inspect["Id"] or receipt["image_id"] != IMAGE_ID or receipt["device_requests"] is not None:
        fail("MOUNT_RECEIPT_CONTAINER")
    for field in ("exact_five_read_only_inputs", "one_read_write_output", "tmpfs_only_at_tmp", "read_only_root_filesystem"):
        if receipt[field] is not True:
            fail(f"MOUNT_RECEIPT_FIELD {field}")
    for field in ("GPU_used", "training_started", "PACS_records_decoded", "PACS_labels_inspected", "legacy_workspace_mounted"):
        if receipt[field] is not False:
            fail(f"MOUNT_RECEIPT_FIELD {field}")
    if receipt["network_mode"] != "none":
        fail("MOUNT_RECEIPT_NETWORK")

    # 7. probe image
    image = load("IMAGE_INSPECT.json")
    if len(image) != 1 or image[0]["Id"] != IMAGE_ID:
        fail("IMAGE_ID")
    if image[0]["Config"]["Labels"]["org.opencontainers.image.revision"] != IMAGE_REVISION:
        fail("IMAGE_REVISION_LABEL")
    if seal["prebinding_probe_image_id"] != IMAGE_ID:
        fail("SEAL_IMAGE")

    # 8. optional host check of the staged inputs (skipped on other machines)
    staged = Path(LOCAL_INPUT_DIR)
    if staged.is_dir():
        if sorted(p.name for p in staged.iterdir()) != sorted(name for name, *_ in EXPECTED_INPUTS):
            fail("STAGED_INPUT_SET")
        for name, size, sha256, _ in EXPECTED_INPUTS:
            path = staged / name
            if path.is_symlink() or path.stat().st_size != size or (path.stat().st_mode & 0o222) or digest(path) != sha256:
                fail(f"STAGED_INPUT {name}")
        print("staged_inputs=5/5 rehashed")
    else:
        print("staged_inputs=SKIPPED_NOT_ON_THIS_HOST")

    print(EXPECTED_STATUS)
    print(f"bundle_root_sha256={root_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
