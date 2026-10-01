import json
import subprocess
from copy import deepcopy

import pytest

from fedcore.experiments.v3_contract import ContractError, sha256_file
from fedcore.experiments import v3_host_executor as host


def _host_artifacts():
    return host.HostArtifacts(
        static_plan={},
        input_binding={},
        mount_receipt={},
        readiness={},
        authorization={"authorization_sha256": "a" * 64},
        authorized_plan={
            "authorized_plan_sha256": "b" * 64,
            "docker_create_argv": ["docker", "create", "image"],
            "docker_start_argv_template": [
                "docker",
                "start",
                "--attach",
                "__CONTAINER_ID__",
            ],
        },
    )


def _comparison_inspect(container_id="d" * 64):
    return [
        {
            "Id": container_id,
            "Created": "2026-10-01T00:00:00Z",
            "State": {"Status": "created"},
            "Config": {"Env": ["FIRST=1", "SECOND=2"]},
            "Mounts": [
                {
                    "Type": "bind",
                    "Source": "/sealed/input",
                    "Destination": "/input",
                    "RW": False,
                    "Mode": "ro",
                },
                {
                    "Type": "bind",
                    "Source": "/sealed/output",
                    "Destination": "/output",
                    "RW": True,
                    "Mode": "rw",
                },
            ],
        }
    ]


def test_attached_watchdog_streams_logs_and_seals_terminal_receipt(tmp_path):
    calls = []

    def runner(argv, **kwargs):
        calls.append((list(argv), dict(kwargs)))
        kwargs["stdout"].write(b"registered stdout\n")
        kwargs["stderr"].write(b"registered stderr\n")
        return subprocess.CompletedProcess(list(argv), 0)

    result = host._run_attached_with_watchdog(
        runner,
        ["docker", "start", "--attach", "a" * 64],
        container_id="a" * 64,
        log_dir=tmp_path / "logs",
        timeout_seconds=17,
    )

    assert result.returncode == 0
    assert "capture_output" not in calls[0][1]
    assert calls[0][1]["stdout"] is not None
    assert calls[0][1]["stderr"] is not None
    assert calls[0][1]["timeout"] == 17
    assert (tmp_path / "logs/container.stdout.log").read_bytes() == b"registered stdout\n"
    receipt = json.loads(
        (tmp_path / "logs/HOST_TERMINAL_RECEIPT.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "PROCESS_EXITED_ZERO"
    assert receipt["timed_out"] is False
    assert receipt["walltime_cap_seconds"] == 17
    assert receipt["cpu_core_limit"] == 0.5
    assert receipt["post_timeout_cleanup_command_bound_seconds"] == 90
    assert receipt["scientific_cleanup_reserve_seconds"] == 300
    assert receipt["scientific_cleanup_safety_margin_seconds"] == 210
    assert receipt["maximum_accounted_gpu_seconds"] == (2 * 60 * 60) + 17 + 300
    assert receipt["gpu_cap_invariant_verified"] is True


def test_watchdog_precharges_cleanup_and_rejects_old_46_hour_timeout(tmp_path):
    assert host.SCIENTIFIC_WALLTIME_SECONDS == (46 * 60 * 60) - 300
    assert host.POST_TIMEOUT_CLEANUP_COMMAND_BOUND_SECONDS == 90
    assert host.SCIENTIFIC_CLEANUP_RESERVE_SECONDS == 300
    assert host.SCIENTIFIC_CLEANUP_SAFETY_MARGIN_SECONDS == 210
    assert (
        int(host.SEALED_CAPACITY_GPU_HOURS * 60 * 60)
        + host.SCIENTIFIC_WALLTIME_SECONDS
        + host.SCIENTIFIC_CLEANUP_RESERVE_SECONDS
    ) == int(host.TOTAL_GPU_HOUR_CAP * 60 * 60)

    log_dir = tmp_path / "must-not-start"
    with pytest.raises(ContractError, match="cleanup-reserved cap"):
        host._run_attached_with_watchdog(
            lambda *args, **kwargs: pytest.fail("over-cap runner must not be called"),
            ["docker", "start", "--attach", "f" * 64],
            container_id="f" * 64,
            log_dir=log_dir,
            timeout_seconds=46 * 60 * 60,
        )
    assert not log_dir.exists()


def test_attached_watchdog_stops_container_and_records_cap_exhaustion(tmp_path):
    calls = []

    def runner(argv, **kwargs):
        calls.append(list(argv))
        if argv[:3] == ["docker", "start", "--attach"]:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        if argv[:3] == ["docker", "inspect", "--format"]:
            return subprocess.CompletedProcess(list(argv), 0, "false\n", "")
        return subprocess.CompletedProcess(list(argv), 0, "stopped\n", "")

    with pytest.raises(ContractError, match="resource cap"):
        host._run_attached_with_watchdog(
            runner,
            ["docker", "start", "--attach", "b" * 64],
            container_id="b" * 64,
            log_dir=tmp_path / "logs",
            timeout_seconds=1,
        )

    assert ["docker", "stop", "--time", "30", "b" * 64] in calls
    receipt = json.loads(
        (tmp_path / "logs/HOST_TERMINAL_RECEIPT.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "HOLD_INCOMPLETE_RESOURCE_CAP"
    assert receipt["timed_out"] is True
    assert receipt["returncode"] == 124
    assert receipt["stop_outcome"]["returncode"] == 0
    assert receipt["kill_outcome"]["attempted"] is False
    assert receipt["termination_command_succeeded"] is True
    assert receipt["container_confirmed_stopped"] is True
    assert receipt["post_timeout_cleanup_command_bound_seconds"] == 90
    assert receipt["scientific_cleanup_reserve_seconds"] == 300
    assert receipt["scientific_cleanup_safety_margin_seconds"] == 210
    assert receipt["gpu_cap_invariant_verified"] is True


def test_attached_watchdog_falls_back_to_bounded_kill_and_seals_receipt(tmp_path):
    calls = []

    def runner(argv, **kwargs):
        calls.append((list(argv), dict(kwargs)))
        if argv[:3] == ["docker", "start", "--attach"]:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        if argv[:2] == ["docker", "stop"]:
            return subprocess.CompletedProcess(list(argv), 1, "", "stop failed")
        if argv[:3] == ["docker", "inspect", "--format"]:
            inspect_count = sum(
                call[0][:3] == ["docker", "inspect", "--format"] for call in calls
            )
            rendered = "true\n" if inspect_count == 1 else "false\n"
            return subprocess.CompletedProcess(list(argv), 0, rendered, "")
        if argv[:2] == ["docker", "kill"]:
            return subprocess.CompletedProcess(list(argv), 0, "killed", "")
        raise AssertionError(argv)

    with pytest.raises(ContractError, match="resource cap"):
        host._run_attached_with_watchdog(
            runner,
            ["docker", "start", "--attach", "7" * 64],
            container_id="7" * 64,
            log_dir=tmp_path / "logs",
            timeout_seconds=1,
        )

    receipt = json.loads(
        (tmp_path / "logs/HOST_TERMINAL_RECEIPT.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "HOLD_INCOMPLETE_RESOURCE_CAP"
    assert receipt["stop_outcome"]["returncode"] == 1
    assert receipt["stop_outcome"]["timeout_seconds"] == 45
    assert receipt["kill_outcome"]["returncode"] == 0
    assert receipt["kill_outcome"]["timeout_seconds"] == 15
    assert receipt["termination_command_succeeded"] is True
    assert receipt["container_confirmed_stopped"] is True


def test_attached_watchdog_kills_when_stop_zero_but_inspect_still_running(tmp_path):
    inspect_calls = 0

    def runner(argv, **kwargs):
        nonlocal inspect_calls
        if argv[:3] == ["docker", "start", "--attach"]:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        if argv[:2] == ["docker", "stop"]:
            return subprocess.CompletedProcess(list(argv), 0, "stopped", "")
        if argv[:3] == ["docker", "inspect", "--format"]:
            inspect_calls += 1
            rendered = "true\n" if inspect_calls == 1 else "false\n"
            return subprocess.CompletedProcess(list(argv), 0, rendered, "")
        if argv[:2] == ["docker", "kill"]:
            return subprocess.CompletedProcess(list(argv), 0, "killed", "")
        raise AssertionError(argv)

    with pytest.raises(ContractError, match="resource cap"):
        host._run_attached_with_watchdog(
            runner,
            ["docker", "start", "--attach", "6" * 64],
            container_id="6" * 64,
            log_dir=tmp_path / "logs",
            timeout_seconds=1,
        )

    receipt = json.loads(
        (tmp_path / "logs/HOST_TERMINAL_RECEIPT.json").read_text(encoding="utf-8")
    )
    assert receipt["stop_outcome"]["returncode"] == 0
    assert receipt["post_stop_inspect_outcome"]["running"] is True
    assert receipt["kill_outcome"]["returncode"] == 0
    assert receipt["post_kill_inspect_outcome"]["running"] is False
    assert receipt["container_confirmed_stopped"] is True


def test_attached_watchdog_seals_receipt_when_stop_and_kill_hang(tmp_path):
    def runner(argv, **kwargs):
        if argv[:3] == ["docker", "start", "--attach"]:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        if argv[:2] in (["docker", "stop"], ["docker", "kill"]):
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        if argv[:3] == ["docker", "inspect", "--format"]:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        raise AssertionError(argv)

    with pytest.raises(ContractError, match="resource cap"):
        host._run_attached_with_watchdog(
            runner,
            ["docker", "start", "--attach", "8" * 64],
            container_id="8" * 64,
            log_dir=tmp_path / "logs",
            timeout_seconds=1,
        )

    receipt_path = tmp_path / "logs/HOST_TERMINAL_RECEIPT.json"
    assert receipt_path.is_file()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["stop_outcome"]["failure_type"] == "TimeoutExpired"
    assert receipt["kill_outcome"]["failure_type"] == "TimeoutExpired"
    assert receipt["termination_command_succeeded"] is False
    assert receipt["post_stop_inspect_outcome"]["failure_type"] == "TimeoutExpired"
    assert receipt["post_kill_inspect_outcome"]["failure_type"] == "TimeoutExpired"
    assert receipt["container_confirmed_stopped"] is False
    assert receipt["status"] == "HOLD_INCOMPLETE_RESOURCE_CAP_CLEANUP_UNCONFIRMED"


def test_attached_watchdog_refuses_overwrite_of_prior_logs(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    with pytest.raises(ContractError, match="already exists"):
        host._run_attached_with_watchdog(
            lambda *args, **kwargs: subprocess.CompletedProcess([], 0),
            ["docker", "start", "--attach", "c" * 64],
            container_id="c" * 64,
            log_dir=log_dir,
        )


def test_attached_watchdog_seals_launch_io_failure_before_reraising(tmp_path):
    def runner(argv, **kwargs):
        kwargs["stderr"].write(b"docker transport failed\n")
        raise OSError("cannot start docker client")

    with pytest.raises(ContractError, match="after terminal receipt sealing"):
        host._run_attached_with_watchdog(
            runner,
            ["docker", "start", "--attach", "e" * 64],
            container_id="e" * 64,
            log_dir=tmp_path / "logs",
            timeout_seconds=17,
        )

    assert (tmp_path / "logs/container.stdout.log").read_bytes() == b""
    assert (tmp_path / "logs/container.stderr.log").read_bytes() == (
        b"docker transport failed\n"
    )
    receipt = json.loads(
        (tmp_path / "logs/HOST_TERMINAL_RECEIPT.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "HOLD_INCOMPLETE"
    assert receipt["returncode"] is None
    assert receipt["timed_out"] is False
    assert receipt["failure_type"] == "OSError"
    assert receipt["failure_message"] == "cannot start docker client"
    assert len(receipt["stdout_sha256"]) == 64
    assert len(receipt["stderr_sha256"]) == 64


def test_host_prepare_rejects_tampered_source_probe_receipt(monkeypatch, tmp_path):
    control = tmp_path / "control"
    control.mkdir()

    def write(name, value):
        path = control / name
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        return path

    static_plan = write("HOST_LAUNCH_PLAN.json", {})
    input_binding = write("INPUT_BINDING.json", {})
    mount_receipt = write("MOUNT_RECEIPT.json", {})
    source_manifest = write("SOURCE_MANIFEST.json", {"sealed": True})
    source_probe = write("SOURCE_PROBE_RECEIPT.json", {"sealed": True})
    test_report = write("TEST_REPORT.json", {"sealed": True})
    readiness = {
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_probe_receipt_sha256": sha256_file(source_probe),
        "test_report_sha256": sha256_file(test_report),
    }
    write("EXECUTION_READINESS.json", readiness)
    write("RUN_AUTHORIZATION.json", {})
    write("AUTHORIZED_CAMPAIGN_PLAN.json", {})
    source_probe.write_text('{"sealed":false}\n', encoding="utf-8")
    for path in control.iterdir():
        path.chmod(0o444)

    monkeypatch.setattr(host, "EXPECTED_HOST_CONTROL_DIR", control.resolve())
    monkeypatch.setattr(
        host, "STATIC_LAUNCH_PLAN_FILE_SHA256", sha256_file(static_plan)
    )
    monkeypatch.setattr(
        host, "STATIC_INPUT_BINDING_FILE_SHA256", sha256_file(input_binding)
    )
    monkeypatch.setattr(
        host, "STATIC_MOUNT_RECEIPT_FILE_SHA256", sha256_file(mount_receipt)
    )
    monkeypatch.setattr(host, "validate_execution_readiness", lambda *a, **k: None)
    monkeypatch.setattr(host, "validate_run_authorization", lambda *a, **k: None)
    monkeypatch.setattr(
        host, "validate_authorized_campaign_plan", lambda *a, **k: None
    )

    with pytest.raises(ContractError, match="source-probe receipt bytes differ"):
        host._load_host_artifacts(control, prepared=False)


def test_prepare_then_start_reinspects_exact_stopped_container(monkeypatch, tmp_path):
    control = tmp_path / "control"
    output = tmp_path / "output"
    control.mkdir()
    output.mkdir()
    artifacts = _host_artifacts()
    container_id = "d" * 64
    inspect_value = _comparison_inspect(container_id)
    receipt = {
        "container_id": container_id,
        "runtime_receipt_sha256": "e" * 64,
    }
    commands = []
    inspect_calls = 0
    inspect_mount_orders = []
    lease_calls = []

    def runner(argv, **kwargs):
        nonlocal inspect_calls
        commands.append(list(argv))
        if argv[:2] == ["docker", "create"]:
            return subprocess.CompletedProcess(list(argv), 0, container_id + "\n", "")
        if argv[:2] == ["docker", "inspect"]:
            inspect_calls += 1
            rendered = deepcopy(inspect_value)
            if inspect_calls == 2:
                rendered[0]["Mounts"].reverse()
            inspect_mount_orders.append(
                [row["Destination"] for row in rendered[0]["Mounts"]]
            )
            return subprocess.CompletedProcess(
                list(argv), 0, json.dumps(rendered), ""
            )
        raise AssertionError(argv)

    monkeypatch.setattr(host, "_load_host_artifacts", lambda *a, **k: artifacts)
    monkeypatch.setattr(host, "_require_directory", lambda *a, **k: output)
    monkeypatch.setattr(
        host,
        "acquire_exclusive_lease",
        lambda *args, **kwargs: lease_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        host, "build_runtime_receipt_from_inspect", lambda *a, **k: dict(receipt)
    )

    prepared = host.prepare_scientific_container(
        control_dir=control,
        output_dir=output,
        lease_path=tmp_path / "prepare.lease",
        runner=runner,
    )
    assert prepared["status"] == "PASS_CREATED_CONTAINER_INSPECTED_HOLD_START"
    assert prepared["execution_started"] is False
    assert commands == [
        ["docker", "create", "image"],
        ["docker", "inspect", container_id],
    ]

    monkeypatch.setattr(host, "validate_runtime_receipt", lambda *a, **k: None)
    attached = {}
    attach_calls = 0

    def attach(*args, **kwargs):
        nonlocal attach_calls
        attach_calls += 1
        attached["argv"] = list(args[1])
        attached["container_id"] = kwargs["container_id"]
        return subprocess.CompletedProcess(list(args[1]), 0)

    monkeypatch.setattr(host, "_run_attached_with_watchdog", attach)
    commands.clear()
    result = host.start_scientific_container(
        control_dir=control,
        output_dir=output,
        lease_path=tmp_path / "start.lease",
        log_dir=tmp_path / "logs",
        runner=runner,
    )
    assert result.returncode == 0
    assert commands == [["docker", "inspect", container_id]]
    assert attached == {
        "argv": ["docker", "start", "--attach", container_id],
        "container_id": container_id,
    }
    assert inspect_calls == 2
    assert inspect_mount_orders == [["/input", "/output"], ["/output", "/input"]]
    assert attach_calls == 1
    assert len(lease_calls) == 2

    inspect_value[0]["Created"] = "2026-10-01T00:00:01Z"
    with pytest.raises(ContractError, match="stopped container drifted"):
        host.start_scientific_container(
            control_dir=control,
            output_dir=output,
            lease_path=tmp_path / "drift-start.lease",
            log_dir=tmp_path / "drift-logs",
            runner=runner,
        )
    assert inspect_calls == 3
    assert attach_calls == 1
    assert len(lease_calls) == 2


def test_runtime_inspect_comparison_rejects_mount_content_drift():
    prepared = _comparison_inspect()
    fresh = deepcopy(prepared)
    fresh[0]["Mounts"].reverse()
    fresh[0]["Mounts"][0]["RW"] = False

    with pytest.raises(ContractError, match="stopped container drifted"):
        host._validate_stopped_container_unchanged(prepared, fresh)


def test_runtime_inspect_comparison_rejects_non_mount_field_drift():
    prepared = _comparison_inspect()
    fresh = deepcopy(prepared)
    fresh[0]["Mounts"].reverse()
    fresh[0]["Created"] = "2026-10-01T00:00:01Z"

    with pytest.raises(ContractError, match="stopped container drifted"):
        host._validate_stopped_container_unchanged(prepared, fresh)


def test_runtime_inspect_comparison_preserves_raw_rows_and_other_list_order():
    prepared = _comparison_inspect()
    fresh = deepcopy(prepared)
    fresh[0]["Mounts"].reverse()
    prepared_before = deepcopy(prepared)
    fresh_before = deepcopy(fresh)

    host._validate_stopped_container_unchanged(prepared, fresh)
    assert prepared == prepared_before
    assert fresh == fresh_before

    mount_field_drift = deepcopy(fresh)
    mount_field_drift[0]["Mounts"][0]["Mode"] = "delegated"
    with pytest.raises(ContractError, match="stopped container drifted"):
        host._validate_stopped_container_unchanged(prepared, mount_field_drift)

    other_list_drift = deepcopy(fresh)
    other_list_drift[0]["Config"]["Env"].reverse()
    with pytest.raises(ContractError, match="stopped container drifted"):
        host._validate_stopped_container_unchanged(prepared, other_list_drift)
