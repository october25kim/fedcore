"""Tests for the restart-safe local campaign scheduler.

Run directly with ``python tests/test_scheduler.py`` or collect with pytest.
The tests use only the Python standard library and never invoke a shell.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedcore.campaign.scheduler import (  # noqa: E402
    ExclusiveFileLock,
    JobLockedError,
    JobSpec,
    PersistentLocalScheduler,
    SpecMismatchError,
    atomic_write_json,
    sha256_file,
    validate_artifact,
)


WRITE_JSON = r"""
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"schema_version": 7, "metric": 0.25}), encoding="utf-8")
print("worker stdout")
print("worker stderr", file=sys.stderr)
"""


RETRY_ONCE = r"""
import json, pathlib, sys
artifact = pathlib.Path(sys.argv[1])
counter = pathlib.Path(sys.argv[2])
artifact.parent.mkdir(parents=True, exist_ok=True)
attempt = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(attempt), encoding="utf-8")
print("attempt", attempt)
if attempt == 1:
    print("technical failure", file=sys.stderr)
    raise SystemExit(23)
artifact.write_text(json.dumps({"schema_version": 7, "metric": 0.5}), encoding="utf-8")
"""


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.queue = self.root / "queue"
        self.artifact = self.root / "results" / "artifact.json"

    def tearDown(self):
        self.tempdir.cleanup()

    def job(self, *, experiment_id="cifar/split-0/model-0", argv=None, retry_limit=0):
        job_key = experiment_id.replace("/", "-")
        return JobSpec(
            experiment_id=experiment_id,
            argv=argv or [sys.executable, "-c", WRITE_JSON, str(self.artifact)],
            expected_artifact=str(self.artifact),
            config_hash="config-sha256-abc",
            code_commit="commit-0123456789",
            dataset_fold_hash="fold-sha256-def",
            seeds={
                "model_seed": 0,
                "split_seed": 100,
                "partition_seed": 200,
                "audit_seed": 300,
                "traffic_seed": 400,
            },
            retry_limit=retry_limit,
            checkpoint_path=str(self.root / "checkpoints" / f"{job_key}.pt"),
            stdout_path=str(self.root / "logs" / f"{job_key}.stdout.log"),
            stderr_path=str(self.root / "logs" / f"{job_key}.stderr.log"),
            manifest_path=str(self.root / "manifests" / f"{job_key}.manifest.json"),
            heartbeat_path=str(self.root / "heartbeats" / f"{job_key}.json"),
            artifact_schema={
                "format": "json",
                "required_keys": ["schema_version", "metric"],
                "expected_values": {"schema_version": 7},
                "field_types": {"metric": "number"},
                "min_size_bytes": 2,
            },
        )

    def scheduler(self, **kwargs):
        return PersistentLocalScheduler(
            self.queue,
            heartbeat_interval=kwargs.get("heartbeat_interval", 0.02),
            stale_after=kwargs.get("stale_after", 0.05),
            poll_interval=kwargs.get("poll_interval", 0.005),
        )

    def test_success_is_durable_manifested_and_reused_after_restart(self):
        job = self.job()
        first_scheduler = self.scheduler()
        outcome = first_scheduler.run_job(job)
        self.assertEqual(outcome.status, "succeeded")
        self.assertTrue(outcome.artifact_validation.manifest_checked)

        state = first_scheduler.load_state(job.experiment_id)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(state["attempt"], 1)
        self.assertIsNotNone(state["started_at"])
        self.assertIsNotNone(state["ended_at"])
        self.assertEqual(state["spec"]["argv"], list(job.argv))
        self.assertEqual(state["spec"]["seeds"], dict(job.seeds))
        self.assertIn("worker stdout", Path(job.stdout_path).read_text())
        self.assertIn("worker stderr", Path(job.stderr_path).read_text())

        manifest = json.loads(Path(job.manifest_path).read_text())
        self.assertEqual(
            manifest["artifact"]["sha256"], sha256_file(job.expected_artifact)
        )
        self.assertEqual(manifest["spec_hash"], job.spec_hash)
        self.assertEqual(manifest["config_hash"], job.config_hash)
        self.assertEqual(manifest["seeds"], dict(job.seeds))

        restarted = self.scheduler()
        skipped = restarted.run_job(job)
        self.assertEqual(skipped.status, "succeeded")
        self.assertIn("skipped", skipped.message)
        self.assertEqual(restarted.load_state(job.experiment_id)["attempt"], 1)

    def test_retry_uses_the_identical_frozen_argv_config_and_seeds(self):
        counter = self.root / "attempt-counter.txt"
        job = self.job(
            argv=[sys.executable, "-c", RETRY_ONCE, str(self.artifact), str(counter)],
            retry_limit=1,
        )
        scheduler = self.scheduler()
        submitted_spec = job.to_dict()
        outcomes = scheduler.run_until_idle() if False else []
        # Submit explicitly, then exercise the persisted queue path used after restart.
        scheduler.submit(job)
        outcomes = scheduler.run_until_idle()
        self.assertEqual(
            [outcome.status for outcome in outcomes], ["pending", "succeeded"]
        )
        state = scheduler.load_state(job.experiment_id)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(state["attempt"], 2)
        self.assertEqual(len(state["attempts"]), 2)
        self.assertEqual(state["attempts"][0]["returncode"], 23)
        self.assertEqual(state["spec"], submitted_spec)
        self.assertEqual(counter.read_text(), "2")

    def test_same_semantic_id_cannot_change_config_or_seed_on_retry(self):
        scheduler = self.scheduler()
        job = self.job()
        scheduler.submit(job)
        changed = JobSpec.from_dict(
            {
                **job.to_dict(),
                "seeds": {**dict(job.seeds), "audit_seed": 999},
            }
        )
        with self.assertRaises(SpecMismatchError):
            scheduler.submit(changed)

    def test_artifact_validation_is_schema_and_hash_aware(self):
        bad_code = (
            "import json,pathlib,sys; "
            "p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); "
            "p.write_text(json.dumps({'schema_version': 7}))"
        )
        bad_job = self.job(argv=[sys.executable, "-c", bad_code, str(self.artifact)])
        scheduler = self.scheduler()
        outcome = scheduler.run_job(bad_job)
        self.assertEqual(outcome.status, "failed")
        self.assertIn("missing keys", outcome.message)
        self.assertFalse(Path(bad_job.manifest_path).exists())

        # A valid run is invalidated if bytes change without a matching manifest update.
        second_artifact = self.root / "results" / "second.json"
        good_job = JobSpec.from_dict(
            {
                **self.job(experiment_id="good/hash-aware").to_dict(),
                "expected_artifact": str(second_artifact),
                "argv": [sys.executable, "-c", WRITE_JSON, str(second_artifact)],
            }
        )
        self.assertEqual(scheduler.run_job(good_job).status, "succeeded")
        second_artifact.write_text(
            json.dumps({"schema_version": 7, "metric": 999}), encoding="utf-8"
        )
        validation = validate_artifact(good_job, require_manifest=True)
        self.assertFalse(validation.valid)
        self.assertIn("SHA-256", validation.reason)

    def test_stale_running_state_is_requeued_and_orphan_lock_removed(self):
        job = self.job(retry_limit=1)
        scheduler = self.scheduler(stale_after=0.0)
        state = scheduler.submit(job)
        old = "2000-01-01T00:00:00.000Z"
        state.update(
            {
                "status": "running",
                "attempt": 1,
                "started_at": old,
                "heartbeat_at": old,
                "updated_at": old,
                "child_pid": 999_999_999,
            }
        )
        state["attempts"] = [
            {
                "attempt": 1,
                "started_at": old,
                "heartbeat_at": old,
                "ended_at": None,
                "status": "running",
                "child_pid": 999_999_999,
                "returncode": None,
                "error": None,
            }
        ]
        atomic_write_json(scheduler.state_path(job.experiment_id), state)
        lock_path = scheduler.lock_path(job.experiment_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)

        outcomes = scheduler.recover_stale_jobs(now=time.time())
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].status, "pending")
        recovered = scheduler.load_state(job.experiment_id)
        self.assertEqual(recovered["status"], "pending")
        self.assertEqual(recovered["attempt"], 1)
        self.assertFalse(lock_path.exists())
        self.assertIn("stale running job", recovered["last_error"])

        self.assertEqual(scheduler.run_until_idle()[-1].status, "succeeded")
        self.assertEqual(scheduler.load_state(job.experiment_id)["attempt"], 2)

    def test_dry_run_never_executes_and_argv_cannot_be_a_shell_string(self):
        job = self.job()
        scheduler = self.scheduler()
        outcome = scheduler.run_job(job, dry_run=True)
        self.assertEqual(outcome.status, "dry-run")
        self.assertFalse(self.artifact.exists())
        self.assertFalse(Path(job.stdout_path).exists())
        self.assertEqual(scheduler.load_state(job.experiment_id)["status"], "pending")
        with self.assertRaises(TypeError):
            JobSpec(
                experiment_id="unsafe",
                argv="touch should-not-exist",  # type: ignore[arg-type]
                expected_artifact=str(self.artifact),
                config_hash="c",
                code_commit="d",
                dataset_fold_hash="f",
                seeds={},
            )

    def test_lock_is_exclusive_and_owner_safe(self):
        lock_path = self.root / "locks" / "one.lock"
        first = ExclusiveFileLock(lock_path)
        second = ExclusiveFileLock(lock_path)
        first.acquire()
        with self.assertRaises(JobLockedError):
            second.acquire()
        first.release()
        second.acquire()
        second.release()
        self.assertFalse(lock_path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
