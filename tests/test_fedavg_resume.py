"""FedAvg checkpoint/restart exact replay test (requires torch)."""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_interrupted_resume_matches_uninterrupted():
    try:
        import torch
        from torch import nn
        from torch.utils.data import TensorDataset
    except ImportError:
        print("fedavg resume test: SKIP (torch unavailable)")
        return
    from fedcore.models.fed_train import fedavg

    x = torch.arange(80, dtype=torch.float32).reshape(40, 2) / 80.0
    y = (torch.arange(40) % 2).long()
    clients = [TensorDataset(x[:20], y[:20]), TensorDataset(x[20:], y[20:])]

    def make_model():
        return nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))

    kwargs = dict(
        make_model_fn=make_model,
        client_datasets=clients,
        rounds=3,
        local_epochs=1,
        lr=0.02,
        batch_size=5,
        device="cpu",
        loader_seed=777,
        checkpoint_every=1,
        checkpoint_metadata={"training_config_sha256": "fixture"},
    )
    with tempfile.TemporaryDirectory() as td:
        torch.manual_seed(123)
        np.random.seed(123)
        full = fedavg(**kwargs, checkpoint_path=os.path.join(td, "full.pt"))

        class ExpectedInterruption(RuntimeError):
            pass

        def interrupt(round_index):
            if round_index == 0:
                raise ExpectedInterruption

        resume_path = os.path.join(td, "resume.pt")
        torch.manual_seed(123)
        np.random.seed(123)
        try:
            fedavg(**kwargs, checkpoint_path=resume_path, round_end_callback=interrupt)
        except ExpectedInterruption:
            pass
        else:
            raise AssertionError("fixture interruption did not occur")
        resumed = fedavg(**kwargs, checkpoint_path=resume_path, resume=True)
        for name, tensor in full.state_dict().items():
            assert torch.equal(tensor, resumed.state_dict()[name]), name


def main():
    test_interrupted_resume_matches_uninterrupted()
    print("fedavg resume tests: PASS")


if __name__ == "__main__":
    main()
