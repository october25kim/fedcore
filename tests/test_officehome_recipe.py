"""Office-Home recipe tests: cosine-warmup LR schedule (torch-free part)."""

from __future__ import annotations

import math

import pytest

from fedcore.models.officehome_train import cosine_warmup_lr


def test_warmup_ramps_then_cosine_decays():
    base, total, warmup = 1e-4, 30, 2
    lrs = [cosine_warmup_lr(r, base, total, warmup) for r in range(total)]
    # Warmup: strictly increasing to base at the end of warmup.
    assert lrs[0] == pytest.approx(base * 0.5)
    assert lrs[1] == pytest.approx(base)
    # Post-warmup cosine decay: non-increasing, ending near 0.
    for i in range(warmup, total - 1):
        assert lrs[i] >= lrs[i + 1] - 1e-18
    assert lrs[-1] == pytest.approx(base * 0.5 * (1 + math.cos(math.pi * (total - 1 - warmup) / (total - warmup))))
    assert lrs[-1] < base


def test_schedule_is_pure_function_of_round():
    # Determinism is what makes --resume exact.
    for r in range(30):
        assert cosine_warmup_lr(r, 1e-3, 30, 2) == cosine_warmup_lr(r, 1e-3, 30, 2)


def test_smoke_two_round_schedule():
    # For the 2-round smoke, both rounds are warmup.
    lrs = [cosine_warmup_lr(r, 1e-4, 2, 2) for r in range(2)]
    assert lrs[0] == pytest.approx(5e-5)
    assert lrs[1] == pytest.approx(1e-4)


def test_out_of_range_round_fails_closed():
    with pytest.raises(ValueError):
        cosine_warmup_lr(30, 1e-4, 30, 2)
    with pytest.raises(ValueError):
        cosine_warmup_lr(-1, 1e-4, 30, 2)
