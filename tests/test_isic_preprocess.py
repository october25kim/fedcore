"""The FLamby preprocessing pass (prereg data.fed_isic2019.preprocessing, A-002).

These tests pin the TRANSCRIPTION, not a reimplementation: the point of the
pre-registration's "verify the exact recipe from the FLamby repo ... do not trust
memory" is that a plausible-looking rewrite silently diverges.  So they assert the
arithmetic FLamby actually performs, including its float-truncation quirk.

Synthetic images only -- no ISIC data is required to run this file.
"""

from __future__ import annotations

import collections
import glob
import os

import numpy as np
import pytest
from PIL import Image

from fedcore.medical.preprocess_isic import (
    FLAMBY_COLOUR_CONSTANCY_POWER,
    FLAMBY_RESIZE_SHORTER_SIDE,
    color_constancy,
    preprocess_corpus,
    resize_and_maintain,
)


def _write_image(path: str, size=(400, 300), colour=(200, 120, 60)) -> None:
    array = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    array[:, :] = colour
    # A little structure, so resizing is not a no-op on a constant field.
    array[: size[1] // 2, : size[0] // 2] = (30, 220, 90)
    Image.fromarray(array).save(path)


def test_colour_constancy_neutralises_the_channel_means():
    """Shades-of-Gray's defining behaviour: it equalises the per-channel norms.

    A strong colour cast must come out closer to neutral. This asserts the
    TRANSCRIBED formula does what Shades-of-Gray is supposed to do, rather than
    asserting it merely runs.
    """
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(64, 64, 3)).astype(np.uint8)
    image[:, :, 2] = np.clip(image[:, :, 2].astype(int) + 60, 0, 255)  # blue cast

    out = color_constancy(image)

    assert out.dtype == image.dtype
    before = image.astype(float).mean((0, 1))
    after = out.astype(float).mean((0, 1))
    assert float(after.std()) < float(before.std())


def test_colour_constancy_changes_pixels_and_is_deterministic():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, size=(32, 48, 3)).astype(np.uint8)

    first = color_constancy(image)
    second = color_constancy(image)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, image), "colour constancy must alter pixels"


def test_colour_constancy_power_matches_flamby_default():
    """``power`` is the prereg's declared 6, and it is not an inert argument."""
    assert FLAMBY_COLOUR_CONSTANCY_POWER == 6
    rng = np.random.default_rng(2)
    image = rng.integers(0, 255, size=(32, 32, 3)).astype(np.uint8)

    assert np.array_equal(color_constancy(image), color_constancy(image, power=6))
    assert not np.array_equal(color_constancy(image, power=1), color_constancy(image))


def test_resize_keeps_aspect_ratio_and_targets_the_shorter_side(tmp_path):
    src = tmp_path / "ISIC_TEST.jpg"
    out = tmp_path / "out"
    out.mkdir()
    _write_image(str(src), size=(800, 400))

    resize_and_maintain(str(src), str(out), (FLAMBY_RESIZE_SHORTER_SIDE,) * 2, True)

    width, height = Image.open(out / "ISIC_TEST.jpg").size
    # 800x400 -> ratio 224/400 = 0.56 exactly -> 448x224, no truncation loss.
    assert (width, height) == (448, 224)
    assert min(width, height) == FLAMBY_RESIZE_SHORTER_SIDE


def test_flamby_int_truncation_can_yield_223_not_224(tmp_path):
    """FLamby's ``int(x * ratio)`` truncates, so the shorter side is NOT always 224.

    On a 1022x767 ISIC raw, ``767 * (224/767)`` is 223.99999999999997 in float64 and
    ``int()`` floors it to 223. This is FLamby's real behaviour and the transcription
    must reproduce it; a from-memory rewrite would "fix" it to 224 and diverge from
    the published corpus. 223 >= 200 so the declared crop is unaffected.
    """
    src = tmp_path / "ISIC_RAW.jpg"
    out = tmp_path / "out"
    out.mkdir()
    _write_image(str(src), size=(1022, 767))

    resize_and_maintain(str(src), str(out), (224, 224), True)

    width, height = Image.open(out / "ISIC_RAW.jpg").size
    assert (width, height) == (298, 223)
    assert min(width, height) >= 200, "the declared 200px crop must remain feasible"


# --------------------------------------------------------------------------- #
# Truncation census (preprocessing fidelity, pre-launch reconciliation item 9)
# --------------------------------------------------------------------------- #
#
# CORRECTION OF RECORD. The Fed-ISIC pre-launch brief asserted that FIVE images
# suffer FLamby's ``int(x * ratio)`` truncation. The real count is 179, established
# twice by independent routes that agree exactly:
#   (a) replaying FLamby's arithmetic on the raw ISIC_2019_Training_Input headers;
#   (b) measuring min(side) on the already-written preprocessed corpus.
# There is no five-image subset; the victims fall into SEVEN raw-size classes, and
# the smallest class has one member, so "five" does not correspond to any natural
# grouping either. These constants pin the true census against silent drift.
#
# Every victim lands on 223, never lower, so 223 >= 200 and the pre-registered
# CenterCrop(200,200) stays feasible for all of them. The quirk is cosmetic for the
# model but load-bearing for TRANSCRIPTION FIDELITY: a from-memory rewrite would
# round to 224 and silently diverge from FLamby's published corpus.

#: (raw_size, truncated_new_size) -> number of ISIC-2019 raws in that class.
FLAMBY_TRUNCATION_SIZE_CLASSES = {
    ((919, 802), (256, 223)): 54,
    ((962, 722), (298, 223)): 3,
    ((963, 629), (342, 223)): 1,
    ((1022, 767), (298, 223)): 22,
    ((1024, 686), (334, 223)): 17,
    ((1024, 690), (332, 223)): 1,
    ((1024, 764), (300, 223)): 81,
}
FLAMBY_TRUNCATION_TOTAL = 179
#: sha256 of the 179 sorted image IDs joined by "\n".
FLAMBY_TRUNCATION_ID_SHA256 = (
    "f09d2051d597ea4b39dba0d4e2efc19d9d31504123ab8fb76f6297ba756abf81"
)

_RAW_DIR = "data/isic2019/ISIC_2019_Training_Input"
_PREPROC_DIR = "data/isic2019/ISIC_2019_Training_Input_preprocessed"


def _flamby_new_size(old_size, target: int = 224):
    """FLamby's exact resize arithmetic, transcribed from resize_images.py."""
    ratio = float(target) / min(old_size)
    return tuple(int(x * ratio) for x in old_size)


def test_truncation_size_classes_are_exactly_the_flamby_arithmetic():
    """Data-free: every pinned class reproduces FLamby's int() truncation.

    This is the regression that matters even on a machine with no ISIC corpus --
    it pins the ARITHMETIC, which is what a rewrite would break.
    """
    for (old_size, expected_new), count in FLAMBY_TRUNCATION_SIZE_CLASSES.items():
        assert _flamby_new_size(old_size) == expected_new, old_size
        assert min(expected_new) == 223, "FLamby truncates to 223, never below"
        assert min(expected_new) >= 200, "the declared 200px crop must stay feasible"
        assert count >= 1
    assert sum(FLAMBY_TRUNCATION_SIZE_CLASSES.values()) == FLAMBY_TRUNCATION_TOTAL
    assert len(FLAMBY_TRUNCATION_SIZE_CLASSES) == 7, "seven raw-size classes, not five"


def test_a_non_truncating_size_is_not_swept_into_the_census():
    """Guards the census against being trivially true for every image."""
    # 1024x768: 768 * (224/768) == 224.0 exactly -> no truncation.
    assert _flamby_new_size((1024, 768)) == (298, 224)
    assert ((1024, 768), (298, 224)) not in FLAMBY_TRUNCATION_SIZE_CLASSES


@pytest.mark.skipif(
    not os.path.isdir(_RAW_DIR), reason="ISIC-2019 raw corpus not present"
)
def test_raw_corpus_truncation_census_is_exactly_179_images():
    """Replays FLamby's arithmetic over the real raws and pins the ID set."""
    import hashlib

    victims = []
    for path in sorted(glob.glob(os.path.join(_RAW_DIR, "*.jpg"))):
        with Image.open(path) as img:
            old_size = img.size
        if min(_flamby_new_size(old_size)) != 224:
            victims.append(os.path.basename(path)[: -len(".jpg")])

    assert len(victims) == FLAMBY_TRUNCATION_TOTAL, (
        f"the brief's 'five images' is wrong; measured {len(victims)}"
    )
    digest = hashlib.sha256("\n".join(sorted(victims)).encode()).hexdigest()
    assert digest == FLAMBY_TRUNCATION_ID_SHA256


@pytest.mark.skipif(
    not os.path.isdir(_PREPROC_DIR), reason="preprocessed ISIC corpus not present"
)
def test_processed_corpus_agrees_with_the_raw_derived_census():
    """Independent route: measure the WRITTEN corpus, not the arithmetic.

    Agreement between this and the raw-derived census is what makes 179 a
    measurement rather than a prediction.
    """
    short_sides = collections.Counter()
    for path in glob.glob(os.path.join(_PREPROC_DIR, "*.jpg")):
        with Image.open(path) as img:
            short_sides[min(img.size)] += 1
    assert short_sides[223] == FLAMBY_TRUNCATION_TOTAL
    assert set(short_sides) == {223, 224}, "no side other than 223/224 may appear"


def test_corpus_pass_is_resumable_and_deterministic(tmp_path):
    src = tmp_path / "in"
    out = tmp_path / "out"
    src.mkdir()
    for i in range(4):
        _write_image(str(src / f"ISIC_{i:03d}.jpg"), colour=(10 + 40 * i, 90, 200))

    first = preprocess_corpus(str(src), str(out), n_jobs=1)
    assert first["n_written"] == 4 and first["n_skipped_existing"] == 0
    digests = {p.name: p.read_bytes() for p in out.glob("*.jpg")}

    second = preprocess_corpus(str(src), str(out), n_jobs=1)
    assert second["n_written"] == 0, "a resumed pass must not rewrite complete files"
    assert second["n_skipped_existing"] == 4

    third = preprocess_corpus(str(src), str(out), n_jobs=1, resume=False)
    assert third["n_written"] == 4, "--no-resume must actually rewrite"
    assert {p.name: p.read_bytes() for p in out.glob("*.jpg")} == digests, (
        "recomputing must be byte-identical: the pass has no RNG"
    )


def test_parallelism_does_not_change_the_corpus(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    for i in range(6):
        _write_image(str(src / f"ISIC_{i:03d}.jpg"), colour=(20 * i, 200 - 20 * i, 77))

    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    preprocess_corpus(str(src), str(serial), n_jobs=1)
    preprocess_corpus(str(src), str(parallel), n_jobs=4)

    for path in sorted(serial.glob("*.jpg")):
        assert path.read_bytes() == (parallel / path.name).read_bytes()


def test_no_temporary_files_survive_a_successful_pass(tmp_path):
    """Atomicity: existence of an output must prove completeness, for --resume."""
    src = tmp_path / "in"
    out = tmp_path / "out"
    src.mkdir()
    _write_image(str(src / "ISIC_000.jpg"))

    preprocess_corpus(str(src), str(out), n_jobs=1)

    assert list(out.glob("*.jpg"))
    assert not [p for p in out.iterdir() if ".tmp." in p.name]


def test_missing_input_folder_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        preprocess_corpus(str(tmp_path / "nope"), str(tmp_path / "out"))


def test_empty_input_folder_fails_closed(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    with pytest.raises(FileNotFoundError):
        preprocess_corpus(str(src), str(tmp_path / "out"))
