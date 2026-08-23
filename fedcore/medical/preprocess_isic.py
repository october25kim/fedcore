"""FLamby's Fed-ISIC2019 input preprocessing pass (resize + colour constancy).

Implements ``data.fed_isic2019.preprocessing`` of the sealed pre-registration
(AMENDMENT A-002)::

    source:           dataset_creation_scripts/resize_images.py
    resize:           shorter side -> 224px, aspect preserved
    colour_constancy: Shades-of-Gray, power=6
    order:            resize -> colour constancy -> (crop) -> Normalize
    cache:            data/isic2019/ISIC_2019_Training_Input_preprocessed/

Provenance: TRANSCRIBED, NOT RECALLED
-------------------------------------
Both routines below were fetched from ``owkin/FLamby`` @ main and transcribed
character-for-character from:

* ``flamby/datasets/fed_isic2019/dataset_creation_scripts/resize_images.py``
  -- ``resize_and_maintain``, and the ``sz = 224`` / ``cc = True`` call site.
* ``flamby/datasets/fed_isic2019/dataset_creation_scripts/color_constancy.py``
  -- ``color_constancy``.

Why this pass exists: FLamby NEVER reads raw ISIC JPEGs.  Its dataset class reads
only ``ISIC_2019_Training_Input_preprocessed``.  On a raw 1024x1024 image the
recipe's ``RandomCrop(200, 200)`` is a ~4%-area patch that usually misses the
lesion; after this pass the shorter side is 224px, so the same crop is the mild
crop FLamby intends.

DECLARED DEVIATIONS from resize_images.py (surfaced, never silently patched)
---------------------------------------------------------------------------
The two transcribed FUNCTIONS are byte-faithful.  The DRIVER around them differs,
in ways that cannot change an output pixel:

1. **No ``dataset_location.yaml``.**  FLamby reads its download-script config for
   the corpus path and writes back ``preprocessing_complete``.  This repo stages
   the corpus itself, so paths are CLI arguments.  No image is affected.
2. **Atomic writes + resume.**  FLamby writes each JPEG in place and refuses to
   re-run wholesale (``if dict["preprocessing_complete"]: sys.exit()``).  Here each
   output is written to a temporary name and ``os.replace``-d into place, so an
   interrupted pass can never leave a truncated JPEG that a resume would mistake
   for finished work.  Existence therefore proves completeness, and ``--resume``
   (default) skips only complete files.  The bytes written are identical.
3. **Explicit ``format="JPEG"``.**  A consequence of (2): PIL infers the encoder
   from the file extension, and the temporary name does not end in ``.jpg``.
   Naming the encoder selects the same plugin with the same defaults (quality=75),
   so this is byte-identical -- and ``tests/test_isic_preprocess.py`` asserts that
   against extension inference rather than asserting it from belief.
4. **``n_jobs`` default 12, not 32.**  A scheduling choice on a 24-core box that is
   concurrently feeding a live GPU campaign's dataloaders.  Order-independent and
   pixel-independent: every image is an independent pure function of its input.
5. **``cv2`` import is lazy.**  FLamby imports cv2 at module top, but uses it ONLY
   in ``color_constancy``'s ``gamma is not None`` branch.  FLamby's own call site
   (``color_constancy(np.array(img))``) leaves ``gamma`` at ``None``, so that branch
   is dead on the executed path.  Importing lazily keeps the module usable where
   opencv is absent while preserving exact semantics; the branch raises loudly if
   ever reached without cv2, rather than degrading silently.

Determinism: neither routine draws a random number.  Each output is a pure
function of one input file, so the corpus is identical regardless of ``n_jobs``,
scheduling order, or how many times an interrupted pass is resumed.

Run::

    python -m fedcore.medical.preprocess_isic \\
        --input-folder data/isic2019/ISIC_2019_Training_Input \\
        --output-folder data/isic2019/ISIC_2019_Training_Input_preprocessed
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from typing import Sequence

import numpy as np
from PIL import Image


#: ``resize_images.py``: ``sz = 224`` -- the SHORTER edge of the resized image.
FLAMBY_RESIZE_SHORTER_SIDE: int = 224

#: ``color_constancy.py``: ``def color_constancy(img, power=6, gamma=None)``.
FLAMBY_COLOUR_CONSTANCY_POWER: int = 6

#: ``resize_images.py``: ``cc = True`` at the call site.
FLAMBY_COLOUR_CONSTANCY_ENABLED: bool = True

#: FLamby's canonical output directory name, and the prereg's declared cache.
FLAMBY_PREPROCESSED_DIRNAME: str = "ISIC_2019_Training_Input_preprocessed"


def color_constancy(img, power=6, gamma=None):
    """Shades-of-Gray colour constancy.

    TRANSCRIBED VERBATIM from ``flamby/datasets/fed_isic2019/
    dataset_creation_scripts/color_constancy.py``.  The only edit is the lazy
    ``cv2`` import inside the ``gamma`` branch (deviation 5 in the module
    docstring); FLamby's own call site never supplies a gamma, so this branch does
    not execute in the pre-registered pass.

    Preprocessing step to make sure that the images appear with similar brightness
    and contrast.

    Parameters
    ----------
    img: 3D numpy array, the original image
    power: int, degree of norm
    gamma: float, value of gamma correction
    """
    img_dtype = img.dtype

    if gamma is not None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - dead on the FLamby path
            raise ImportError(
                "color_constancy(gamma=...) needs opencv (cv2), which FLamby "
                "imports at module top. FLamby's own resize_images.py call site "
                "passes gamma=None, so the pre-registered pass never reaches this "
                "branch; install opencv only if you intend to deviate."
            ) from exc
        img = img.astype("uint8")
        look_up_table = np.ones((256, 1), dtype="uint8") * 0
        for i in range(256):
            look_up_table[i][0] = 255 * pow(i / 255, 1 / gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype("float32")
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0, 1)), 1 / power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec / rgb_norm
    rgb_vec = 1 / (rgb_vec * np.sqrt(3))
    img = np.multiply(img, rgb_vec)

    return img.astype(img_dtype)


def resize_and_maintain(path, output_path, sz: tuple, cc):
    """Preprocessing of images.

    TRANSCRIBED VERBATIM from ``flamby/datasets/fed_isic2019/
    dataset_creation_scripts/resize_images.py``, except that the final ``img.save``
    is made atomic (deviation 2/3 in the module docstring).  The resize arithmetic,
    the BILINEAR resample, and the colour-constancy call are unchanged.

    Mantains aspect ratio fo input image. Possibility to add color constancy.

    Parameters
    ----------
    path : path to input image
    output_path : path to output image
    sz : tuple, shorter edge of resized image is sz[0]
    cc : color constancy is added if True
    """
    fn = os.path.basename(path)
    img = Image.open(path)
    size = sz[0]
    old_size = img.size
    ratio = float(size) / min(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, resample=Image.BILINEAR)
    if cc:
        img = color_constancy(np.array(img))
        img = Image.fromarray(img)
    # FLamby: img.save(os.path.join(output_path, fn)). Written atomically so an
    # interrupted pass cannot leave a truncated JPEG that --resume would skip.
    final = os.path.join(output_path, fn)
    tmp = f"{final}.tmp.{os.getpid()}"
    img.save(tmp, format="JPEG")
    os.replace(tmp, final)


def _process_one(path: str, output_folder: str, sz: int, cc: bool, resume: bool) -> str:
    final = os.path.join(output_folder, os.path.basename(path))
    if resume and os.path.isfile(final):
        return "skipped"
    try:
        resize_and_maintain(path, output_folder, (sz, sz), cc)
    except Exception as exc:  # surfaced by the driver; never silently swallowed
        return f"FAILED {os.path.basename(path)}: {type(exc).__name__}: {exc}"
    return "written"


def preprocess_corpus(
    input_folder: str,
    output_folder: str,
    *,
    sz: int = FLAMBY_RESIZE_SHORTER_SIDE,
    cc: bool = FLAMBY_COLOUR_CONSTANCY_ENABLED,
    n_jobs: int = 12,
    resume: bool = True,
    limit: int | None = None,
) -> dict:
    """Run the pass over every ``*.jpg`` in ``input_folder``. Deterministic.

    Globs exactly as ``resize_images.py`` does (``*.jpg`` in the input folder), so
    the cache mirrors FLamby's.  That is a superset of the 23,247 centre-attributable
    images: ISIC-2019 ships 25,331 JPEGs and FLamby drops the 2,084 null-``lesion_id``
    images at metadata-join time, not here.
    """
    from joblib import Parallel, delayed

    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"input folder does not exist: {input_folder}")
    os.makedirs(output_folder, exist_ok=True)

    # Sorted: the transform is order-independent, but a stable order makes progress
    # reports and any partial-state diagnosis reproducible.
    images = sorted(glob.glob(os.path.join(input_folder, "*.jpg")))
    if limit is not None:
        images = images[: int(limit)]
    if not images:
        raise FileNotFoundError(f"no *.jpg found under {input_folder}")

    started = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process_one)(path, output_folder, sz, cc, resume) for path in images
    )
    elapsed = time.time() - started

    failures = [r for r in results if r.startswith("FAILED")]
    report = {
        "input_folder": input_folder,
        "output_folder": output_folder,
        "resize_shorter_side": int(sz),
        "colour_constancy": bool(cc),
        "colour_constancy_power": FLAMBY_COLOUR_CONSTANCY_POWER,
        "n_jobs": int(n_jobs),
        "n_found": len(images),
        "n_written": sum(1 for r in results if r == "written"),
        "n_skipped_existing": sum(1 for r in results if r == "skipped"),
        "n_failed": len(failures),
        "failures": failures[:20],
        "wall_seconds": round(elapsed, 1),
    }
    if failures:
        raise RuntimeError(
            f"{len(failures)} image(s) failed to preprocess; first: {failures[:3]}"
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-folder", default="data/isic2019/ISIC_2019_Training_Input"
    )
    parser.add_argument(
        "--output-folder",
        default=os.path.join("data", "isic2019", FLAMBY_PREPROCESSED_DIRNAME),
    )
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="re-write every image even if a complete output already exists",
    )
    parser.add_argument("--limit", type=int, default=None, help="debug: first N images")
    args = parser.parse_args(argv)

    report = preprocess_corpus(
        args.input_folder,
        args.output_folder,
        n_jobs=args.n_jobs,
        resume=not args.no_resume,
        limit=args.limit,
    )
    import json

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


__all__ = [
    "FLAMBY_COLOUR_CONSTANCY_ENABLED",
    "FLAMBY_COLOUR_CONSTANCY_POWER",
    "FLAMBY_PREPROCESSED_DIRNAME",
    "FLAMBY_RESIZE_SHORTER_SIDE",
    "color_constancy",
    "preprocess_corpus",
    "resize_and_maintain",
]


if __name__ == "__main__":
    raise SystemExit(main())
