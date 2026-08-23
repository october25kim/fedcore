"""Emit explicit zero-row final assets when the governing plan is unavailable."""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Sequence

from fedcore.campaign.finalize import finalize_unplanned_blocked_state


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocked-state", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    result = finalize_unplanned_blocked_state(args.blocked_state, args.out_dir)
    print(
        json.dumps(
            {
                "status": "blocked",
                "manuscript_ready": result.report["manuscript_ready"],
                "observed_scientific_rows": len(result.rows),
                "out_dir": os.path.abspath(args.out_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
