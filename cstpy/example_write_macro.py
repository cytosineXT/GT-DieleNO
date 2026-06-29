"""Write paper-aligned CST VBA helper macros to disk."""

from __future__ import annotations

import argparse
from pathlib import Path

from .cst_macros import farfield_export_macro, paper_project_preamble_macro
from .paper_settings import FREQUENCY_FULL_GHZ, INCIDENCES_DEG


def main() -> int:
    parser = argparse.ArgumentParser(description="Write GT-DieleNO paper-aligned CST macro snippets.")
    parser.add_argument("--out", type=Path, default=Path("cstpy_macros"))
    parser.add_argument("--theta-inc-deg", type=float, default=INCIDENCES_DEG[0][0])
    parser.add_argument("--phi-inc-deg", type=float, default=INCIDENCES_DEG[0][1])
    parser.add_argument("--export-dir", type=Path, default=Path("farfield_rcs"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "paper_project_preamble.bas").write_text(
        paper_project_preamble_macro(args.theta_inc_deg, args.phi_inc_deg),
        encoding="utf-8",
    )
    (args.out / "dense_farfield_rcs_export.bas").write_text(
        farfield_export_macro(args.export_dir, list(FREQUENCY_FULL_GHZ)),
        encoding="utf-8",
    )
    print(f"Wrote CST macro snippets to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
