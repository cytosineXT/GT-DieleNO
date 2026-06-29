"""Validation and reshaping helpers for CST dense far-field RCS CSV files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .paper_settings import OBSERVATION_GRID


def read_farfield_csv(path: str | Path) -> list[dict[str, float]]:
    """Read a CST dense RCS CSV exported by :mod:`cstpy.cst_macros`."""

    rows: list[dict[str, float]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"theta_deg", "phi_deg", "frequency_ghz", "rcs_m2"}.issubset(reader.fieldnames):
            raise ValueError(f"{path} does not contain the expected far-field RCS columns")
        for row in reader:
            rows.append(
                {
                    "theta_deg": float(row["theta_deg"]),
                    "phi_deg": float(row["phi_deg"]),
                    "frequency_ghz": float(row["frequency_ghz"]),
                    "rcs_m2": float(row["rcs_m2"]),
                }
            )
    return rows


def validate_paper_grid(rows: list[dict[str, float]]) -> dict[str, object]:
    """Check that rows match the paper's 360 x 720 observation grid."""

    grid = OBSERVATION_GRID
    if len(rows) != grid.sample_count:
        raise ValueError(f"expected {grid.sample_count} rows, got {len(rows)}")

    theta_values = grid.theta_values()
    phi_values = grid.phi_values()
    for idx, row in enumerate(rows):
        theta_idx = idx // grid.phi_count
        phi_idx = idx % grid.phi_count
        theta = round(row["theta_deg"], 10)
        phi = round(row["phi_deg"], 10)
        if theta != theta_values[theta_idx] or phi != phi_values[phi_idx]:
            raise ValueError(
                f"grid mismatch at row {idx}: got ({theta}, {phi}), "
                f"expected ({theta_values[theta_idx]}, {phi_values[phi_idx]})"
            )
        if row["rcs_m2"] < 0:
            raise ValueError(f"negative linear RCS at row {idx}: {row['rcs_m2']}")

    frequencies = sorted({row["frequency_ghz"] for row in rows})
    return {
        "shape": grid.shape,
        "row_count": len(rows),
        "frequency_ghz": frequencies,
        "rcs_m2_min": min(row["rcs_m2"] for row in rows),
        "rcs_m2_max": max(row["rcs_m2"] for row in rows),
    }


def write_grid_json(path: str | Path, rows: list[dict[str, float]]) -> None:
    """Write a lightweight JSON representation of the dense linear-RCS grid."""

    grid = OBSERVATION_GRID
    validate_paper_grid(rows)
    values = []
    for theta_idx in range(grid.theta_count):
        start = theta_idx * grid.phi_count
        stop = start + grid.phi_count
        values.append([row["rcs_m2"] for row in rows[start:stop]])
    payload = {
        "shape": grid.shape,
        "theta_deg": grid.theta_values(),
        "phi_deg": grid.phi_values(),
        "frequency_ghz": rows[0]["frequency_ghz"] if rows else None,
        "rcs_m2": values,
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize_csv(path: str | Path) -> dict[str, object]:
    """Read and validate one CSV, returning compact metadata."""

    rows = read_farfield_csv(path)
    summary = validate_paper_grid(rows)
    summary["csv"] = str(path)
    return summary
