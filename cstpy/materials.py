"""Dielectric material library and d00--d09 design definitions."""

from __future__ import annotations

REGIONS = ("body", "left_wing", "right_wing", "tail")

MATERIAL_LIBRARY = {
    eps: {
        "name": f"DIEL_EPS{eps}",
        "epsilon_r": float(eps),
        "mu_r": 1.0,
        "tan_delta_e": 0.01,
        "tan_delta_m": 0.0,
    }
    for eps in range(2, 7)
}

MATERIAL_DESIGNS = {
    "d00": {"body": 2, "left_wing": 2, "right_wing": 2, "tail": 2},
    "d01": {"body": 3, "left_wing": 3, "right_wing": 3, "tail": 3},
    "d02": {"body": 4, "left_wing": 4, "right_wing": 4, "tail": 4},
    "d03": {"body": 5, "left_wing": 5, "right_wing": 5, "tail": 5},
    "d04": {"body": 6, "left_wing": 6, "right_wing": 6, "tail": 6},
    "d05": {"body": 2, "left_wing": 6, "right_wing": 6, "tail": 6},
    "d06": {"body": 6, "left_wing": 2, "right_wing": 2, "tail": 2},
    "d07": {"body": 3, "left_wing": 5, "right_wing": 5, "tail": 4},
    "d08": {"body": 3, "left_wing": 2, "right_wing": 6, "tail": 4},
    "d09": {"body": 4, "left_wing": 6, "right_wing": 6, "tail": 2},
}


def design_epsilons(design_id: str) -> dict[str, int]:
    """Return a region-to-epsilon map for one material design."""

    try:
        return dict(MATERIAL_DESIGNS[design_id])
    except KeyError as exc:
        raise KeyError(f"unknown material design {design_id!r}") from exc


def epsilon_for_region(design_id: str, region: str) -> int:
    """Return the relative permittivity assigned to a named region."""

    if region not in REGIONS:
        raise KeyError(f"unknown material region {region!r}; expected one of {REGIONS}")
    return design_epsilons(design_id)[region]
