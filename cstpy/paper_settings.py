"""Simulation settings that mirror the GT-DieleNO paper."""

from __future__ import annotations

from dataclasses import dataclass

CST_VERSION = "CST Studio Suite 2025"

UNITS = {
    "geometry": "m",
    "frequency": "GHz",
    "time": "ns",
}

SOLVER_SETTINGS = {
    "solver": "time_domain",
    "boundary": "expanded open",
    "background": "Normal",
    "mesh_adaptation": False,
    "prepared_farfields": True,
    "steady_state_limit_db": -30,
    "polarization": "HH",
    "max_bounding_box_m": 1.0,
}

INCIDENCES_DEG = (
    (90.0, 0.0),
    (60.0, 45.0),
    (120.0, 45.0),
)


@dataclass(frozen=True)
class ObservationGrid:
    theta_min_deg: float = 0.0
    theta_max_deg: float = 179.5
    phi_min_deg: float = 0.0
    phi_max_deg: float = 359.5
    theta_step_deg: float = 0.5
    phi_step_deg: float = 0.5

    @property
    def theta_count(self) -> int:
        return int(round((self.theta_max_deg - self.theta_min_deg) / self.theta_step_deg)) + 1

    @property
    def phi_count(self) -> int:
        return int(round((self.phi_max_deg - self.phi_min_deg) / self.phi_step_deg)) + 1

    @property
    def shape(self) -> tuple[int, int]:
        return self.theta_count, self.phi_count

    @property
    def sample_count(self) -> int:
        return self.theta_count * self.phi_count

    def theta_values(self) -> list[float]:
        return [round(self.theta_min_deg + idx * self.theta_step_deg, 10) for idx in range(self.theta_count)]

    def phi_values(self) -> list[float]:
        return [round(self.phi_min_deg + idx * self.phi_step_deg, 10) for idx in range(self.phi_count)]


OBSERVATION_GRID = ObservationGrid()

FREQUENCY_FULL_GHZ = tuple(round(0.10 + 0.01 * idx, 2) for idx in range(91))
FREQUENCY_SCALING_GHZ = tuple(round(0.10 + 0.10 * idx, 2) for idx in range(10))


def frequency_tag(freq_ghz: float) -> str:
    """Return a stable filename token for a GHz frequency."""

    return f"{freq_ghz:g}".replace(".", "p")


def sample_id(
    object_code: str,
    material_design: str,
    theta_inc_deg: float,
    phi_inc_deg: float,
    freq_ghz: float,
) -> str:
    """Return the paper's supervised-sample key as a compact filename token."""

    return (
        f"{object_code}_{material_design}_"
        f"ti{theta_inc_deg:g}_pi{phi_inc_deg:g}_f{frequency_tag(freq_ghz)}"
    )
