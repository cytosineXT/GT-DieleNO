"""Paper-aligned CST helper utilities for GT-DieleNO.

The package exposes the simulation constants, material library, CST VBA macro
templates, and far-field CSV checks used to document the CST side of the paper.
"""

from .paper_settings import (
    CST_VERSION,
    FREQUENCY_FULL_GHZ,
    FREQUENCY_SCALING_GHZ,
    INCIDENCES_DEG,
    OBSERVATION_GRID,
    SOLVER_SETTINGS,
)
from .materials import MATERIAL_DESIGNS, MATERIAL_LIBRARY

__all__ = [
    "CST_VERSION",
    "FREQUENCY_FULL_GHZ",
    "FREQUENCY_SCALING_GHZ",
    "INCIDENCES_DEG",
    "MATERIAL_DESIGNS",
    "MATERIAL_LIBRARY",
    "OBSERVATION_GRID",
    "SOLVER_SETTINGS",
]
