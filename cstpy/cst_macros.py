"""CST VBA macro snippets aligned with the GT-DieleNO paper settings."""

from __future__ import annotations

import math
from pathlib import Path

from .materials import MATERIAL_LIBRARY
from .paper_settings import OBSERVATION_GRID, SOLVER_SETTINGS


def vba_path(path: str | Path) -> str:
    return str(path).replace("\\", "\\\\").replace('"', '""')


def units_macro() -> str:
    return """
With Units
    .Geometry "m"
    .Frequency "GHz"
    .Time "ns"
End With
"""


def background_and_boundaries_macro() -> str:
    return """
With Background
    .Type "Normal"
End With

With Boundary
    .Xmin "expanded open"
    .Xmax "expanded open"
    .Ymin "expanded open"
    .Ymax "expanded open"
    .Zmin "expanded open"
    .Zmax "expanded open"
End With
"""


def material_macro(epsilon_r: int, color: tuple[float, float, float] = (0.4, 0.6, 0.9)) -> str:
    mat = MATERIAL_LIBRARY[int(epsilon_r)]
    r, g, b = color
    return f"""
With Material
    .Reset
    .Name "{mat['name']}"
    .Type "Normal"
    .Epsilon "{mat['epsilon_r']}"
    .Mue "{mat['mu_r']}"
    .TanD "{mat['tan_delta_e']}"
    .TanDM "{mat['tan_delta_m']}"
    .Colour "{r}", "{g}", "{b}"
    .Create
End With
"""


def define_all_materials_macro() -> str:
    colors = {
        2: (0.20, 0.55, 0.95),
        3: (0.25, 0.70, 0.45),
        4: (0.78, 0.58, 0.18),
        5: (0.58, 0.46, 0.82),
        6: (0.95, 0.45, 0.15),
    }
    return "\n".join(material_macro(eps, colors[eps]) for eps in range(2, 7))


def _unit_vector(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(value * value for value in vec))
    if norm <= 1e-12:
        raise ValueError("zero-length vector")
    return tuple(value / norm for value in vec)


def hh_plane_wave_vectors(theta_deg: float, phi_deg: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return propagation and HH electric-field unit vectors.

    The HH vector is represented by projecting a global z-directed reference
    field onto the plane transverse to the propagation direction. If this
    reference is nearly parallel to the propagation direction, a y-directed
    fallback is used.
    """

    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    normal = (
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    )
    reference = (0.0, 0.0, 1.0)
    dot = sum(normal[idx] * reference[idx] for idx in range(3))
    e_vec = tuple(reference[idx] - dot * normal[idx] for idx in range(3))
    if sum(value * value for value in e_vec) < 1e-12:
        reference = (0.0, 1.0, 0.0)
        dot = sum(normal[idx] * reference[idx] for idx in range(3))
        e_vec = tuple(reference[idx] - dot * normal[idx] for idx in range(3))
    return _unit_vector(normal), _unit_vector(e_vec)


def plane_wave_hh_macro(theta_deg: float, phi_deg: float) -> str:
    normal, e_vec = hh_plane_wave_vectors(theta_deg, phi_deg)
    return f"""
With PlaneWave
    .Reset
    .Normal ({normal[0]:.12g}, {normal[1]:.12g}, {normal[2]:.12g})
    .EVector ({e_vec[0]:.12g}, {e_vec[1]:.12g}, {e_vec[2]:.12g})
    .Polarization ("Linear")
    .ReferenceFrequency (0.0)
    .SetUserDecouplingPlane (False)
    .Store
End With

With Solver
    .ResetExcitationList
    .ActivateExcitation "planewave", 1, 1, True
End With
"""


def frequency_range_macro(fmin_ghz: float = 0.1, fmax_ghz: float = 1.0) -> str:
    return f"""
Solver.FrequencyRange "{fmin_ghz:g}", "{fmax_ghz:g}"
"""


def farfield_monitor_macro(name: str, frequency_ghz: float) -> str:
    return f"""
With Monitor
    .Reset
    .Name "{name}_{frequency_ghz:g}GHz"
    .Domain "Frequency"
    .FieldType "Farfield"
    .Frequency "{frequency_ghz:g}"
    .UseSubvolume "False"
    .EnableNearfieldCalculation "True"
    .Create
End With
"""


def solver_settings_macro() -> str:
    steady_state = SOLVER_SETTINGS["steady_state_limit_db"]
    return f"""
With Solver
    .MeshAdaption False
    .PrepareFarfields True
    .SteadyStateLimit "{steady_state}"
End With
"""


def import_stl_macro(stl_path: str | Path, component: str, solid_name: str, material_name: str) -> str:
    return f"""
With STL
    .Reset
    .FileName ("{vba_path(stl_path)}")
    .Name ("{solid_name}")
    .Component ("{component}")
    .ImportToActiveCoordinateSystem (False)
    .Read
End With

Solid.ChangeMaterial "{component}:{solid_name}", "{material_name}"
"""


def paper_project_preamble_macro(theta_inc_deg: float, phi_inc_deg: float) -> str:
    """Return CST History/VBA text for paper-level global settings."""

    return "\n".join(
        [
            units_macro(),
            background_and_boundaries_macro(),
            define_all_materials_macro(),
            frequency_range_macro(0.1, 1.0),
            plane_wave_hh_macro(theta_inc_deg, phi_inc_deg),
            solver_settings_macro(),
        ]
    )


def farfield_export_macro(output_dir: str | Path, frequencies_ghz: list[float]) -> str:
    """Return a CST FarfieldCalculator macro for dense HH RCS CSV export."""

    grid = OBSERVATION_GRID
    freq_list = ";".join(f"{freq:g}" for freq in frequencies_ghz)
    return f'''
Option Explicit

Sub Main
    Dim thetaMin As Double, thetaMax As Double, thetaStep As Double
    Dim phiMin As Double, phiMax As Double, phiStep As Double
    Dim outDir As String, freqList As String
    thetaMin = {grid.theta_min_deg}
    thetaMax = {grid.theta_max_deg}
    thetaStep = {grid.theta_step_deg}
    phiMin = {grid.phi_min_deg}
    phiMax = {grid.phi_max_deg}
    phiStep = {grid.phi_step_deg}
    outDir = "{vba_path(output_dir)}\\\\"
    freqList = "{freq_list}"

    Dim ffname As String
    ffname = Resulttree.GetFirstChildName("Farfields")
    If (ffname = "Farfields\\Farfield Cuts") Then ffname = Resulttree.GetNextItemName(ffname)

    While (ffname <> "")
        FarfieldCalculator.ResetAsCurrentPlot
        FarfieldCalculator.SetScaleLinear(True)
        FarfieldCalculator.DBUnit("0")

        Dim freqParts As Variant, iFreq As Long
        freqParts = Split(freqList, ";")
        For iFreq = LBound(freqParts) To UBound(freqParts)
            Dim sfrq As String
            sfrq = CStr(freqParts(iFreq))
            FarfieldCalculator.ClearList

            Dim nTheta As Long, nPhi As Long, iPoint As Long
            nTheta = Fix((thetaMax - thetaMin) / thetaStep) + 1
            nPhi = Fix((phiMax - phiMin) / phiStep) + 1

            Dim thetaArr() As String, phiArr() As String
            Dim radiusArr(0) As String, coordArr(0) As String
            Dim typeArr(0) As String, sampleArr(0) As String
            ReDim thetaArr(nTheta * nPhi - 1)
            ReDim phiArr(nTheta * nPhi - 1)

            Dim th As Double, ph As Double
            iPoint = 0
            For th = thetaMin To thetaMax + 0.01 * thetaStep Step thetaStep
                For ph = phiMin To phiMax + 0.01 * phiStep Step phiStep
                    thetaArr(iPoint) = CStr(th)
                    phiArr(iPoint) = CStr(ph)
                    iPoint = iPoint + 1
                Next ph
            Next th

            radiusArr(0) = "0.0"
            coordArr(0) = "spherical"
            typeArr(0) = "frequency"
            sampleArr(0) = sfrq
            FarfieldCalculator.AddListEvaluationPoints thetaArr, phiArr, radiusArr, coordArr, typeArr, sampleArr
            FarfieldCalculator.CalculateList ffname, "farfield"

            Dim outTheta() As Double, outPhi() As Double, outRcs() As Double
            outTheta = FarfieldCalculator.GetList("epattern", "Point_T")
            outPhi = FarfieldCalculator.GetList("epattern", "Point_P")
            outRcs = FarfieldCalculator.GetList("rcs", "spherical abs")

            Dim fileNo As Integer, i As Long, outFile As String
            outFile = outDir + "farfield_f" + Replace(sfrq, ".", "p") + "_rcs.csv"
            fileNo = FreeFile()
            Open outFile For Output As #fileNo
            Print #fileNo, "theta_deg,phi_deg,frequency_ghz,rcs_m2"
            For i = LBound(outTheta) To UBound(outTheta)
                Print #fileNo, CStr(outTheta(i)) & "," & CStr(outPhi(i)) & "," & sfrq & "," & CStr(outRcs(i))
            Next i
            Close #fileNo
        Next iFreq

        ffname = Resulttree.GetNextItemName(ffname)
    Wend
End Sub
'''
