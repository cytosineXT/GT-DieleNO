# cstpy

`cstpy` contains paper-aligned helper code for the CST side of GT-DieleNO.
It records the CST Studio Suite 2025 settings and dense far-field export
conventions used by the manuscript:

- units: m, GHz, ns;
- time-domain solver with expanded-open boundaries, normal background, prepared far fields, no mesh adaptation, and a `-30` dB steady-state limit;
- fixed HH scalar RCS channel;
- incidence angles `(90, 0)`, `(60, 45)`, and `(120, 45)` degrees;
- dense observation grid `theta=0:0.5:179.5`, `phi=0:0.5:359.5`, giving `360 x 720` samples;
- full-frequency grid `0.10:0.01:1.00` GHz and geometry-scaling grid `0.10:0.10:1.00` GHz;
- dielectric material designs d00--d09 with `mu_r=1`, `tan_delta_e=0.01`, and magnetic loss tangent `0`.

## Files

- `paper_settings.py`: constants matching the paper's simulation protocol.
- `materials.py`: dielectric material library and d00--d09 design maps.
- `cst_macros.py`: CST VBA/History macro snippets for units, materials, HH plane wave, monitors, solver settings, STL import, and dense RCS export.
- `farfield_csv.py`: CSV validation and lightweight JSON reshaping for exported dense linear-RCS maps.
- `example_write_macro.py`: writes the project preamble and dense far-field export macros.

## Usage

```powershell
python -m cstpy.example_write_macro --out .\macro_preview
```

The generated `.bas` files are intended for CST Studio Suite 2025 automation.
CST installations can differ in their History/VBA object names, so inspect the
generated snippets against a small manual CST project before launching a long
batch run.

To validate one exported dense RCS CSV:

```python
from cstpy.farfield_csv import summarize_csv

summary = summarize_csv("farfield_f0p1_rcs.csv")
print(summary)
```
