# GT-DieleNO Code Release

This folder contains compact standalone PyTorch definitions of the GT-DieleNO network architecture used in the paper and the earlier conference GTCDiele baseline network, plus paper-aligned CST helper code in `cstpy`.

## Files

- `gtdieleno.py`: GT-DieleNO model definition, including the tetrahedral GNO encoder, Transolver-style latent mixer, FiLM conditioning, and DeepONet-style query decoder.
- `gtcdiele.py`: conference GTCDiele baseline network definition, including the tetrahedral encoder, attention pooling, full-map MLP decoder, and optional smoothing layer.
- `requirements.txt`: minimal dependency note for running the standalone module.
- `cstpy/`: CST Studio Suite 2025 helper snippets and validation utilities aligned with the simulation settings stated in the paper.

## Minimal Usage

```python
import torch
from gtdieleno import GTDieleNO, count_parameters

model = GTDieleNO(
    tetra_feature_dim=11,
    query_dim=13,
    out_dim=1,
    hidden_dim=128,
    latent_dim=128,
    gno_depth=3,
    num_latents=16,
    nonlocal_knn=12,
    decoder_rank=64,
    transolver_layers=2,
    transolver_heads=4,
    use_film=True,
)

T = 1024
Q = 4096
tetra_features = torch.randn(1, T, 11)
tetra_positions = torch.randn(1, T, 3)
edge_index = torch.randint(0, T, (2, 4096))
query = torch.randn(1, Q, 13)
query[:, :, [2, 3, 4, 9, 10, 11, 12]] = query[:, :1, [2, 3, 4, 9, 10, 11, 12]]
tetra_quad_weights = torch.rand(1, T)

pred = model(tetra_features, tetra_positions, edge_index, query, tetra_quad_weights)
print(pred.shape)          # [1, Q, 1]
print(count_parameters(model))
```

Conference GTCDiele baseline:

```python
import torch
from gtcdiele import GTCDiele, count_parameters

model = GTCDiele()

T = 1024
tetra_features = torch.randn(1, T, 21)
in_em = torch.tensor([[40.0, 105.0, 4.25]])  # incident theta, incident phi, frequency

pred_map = model(tetra_features, in_em, smooth=True)
print(pred_map.shape)      # [1, 360, 720]
print(count_parameters(model))
```

## Interface

- `tetra_features`: `[1, T, 11]` conditioned tetrahedral-cell features.
- `tetra_positions`: `[1, T, 3]` scale-normalized tetrahedral centroids used by the edge kernel, matching the paper's normalized centroid coordinates.
- `edge_index`: `[2, E]` directed local tetrahedral graph edges.
- `query`: `[1, Q, 13]` far-field query features. With FiLM conditioning enabled, one forward pass assumes a shared frequency/incidence condition across the `Q` queries; the observation-angle entries can vary within the batch.
- `tetra_quad_weights`: optional `[1, T]` volume/quadrature weights.
- output: `[1, Q, 1]` predicted scalar far-field response in the learning target domain.

Both network files are written as plain PyTorch modules and do not require `torch_geometric`.

## Smoke Test

```powershell
pip install -r requirements.txt
python .\gtdieleno.py
python .\gtcdiele.py
```

After PyTorch is installed, each command runs a small CPU forward pass and prints the output shape and parameter count. The `gtcdiele.py` smoke test uses a reduced decoder size to keep the test lightweight; the default `GTCDiele()` constructor keeps the conference full-map `360 x 720` output setting.
