# GT-DieleNO Code Release

This folder contains a compact standalone PyTorch definition of the GT-DieleNO network architecture used in the paper, plus paper-aligned CST helper code in `cstpy`.

## Files

- `gtdieleno.py`: GT-DieleNO model definition, including the tetrahedral GNO encoder, Transolver-style latent mixer, FiLM conditioning, and DeepONet-style query decoder.
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

## Interface

- `tetra_features`: `[1, T, 11]` conditioned tetrahedral-cell features.
- `tetra_positions`: `[1, T, 3]` scale-normalized tetrahedral centroids used by the edge kernel, matching the paper's normalized centroid coordinates.
- `edge_index`: `[2, E]` directed local tetrahedral graph edges.
- `query`: `[1, Q, 13]` far-field query features. With FiLM conditioning enabled, one forward pass assumes a shared frequency/incidence condition across the `Q` queries; the observation-angle entries can vary within the batch.
- `tetra_quad_weights`: optional `[1, T]` volume/quadrature weights.
- output: `[1, Q, 1]` predicted scalar far-field response in the learning target domain.

The implementation is written as a plain PyTorch module and does not require `torch_geometric`.

## Smoke Test

```powershell
pip install -r requirements.txt
python .\gtdieleno.py
```

After PyTorch is installed, the command runs a small CPU forward pass and prints the output shape and parameter count.
