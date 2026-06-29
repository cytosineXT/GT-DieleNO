from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    depth: int = 3,
    activation: type[nn.Module] = nn.SiLU,
    final_activation: nn.Module | None = None,
) -> nn.Sequential:
    if depth < 1:
        raise ValueError("depth must be >= 1")
    layers: list[nn.Module] = []
    last_dim = in_dim
    for _ in range(depth - 1):
        layers.extend([nn.Linear(last_dim, hidden_dim), activation()])
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, out_dim))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


class FourierFeatureEncoder(nn.Module):
    def __init__(self, in_dim: int, num_bands: int = 5, max_frequency: float = 8.0):
        super().__init__()
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")
        bands = torch.logspace(0, math.log10(max_frequency), num_bands)
        self.register_buffer("bands", bands, persistent=False)
        self.out_dim = in_dim * (1 + 2 * num_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb = x.unsqueeze(-1) * self.bands
        return torch.cat([x, torch.sin(math.pi * xb).flatten(-2), torch.cos(math.pi * xb).flatten(-2)], dim=-1)


class FiLMModulator(nn.Module):
    def __init__(self, condition_dim: int, feature_dim: int, hidden_dim: int = 128, zero_init: bool = True):
        super().__init__()
        if condition_dim <= 0:
            raise ValueError("condition_dim must be positive")
        self.condition_dim = condition_dim
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * feature_dim),
        )
        if zero_init:
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 2:
            raise ValueError(f"condition must have shape [B,C], got {tuple(condition.shape)}")
        if condition.shape[-1] != self.condition_dim:
            raise ValueError(f"condition dim must be {self.condition_dim}, got {condition.shape[-1]}")
        gamma, beta = self.net(condition).chunk(2, dim=-1)
        while gamma.ndim < x.ndim:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return x * (1.0 + gamma) + beta


class OperatorQueryDecoder(nn.Module):
    """DeepONet-style query decoder for arbitrary far-field samples."""

    def __init__(
        self,
        latent_dim: int,
        query_dim: int,
        out_dim: int = 1,
        hidden_dim: int = 128,
        rank: int = 64,
        fourier_bands: int = 5,
        use_residual: bool = True,
        condition_dim: int = 0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.out_dim = out_dim
        self.rank = rank
        self.condition_dim = condition_dim
        self.query_encoder = FourierFeatureEncoder(query_dim, num_bands=fourier_bands)
        self.query_film = (
            FiLMModulator(condition_dim, self.query_encoder.out_dim, hidden_dim=hidden_dim)
            if condition_dim > 0
            else None
        )
        self.branch = make_mlp(latent_dim, hidden_dim, out_dim * rank, depth=3)
        self.trunk = make_mlp(self.query_encoder.out_dim, hidden_dim, out_dim * rank, depth=3)
        self.query_bias = make_mlp(self.query_encoder.out_dim, hidden_dim, out_dim, depth=2)
        self.residual = (
            make_mlp(latent_dim + self.query_encoder.out_dim, hidden_dim, out_dim, depth=3)
            if use_residual
            else None
        )

    def forward(
        self,
        latent: torch.Tensor,
        query: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latent.ndim != 2:
            raise ValueError(f"latent must have shape [B,D], got {tuple(latent.shape)}")
        if query.ndim != 3:
            raise ValueError(f"query must have shape [B,Q,C], got {tuple(query.shape)}")
        encoded_query = self.query_encoder(query)
        if self.query_film is not None:
            if condition is None:
                raise ValueError("condition is required when query FiLM is enabled")
            encoded_query = self.query_film(encoded_query, condition)
        branch = self.branch(latent).view(latent.shape[0], 1, self.out_dim, self.rank)
        trunk = self.trunk(encoded_query).view(query.shape[0], query.shape[1], self.out_dim, self.rank)
        pred = (branch * trunk).sum(dim=-1) / math.sqrt(self.rank)
        pred = pred + self.query_bias(encoded_query)
        if self.residual is not None:
            latent_expanded = latent.unsqueeze(1).expand(-1, query.shape[1], -1)
            pred = pred + self.residual(torch.cat([latent_expanded, encoded_query], dim=-1))
        return pred


def build_knn_edges(pos: torch.Tensor, k: int, chunk_size: int = 2048) -> torch.Tensor:
    if k <= 0 or pos.shape[0] <= 1:
        return torch.zeros((2, 0), dtype=torch.long, device=pos.device)
    k_eff = min(k, pos.shape[0] - 1)
    num_nodes = pos.shape[0]
    chunk = max(1, min(int(os.environ.get("GTDIELENO_KNN_CHUNK", str(chunk_size))), num_nodes))
    out_device = pos.device
    work_pos = pos.detach()
    knn_device = os.environ.get("GTDIELENO_KNN_DEVICE", "cpu").strip().lower()
    if knn_device != "cuda":
        work_pos = work_pos.to(device="cpu", dtype=torch.float32)
    else:
        work_pos = work_pos.to(dtype=torch.float32)

    src_parts: list[torch.Tensor] = []
    dst_parts: list[torch.Tensor] = []
    all_idx = torch.arange(num_nodes, device=work_pos.device)
    for start in range(0, num_nodes, chunk):
        end = min(start + chunk, num_nodes)
        dist = torch.cdist(work_pos[start:end], work_pos)
        row_idx = torch.arange(end - start, device=work_pos.device)
        dist[row_idx, all_idx[start:end]] = float("inf")
        src_parts.append(torch.topk(dist, k=k_eff, largest=False, dim=1).indices.reshape(-1))
        dst_parts.append(torch.arange(start, end, device=work_pos.device).repeat_interleave(k_eff))
    src = torch.cat(src_parts, dim=0)
    dst = torch.cat(dst_parts, dim=0)
    return torch.stack([src, dst], dim=0).to(device=out_device)


def merge_edge_indices(
    local_edge_index: torch.Tensor | None,
    nonlocal_edge_index: torch.Tensor | None,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = None
    parts: list[torch.Tensor] = []
    types: list[torch.Tensor] = []
    if local_edge_index is not None and local_edge_index.numel():
        device = local_edge_index.device
        parts.append(local_edge_index.long())
        types.append(torch.zeros(local_edge_index.shape[1], dtype=torch.long, device=local_edge_index.device))
    if nonlocal_edge_index is not None and nonlocal_edge_index.numel():
        device = nonlocal_edge_index.device
        parts.append(nonlocal_edge_index.long())
        types.append(torch.ones(nonlocal_edge_index.shape[1], dtype=torch.long, device=nonlocal_edge_index.device))
    if not parts:
        return (
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros((0,), dtype=torch.long, device=device),
        )

    edge_index = torch.cat(parts, dim=1)
    edge_type = torch.cat(types, dim=0)
    keys = edge_index[0] * num_nodes + edge_index[1]
    order = torch.argsort(keys * 2 + edge_type)
    sorted_keys = keys[order]
    keep = torch.ones_like(sorted_keys, dtype=torch.bool)
    keep[1:] = sorted_keys[1:] != sorted_keys[:-1]
    selected = order[keep]
    return edge_index[:, selected], edge_type[selected]


class GNOBlock(nn.Module):
    """Volume-weighted graph neural-operator block over tetrahedral cells."""

    def __init__(
        self,
        dim: int,
        pos_dim: int = 3,
        hidden_dim: int | None = None,
        edge_type_dim: int = 2,
        normalize_quadrature: bool = True,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.edge_type_dim = edge_type_dim
        self.normalize_quadrature = normalize_quadrature
        self.edge_chunk_size = max(1, int(os.environ.get("GTDIELENO_GNO_EDGE_CHUNK", "131072")))
        self.value = nn.Linear(dim, dim, bias=False)
        self.kernel = make_mlp((2 * dim) + pos_dim + 1 + edge_type_dim, hidden_dim, dim, depth=3)
        self.update = make_mlp(2 * dim, hidden_dim, dim, depth=2)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        quad_weights: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 2 or pos.ndim != 2:
            raise ValueError("GNOBlock expects unbatched x [T,D] and pos [T,3]")
        if edge_index.numel() == 0:
            return self.norm(x + self.update(torch.cat([x, torch.zeros_like(x)], dim=-1)))

        agg = torch.zeros_like(x)
        weight_sum = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype) if self.normalize_quadrature else None
        src_all, dst_all = edge_index[0].long(), edge_index[1].long()
        if edge_type is None:
            edge_type_all = torch.zeros(src_all.shape[0], dtype=torch.long, device=x.device)
        else:
            edge_type_all = edge_type.clamp(0, self.edge_type_dim - 1)
        quad_weights_flat = None
        if quad_weights is not None:
            quad_weights_flat = quad_weights.reshape(-1) if quad_weights.ndim != 1 else quad_weights

        chunk_size = max(1, min(self.edge_chunk_size, src_all.shape[0]))
        for start in range(0, src_all.shape[0], chunk_size):
            end = min(start + chunk_size, src_all.shape[0])
            src = src_all[start:end]
            dst = dst_all[start:end]
            rel = pos[src] - pos[dst]
            dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
            edge_type_feature = F.one_hot(edge_type_all[start:end], num_classes=self.edge_type_dim).to(dtype=x.dtype)
            if quad_weights_flat is None:
                src_weight = torch.ones((src.shape[0], 1), device=x.device, dtype=x.dtype)
            else:
                src_weight = quad_weights_flat[src].to(device=x.device, dtype=x.dtype).reshape(-1, 1).clamp_min(0.0)
            kernel = self.kernel(torch.cat([x[src], x[dst], rel, dist, edge_type_feature], dim=-1))
            msg = kernel * self.value(x[src])
            agg.index_add_(0, dst, msg * src_weight)
            if weight_sum is not None:
                weight_sum.index_add_(0, dst, src_weight)

        if self.normalize_quadrature:
            assert weight_sum is not None
            agg = agg / weight_sum.clamp_min(1e-12)
        return self.norm(x + self.update(torch.cat([x, agg], dim=-1)))


class TransolverStyleLatentMixer(nn.Module):
    """Slice-token latent mixer inspired by Transolver."""

    def __init__(self, dim: int, num_latents: int = 16, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.slice_queries = nn.Parameter(torch.randn(num_latents, dim) / math.sqrt(dim))
        self.slice_score = nn.Linear(dim, num_latents)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.mixer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, quad_weights: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.slice_score(x) + x @ self.slice_queries.t()
        assignment = torch.softmax(scores, dim=0)
        if quad_weights is not None:
            if quad_weights.ndim != 1:
                quad_weights = quad_weights.reshape(-1)
            assignment = assignment * quad_weights.to(device=x.device, dtype=x.dtype).reshape(-1, 1).clamp_min(0.0)
        token_norm = assignment.sum(dim=0, keepdim=True).t().clamp_min(1e-12)
        tokens = (assignment.t() @ x) / token_norm
        mixed = self.mixer(tokens.unsqueeze(0)).squeeze(0)
        return self.norm(mixed.mean(dim=0, keepdim=True)).squeeze(0)


class TetraGNOEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        gno_depth: int = 3,
        num_latents: int = 16,
        nonlocal_knn: int = 12,
        normalize_quadrature: bool = True,
        transolver_layers: int = 2,
        transolver_heads: int = 4,
        checkpoint_blocks: bool = False,
    ):
        super().__init__()
        self.nonlocal_knn = nonlocal_knn
        self.checkpoint_blocks = checkpoint_blocks
        self.input = make_mlp(in_dim, hidden_dim, latent_dim, depth=2)
        self.blocks = nn.ModuleList(
            [GNOBlock(latent_dim, normalize_quadrature=normalize_quadrature) for _ in range(gno_depth)]
        )
        self.mixer = TransolverStyleLatentMixer(
            latent_dim,
            num_latents=num_latents,
            num_layers=transolver_layers,
            num_heads=transolver_heads,
        )
        self.post = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.SiLU(), nn.LayerNorm(latent_dim))

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        quad_weights: torch.Tensor | None = None,
        nonlocal_edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.input(features)
        edge_index = edge_index.to(device=positions.device)
        if nonlocal_edge_index is None:
            nonlocal_edges = build_knn_edges(positions, self.nonlocal_knn)
        else:
            nonlocal_edges = nonlocal_edge_index.to(device=positions.device)
        operator_edges, edge_type = merge_edge_indices(edge_index, nonlocal_edges, num_nodes=features.shape[0])
        for block in self.blocks:
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                def block_forward(x_in: torch.Tensor, *, block: GNOBlock = block) -> torch.Tensor:
                    return block(x_in, positions, operator_edges, quad_weights=quad_weights, edge_type=edge_type)

                x = checkpoint(block_forward, x, use_reentrant=False)
            else:
                x = block(x, positions, operator_edges, quad_weights=quad_weights, edge_type=edge_type)
        return self.post(self.mixer(x, quad_weights=quad_weights))


class GTDieleNO(nn.Module):
    """Graph-tetrahedral dielectric neural operator for queryable RCS prediction."""

    def __init__(
        self,
        tetra_feature_dim: int = 11,
        query_dim: int = 13,
        out_dim: int = 1,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        gno_depth: int = 3,
        num_latents: int = 16,
        nonlocal_knn: int = 12,
        decoder_rank: int = 64,
        transolver_layers: int = 2,
        transolver_heads: int = 4,
        use_film: bool = True,
        film_condition_indices: tuple[int, ...] = (2, 3, 4, 9, 10, 11, 12),
        encoder_condition_dim: int = 0,
        encoder_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_film = use_film
        if encoder_condition_dim < 0:
            raise ValueError("encoder_condition_dim must be non-negative")
        if encoder_condition_dim not in (0, len(film_condition_indices)):
            raise ValueError(
                "encoder_condition_dim currently supports either 0 or the full "
                f"scattering condition dim {len(film_condition_indices)}, got {encoder_condition_dim}"
            )
        self.encoder_condition_dim = encoder_condition_dim
        condition_dim = len(film_condition_indices) if use_film else 0
        self.register_buffer(
            "film_condition_indices",
            torch.tensor(film_condition_indices if use_film else (), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "scattering_condition_indices",
            torch.tensor(film_condition_indices, dtype=torch.long),
            persistent=False,
        )
        self.encoder = TetraGNOEncoder(
            in_dim=tetra_feature_dim + encoder_condition_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            gno_depth=gno_depth,
            num_latents=num_latents,
            nonlocal_knn=nonlocal_knn,
            transolver_layers=transolver_layers,
            transolver_heads=transolver_heads,
            checkpoint_blocks=encoder_checkpoint,
        )
        self.latent_film = FiLMModulator(condition_dim, latent_dim, hidden_dim=hidden_dim) if use_film else None
        self.decoder = OperatorQueryDecoder(
            latent_dim=latent_dim,
            query_dim=query_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            rank=decoder_rank,
            fourier_bands=5,
            condition_dim=condition_dim,
        )

    def scattering_condition_from_query(self, query: torch.Tensor) -> torch.Tensor:
        if query.ndim != 3:
            raise ValueError(f"query must have shape [B,Q,C], got {tuple(query.shape)}")
        if query.shape[1] < 1:
            raise ValueError("query must contain at least one point for scattering conditioning")
        max_index = int(self.scattering_condition_indices.max().item())
        if query.shape[-1] <= max_index:
            raise ValueError(f"query dim {query.shape[-1]} is too small for condition index {max_index}")
        return query[:, 0, :].index_select(-1, self.scattering_condition_indices.to(device=query.device))

    def condition_from_query(self, query: torch.Tensor) -> torch.Tensor | None:
        if not self.use_film:
            return None
        return self.scattering_condition_from_query(query)

    def encoder_condition_from_query(self, query: torch.Tensor) -> torch.Tensor | None:
        if self.encoder_condition_dim <= 0:
            return None
        condition = self.scattering_condition_from_query(query)
        if condition.shape[-1] != self.encoder_condition_dim:
            raise ValueError(f"encoder condition dim must be {self.encoder_condition_dim}, got {condition.shape[-1]}")
        return condition

    def encode_tetra(
        self,
        tetra_features: torch.Tensor,
        tetra_positions: torch.Tensor,
        edge_index: torch.Tensor,
        tetra_quad_weights: torch.Tensor | None = None,
        encoder_condition: torch.Tensor | None = None,
        nonlocal_edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tetra_features.ndim == 3:
            if tetra_features.shape[0] != 1:
                raise ValueError("GTDieleNO currently supports batch size 1 for ragged tetra graphs")
            tetra_features = tetra_features.squeeze(0)
            tetra_positions = tetra_positions.squeeze(0)
            if tetra_quad_weights is not None and tetra_quad_weights.ndim == 2:
                tetra_quad_weights = tetra_quad_weights.squeeze(0)
        if self.encoder_condition_dim > 0:
            if encoder_condition is None:
                raise ValueError("encoder_condition is required when encoder conditioning is enabled")
            if encoder_condition.ndim == 2:
                if encoder_condition.shape[0] != 1:
                    raise ValueError("GTDieleNO currently supports batch size 1 for ragged encoder conditioning")
                encoder_condition = encoder_condition.squeeze(0)
            elif encoder_condition.ndim != 1:
                raise ValueError(f"encoder_condition must have shape [C] or [1,C], got {tuple(encoder_condition.shape)}")
            if encoder_condition.shape[-1] != self.encoder_condition_dim:
                raise ValueError(f"encoder_condition dim must be {self.encoder_condition_dim}, got {encoder_condition.shape[-1]}")
            cond_nodes = encoder_condition.to(device=tetra_features.device, dtype=tetra_features.dtype).reshape(1, -1)
            cond_nodes = cond_nodes.expand(tetra_features.shape[0], -1)
            tetra_features = torch.cat([tetra_features, cond_nodes], dim=-1)
        elif encoder_condition is not None:
            raise ValueError("encoder_condition was provided but encoder conditioning is disabled")
        return self.encoder(
            tetra_features,
            tetra_positions,
            edge_index,
            quad_weights=tetra_quad_weights,
            nonlocal_edge_index=nonlocal_edge_index,
        ).unsqueeze(0)

    def decode_queries(self, latent: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        condition = self.condition_from_query(query)
        if self.latent_film is not None:
            if condition is None:
                raise ValueError("condition is required when GTDieleNO FiLM is enabled")
            latent = self.latent_film(latent, condition)
        return self.decoder(latent, query, condition=condition)

    def forward(
        self,
        tetra_features: torch.Tensor,
        tetra_positions: torch.Tensor,
        edge_index: torch.Tensor,
        query: torch.Tensor,
        tetra_quad_weights: torch.Tensor | None = None,
        nonlocal_edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoder_condition = self.encoder_condition_from_query(query)
        latent = self.encode_tetra(
            tetra_features,
            tetra_positions,
            edge_index,
            tetra_quad_weights=tetra_quad_weights,
            encoder_condition=encoder_condition,
            nonlocal_edge_index=nonlocal_edge_index,
        )
        return self.decode_queries(latent, query)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
