from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableMedianFilter(nn.Module):
    """Reflection-padded median filter used by the conference GTCDiele decoder."""

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.kernel_size = int(kernel_size)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim != 4:
            raise ValueError(f"img must have shape [B,C,H,W], got {tuple(img.shape)}")
        pad = self.kernel_size // 2
        img = F.pad(img, (pad, pad, pad, pad), mode="reflect")
        patches = F.unfold(img, kernel_size=self.kernel_size)
        bsz, channels, _, _ = img.shape
        patches = patches.view(bsz, channels, self.kernel_size * self.kernel_size, -1)
        median = patches.median(dim=2).values
        return median.view(bsz, channels, img.shape[-2] - 2 * pad, img.shape[-1] - 2 * pad)


class LearnableGaussianFilter(nn.Module):
    """Depthwise Gaussian filter with a learnable standard deviation."""

    def __init__(self, kernel_size: int = 5, init_sigma: float = 4.0):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.kernel_size = int(kernel_size)
        self.sigma = nn.Parameter(torch.tensor(float(init_sigma), dtype=torch.float32))

    def gaussian_kernel(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        radius = self.kernel_size // 2
        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        sigma = self.sigma.to(device=device, dtype=dtype).clamp_min(torch.finfo(dtype).eps)
        kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma.square()))
        kernel = kernel / kernel.sum().clamp_min(torch.finfo(dtype).eps)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim != 4:
            raise ValueError(f"img must have shape [B,C,H,W], got {tuple(img.shape)}")
        kernel = self.gaussian_kernel(device=img.device, dtype=img.dtype)
        kernel = kernel.repeat(img.shape[1], 1, 1, 1)
        return F.conv2d(img, kernel, padding=self.kernel_size // 2, groups=img.shape[1])


class SmoothingLayer(nn.Module):
    """Median-plus-Gaussian smoothing layer from the conference implementation."""

    def __init__(self, kernel_size: int = 5, init_sigma: float = 4.0):
        super().__init__()
        self.median_filter = LearnableMedianFilter(kernel_size=kernel_size)
        self.gaussian_filter = LearnableGaussianFilter(kernel_size=kernel_size, init_sigma=init_sigma)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim != 3:
            raise ValueError(f"img must have shape [B,H,W], got {tuple(img.shape)}")
        img = img.unsqueeze(1)
        img = self.median_filter(img)
        img = self.gaussian_filter(img)
        return img.squeeze(1)


class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.score_net(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


class GTCEncoder(nn.Module):
    """Tetrahedral feature encoder used by the conference GTCDiele baseline.

    Input tetrahedral features follow the conference code convention:
    `[vertices(12), volume(1), centroid(3), solid_angles(4), dielectric(1)]`.
    The mesh-face embedding layers are kept for checkpoint compatibility with
    the original released architecture, although the tetrahedral forward path
    does not consume them.
    """

    def __init__(
        self,
        dim_coor_embed: int = 64,
        dim_area_embed: int = 16,
        dim_normal_embed: int = 64,
        dim_angle_embed: int = 16,
        encoder_dims_through_depth: tuple[int, ...] = (64, 128, 256, 256, 256),
        attn_encoder_depth: int = 1,
        attn_dropout: float = 0.0,
        pooling_type: str = "cls",
    ):
        super().__init__()
        if len(encoder_dims_through_depth) < 1:
            raise ValueError("encoder_dims_through_depth must contain at least one dimension")
        self.pooling_type = pooling_type.lower()

        self.coor_embed = nn.Linear(9, 9 * dim_coor_embed)
        self.area_embed = nn.Linear(1, dim_area_embed)
        self.normal_embed = nn.Linear(3, 3 * dim_normal_embed)
        self.angle_embed = nn.Linear(3, 3 * dim_angle_embed)

        self.tet_coor_embed = nn.Linear(12, 12 * dim_coor_embed)
        self.tet_vol_embed = nn.Linear(1, dim_area_embed)
        self.tet_center_embed = nn.Linear(3, 3 * dim_normal_embed)
        self.tet_angle_embed = nn.Linear(4, 4 * dim_angle_embed)
        self.tet_dielectric_embed = nn.Linear(1, 16)

        tet_init_dim = 12 * dim_coor_embed + dim_area_embed + 3 * dim_normal_embed + 4 * dim_angle_embed + 16
        init_dim, *encoder_dims = encoder_dims_through_depth
        self.tet_init_linear = nn.Linear(tet_init_dim, init_dim)
        self.init_encoder_act_and_norm = nn.Sequential(nn.SiLU(), nn.LayerNorm(init_dim))

        self.encoders = nn.ModuleList()
        self.encoder_act_and_norm = nn.ModuleList()
        curr_dim = init_dim
        for dim_layer in encoder_dims:
            self.encoders.append(nn.Linear(curr_dim, dim_layer))
            self.encoder_act_and_norm.append(nn.Sequential(nn.SiLU(), nn.LayerNorm(dim_layer)))
            curr_dim = dim_layer
        self.final_dim = curr_dim

        if self.pooling_type == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, curr_dim))
        elif self.pooling_type == "ap":
            self.attn_pooling = AttentionPooling(curr_dim)
        elif self.pooling_type not in {"mean", "max"}:
            raise ValueError(f"Unknown pooling type: {pooling_type}")

        self.encoder_attn_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=curr_dim,
                    nhead=8,
                    dropout=attn_dropout,
                    batch_first=True,
                )
                for _ in range(attn_encoder_depth)
            ]
        )

    @staticmethod
    def parse_tetra_features(x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.shape[-1] != 21:
            raise ValueError(f"tetra features must have 21 channels, got {x.shape[-1]}")
        return {
            "coords": x[..., :12],
            "volume": x[..., 12:13],
            "centroid": x[..., 13:16],
            "angles": x[..., 16:20],
            "dielectric": x[..., 20:21],
        }

    def forward(self, x: torch.Tensor, face_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B,N,21], got {tuple(x.shape)}")
        bsz, _, _ = x.shape
        features = self.parse_tetra_features(x)

        face_embed = torch.cat(
            [
                self.tet_coor_embed(features["coords"]),
                self.tet_angle_embed(features["angles"]),
                self.tet_vol_embed(features["volume"]),
                self.tet_center_embed(features["centroid"]),
                self.tet_dielectric_embed(features["dielectric"]),
            ],
            dim=-1,
        )

        orig_shape = face_embed.shape
        face_embed = face_embed.reshape(-1, face_embed.shape[-1])
        face_embed = self.tet_init_linear(face_embed)
        face_embed = self.init_encoder_act_and_norm(face_embed)
        for linear, act_norm in zip(self.encoders, self.encoder_act_and_norm):
            face_embed = act_norm(linear(face_embed))
        face_embed = face_embed.view(orig_shape[0], orig_shape[1], -1)

        if face_mask is not None:
            if face_mask.shape != x.shape[:2]:
                raise ValueError(f"face_mask must have shape {tuple(x.shape[:2])}, got {tuple(face_mask.shape)}")
            key_padding_mask = ~face_mask
        else:
            key_padding_mask = torch.zeros((bsz, face_embed.shape[1]), dtype=torch.bool, device=face_embed.device)

        if self.pooling_type == "cls":
            cls_tokens = self.cls_token.expand(bsz, -1, -1)
            face_embed = torch.cat((cls_tokens, face_embed), dim=1)
            cls_mask = torch.zeros((bsz, 1), dtype=torch.bool, device=face_embed.device)
            key_padding_mask = torch.cat((cls_mask, key_padding_mask), dim=1)

        for attn_layer in self.encoder_attn_blocks:
            face_embed = attn_layer(face_embed, src_key_padding_mask=key_padding_mask)

        if self.pooling_type == "cls":
            latent = face_embed[:, 0, :]
            face_embed_seq = face_embed[:, 1:, :]
        elif self.pooling_type == "ap":
            latent = self.attn_pooling(face_embed, mask=face_mask)
            face_embed_seq = face_embed
        elif self.pooling_type == "max":
            if face_mask is not None:
                latent = face_embed.masked_fill(~face_mask.unsqueeze(-1), -1e9).max(dim=1).values
            else:
                latent = face_embed.max(dim=1).values
            face_embed_seq = face_embed
        else:
            if face_mask is not None:
                weights = face_mask.unsqueeze(-1).to(face_embed.dtype)
                latent = (face_embed * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            else:
                latent = face_embed.mean(dim=1)
            face_embed_seq = face_embed

        return latent, face_embed_seq


class MLPUpsampleBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, in_h: int, in_w: int):
        super().__init__()
        self.channel_proj = nn.Linear(in_c, out_c)
        self.h_map = nn.Linear(in_h, in_h * 2)
        self.w_map = nn.Linear(in_w, in_w * 2)
        self.norm = nn.LayerNorm(out_c)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must have shape [B,C,H,W], got {tuple(x.shape)}")
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.channel_proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.h_map(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.w_map(x)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = self.norm(x)
        x = self.act(x)
        return x.permute(0, 3, 1, 2).contiguous()


class RCSDecoder(nn.Module):
    """Full-map RCS decoder from the conference GTCDiele network."""

    def __init__(
        self,
        latent_dim: int = 256,
        middim: int = 256,
        seed_length: int = 2250,
        base_height: int = 45,
        base_width: int = 90,
    ):
        super().__init__()
        self.seed_length = int(seed_length)
        self.base_height = int(base_height)
        self.base_width = int(base_width)

        self.latent_proj = nn.Linear(latent_dim, middim * self.seed_length)
        self.incident_angle_linear1 = nn.Linear(2, self.seed_length)
        self.emfreq_embed1 = nn.Linear(1, self.seed_length)

        map_size = self.base_height * self.base_width
        self.fc1d1 = nn.Linear(self.seed_length, map_size)
        self.incident_angle_linear2 = nn.Linear(2, map_size)
        self.emfreq_embed2 = nn.Linear(1, map_size)

        self.mlp_stage1 = MLPUpsampleBlock(middim, middim // 2, self.base_height, self.base_width)
        self.mlp_stage2 = MLPUpsampleBlock(middim // 2, middim // 4, self.base_height * 2, self.base_width * 2)
        self.mlp_stage3 = MLPUpsampleBlock(middim // 4, middim // 8, self.base_height * 4, self.base_width * 4)
        self.final_proj = nn.Linear(middim // 8, 1)

    @staticmethod
    def _parse_condition(in_em: torch.Tensor | list[torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(in_em, list):
            if len(in_em) != 3:
                raise ValueError("in_em list must contain [theta, phi, freq]")
            theta = in_em[0].to(device).float()
            phi = in_em[1].to(device).float()
            freq = in_em[2].to(device).float().unsqueeze(1)
        else:
            if in_em.ndim != 2 or in_em.shape[-1] != 3:
                raise ValueError(f"in_em must have shape [B,3], got {tuple(in_em.shape)}")
            theta = in_em[:, 0].to(device).float()
            phi = in_em[:, 1].to(device).float()
            freq = in_em[:, 2].to(device).float().unsqueeze(1)
        angles = torch.stack([theta / 180.0, phi / 360.0], dim=1)
        return angles, freq

    def forward(self, latent: torch.Tensor, in_em: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        in_angle, in_freq = self._parse_condition(in_em, latent.device)

        condangle1 = self.incident_angle_linear1(in_angle)
        condfreq1 = self.emfreq_embed1(in_freq)

        x = self.latent_proj(latent)
        x = x.view(x.shape[0], -1, self.seed_length)
        x = x + condangle1.unsqueeze(1) + condfreq1.unsqueeze(1)
        x = self.fc1d1(x)

        condangle2 = self.incident_angle_linear2(in_angle)
        condfreq2 = self.emfreq_embed2(in_freq)
        x = x + condangle2.unsqueeze(1) + condfreq2.unsqueeze(1)
        x = x.reshape(x.shape[0], -1, self.base_height, self.base_width).contiguous()

        x = self.mlp_stage1(x)
        x = self.mlp_stage2(x)
        x = self.mlp_stage3(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.final_proj(x)
        return x.squeeze(-1)


class GTC_Tet_RCS_Model(nn.Module):
    """Conference GTCDiele tetrahedral-to-full-map RCS network.

    The default constructor mirrors the conference baseline used as a comparison
    in the GT-DieleNO paper: tetrahedral encoder, attention pooling, full-map MLP
    decoder, and optional median/Gaussian output smoothing.
    """

    def __init__(
        self,
        dim_coor_embed: int = 64,
        latent_dim: int = 256,
        middim: int = 256,
        kernel_size: int = 5,
        init_sigma: float = 4.0,
        device: str | torch.device | None = None,
        seed_length: int = 2250,
        base_height: int = 45,
        base_width: int = 90,
    ):
        super().__init__()
        self.encoder = GTCEncoder(
            dim_coor_embed=dim_coor_embed,
            dim_area_embed=16,
            dim_normal_embed=64,
            dim_angle_embed=16,
            pooling_type="ap",
        )
        self.decoder = RCSDecoder(
            latent_dim=latent_dim,
            middim=middim,
            seed_length=seed_length,
            base_height=base_height,
            base_width=base_width,
        )
        self.smoothing_layer = SmoothingLayer(kernel_size=kernel_size, init_sigma=init_sigma)
        self.loss_type = "L1"
        self.gama = 0.001
        self.device = device

    def forward(
        self,
        x: torch.Tensor,
        in_em: torch.Tensor | list[torch.Tensor],
        GT: torch.Tensor | None = None,
        smooth: bool = True,
        return_metrics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        latent, _ = self.encoder(x)
        decoded = self.decoder(latent, in_em)
        if smooth:
            decoded = self.smoothing_layer(decoded)
        if GT is None:
            return decoded
        loss = self.loss_fn(decoded, GT)
        if return_metrics:
            mse = F.mse_loss(decoded, GT)
            metrics = {"RMSE": torch.sqrt(mse)}
            return loss, decoded, metrics
        return loss, decoded

    def loss_fn(self, decoded: torch.Tensor, GT: torch.Tensor) -> torch.Tensor:
        base_loss = F.l1_loss(decoded, GT)
        max_diff = torch.abs(decoded.amax(dim=(1, 2)) - GT.amax(dim=(1, 2)))
        return base_loss + self.gama * max_diff.mean()


GTCDiele = GTC_Tet_RCS_Model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = GTCDiele(
        dim_coor_embed=8,
        middim=32,
        seed_length=24,
        base_height=4,
        base_width=6,
    )
    model.eval()

    tetra_features = torch.randn(2, 16, 21)
    in_em = torch.tensor([[40.0, 105.0, 4.25], [60.0, 210.0, 7.50]])
    with torch.no_grad():
        pred = model(tetra_features, in_em, smooth=False)

    print(f"output_shape={tuple(pred.shape)}")
    print(f"trainable_parameters={count_parameters(model)}")
