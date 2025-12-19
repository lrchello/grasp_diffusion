# model/grasp_decoder.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def build_2d_grid(K: int, device: torch.device):

    side = int(math.ceil(math.sqrt(K)))
    xs = torch.linspace(-0.5, 0.5, side, device=device)
    ys = torch.linspace(-0.5, 0.5, side, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing='xy')
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (side*side, 2)
    if grid.shape[0] >= K:
        return grid[:K, :].contiguous()
    else:
        repeat_times = (K + grid.shape[0] - 1) // grid.shape[0]
        grid = grid.repeat(repeat_times, 1)[:K, :].contiguous()
        return grid

# ---------------------------
# FoldingDecoder (robust & deterministic)
# ---------------------------
class FoldingDecoder(nn.Module):
    """
    FoldingNet style decoder:
      - z: (B, latent_dim)
      - cond: (B, cond_dim)
      - returns: (B, K, 3)

    All layers created in __init__ (no runtime layer creation).
    """
    def __init__(self, latent_dim: int, cond_dim: int, hidden_dim: int = 256, K: int = 256,
                 n_folding_layers: int = 2, activation=nn.LeakyReLU, dropout: float = 0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.K = int(K)
        self.n_folding_layers = int(n_folding_layers)
        self.activation = activation
        self.dropout = float(dropout)

        # project (z||cond) -> global feature
        self.feature_proj = nn.Sequential(
            nn.Linear(self.latent_dim + self.cond_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            self.activation(negative_slope=0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            self.activation(negative_slope=0.2),
        )

        # expand MLP: (hidden_dim + 2) -> hidden_dim
        self.expand = nn.Sequential(
            nn.Linear(self.hidden_dim + 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            self.activation(negative_slope=0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            self.activation(negative_slope=0.2),
        )

        # folding mlps produce residuals in hidden space
        self.folding_mlps = nn.ModuleList()
        for _ in range(self.n_folding_layers):
            mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                self.activation(negative_slope=0.2),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                self.activation(negative_slope=0.2),
            )
            self.folding_mlps.append(mlp)

        # final head: hidden_dim -> 3
        self.to_xyz = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            self.activation(negative_slope=0.2),
            nn.Linear(self.hidden_dim // 2, 3)
        )

        # small refiner taking (hidden + xyz) -> delta_xyz
        self.refiner = nn.Sequential(
            nn.Linear(self.hidden_dim + 3, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            self.activation(negative_slope=0.2),
            nn.Linear(self.hidden_dim, 3)
        )

        # register fixed grid (will move with model.to(device))
        grid = build_2d_grid(self.K, device=torch.device('cpu'))  # cpu; moved to device in forward
        self.register_buffer("grid", grid, persistent=False)

    def forward(self, z: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim)
            cond: (B, cond_dim) or None
        Returns:
            points: (B, K, 3)
        """
        B = z.size(0)
        device = z.device
        if cond is None:
            cond = torch.zeros(B, self.cond_dim, dtype=z.dtype, device=device)

        # project latent+cond
        feat = torch.cat([z, cond], dim=-1)  # (B, latent+cond)
        feat_proj = self.feature_proj(feat)  # (B, hidden_dim)

        # expand to per-point
        feat_exp = feat_proj.unsqueeze(1).expand(-1, self.K, -1)  # (B,K,hidden)

        grid = self.grid.to(device).unsqueeze(0).expand(B, -1, -1)  # (B, K, 2)

        # initial per-point feature: concat feat_exp and grid -> transform
        x = torch.cat([feat_exp, grid], dim=-1)  # (B,K, hidden+2)
        x = self.expand(x)  # (B,K, hidden)

        # folding residual iterations
        for mlp in self.folding_mlps:
            dx = mlp(x)  # (B,K,hidden)
            x = x + dx  # residual update

        # get coarse xyz
        coarse_xyz = self.to_xyz(x)  # (B,K,3)

        # refine using hidden + coarse_xyz
        ref_in = torch.cat([x, coarse_xyz], dim=-1)  # (B,K, hidden+3)
        delta = self.refiner(ref_in)  # (B,K,3)
        out = coarse_xyz + delta
        return out


class PointSelfAttention(nn.Module):
    def __init__(self, dim, n_heads=4, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        B, K, C = x.shape
        qkv = self.to_qkv(x).view(B, K, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, K, head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, heads, K, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, K, C)
        out = self.proj_out(out)
        out = self.dropout(out)
        return out

class PTDecoder(nn.Module):
    """
    Folding -> lift features -> several self-attn refinement blocks -> final MLP
    """
    def __init__(self, latent_dim: int, cond_dim: int, hidden_dim: int = 256, K: int = 512,
                 n_folding_layers: int = 2, n_attn_layers: int = 2, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.K = int(K)

        # base folding generator (produces coarse points and internal hidden)
        self.folding = FoldingDecoder(latent_dim=latent_dim, cond_dim=cond_dim, hidden_dim=hidden_dim,
                                     K=self.K, n_folding_layers=n_folding_layers)

        # feature lifting: map (xyz + cond) -> hidden_dim
        self.lift_proj = nn.Sequential(
            nn.Linear(3 + cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )

        # attention blocks
        self.attn_blocks = nn.ModuleList([PointSelfAttention(hidden_dim, n_heads=n_heads, dropout=dropout)
                                          for _ in range(n_attn_layers)])

        # final head
        self.to_xyz = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim // 2, 3)
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        device = z.device
        cond = cond if cond is not None else torch.zeros(B, self.cond_dim, device=device, dtype=z.dtype)

        # get coarse points from folding (and internal x not returned; use coarse for lifting)
        coarse = self.folding(z, cond)  # (B, K, 3)

        # lift coarse + cond -> feature
        cond_expand = cond.unsqueeze(1).expand(-1, self.K, -1)  # (B,K,cond_dim)
        lift_in = torch.cat([coarse, cond_expand], dim=-1)  # (B,K, 3+cond_dim)
        feat = self.lift_proj(lift_in)  # (B,K,hidden_dim)

        # self-attention refinement (residual)
        for attn in self.attn_blocks:
            feat = feat + attn(feat)

        xyz = self.to_xyz(feat)  # (B,K,3)
        return xyz

class StrongDecoder(nn.Module):
    """
    Wrapper: impl='fold' or impl='pt'
    """
    def __init__(self, impl: str = 'fold', latent_dim: int = 32, cond_dim: int = 512, **kwargs):
        super().__init__()
        self.impl = impl
        if impl == 'fold':
            self.model = FoldingDecoder(latent_dim=latent_dim, cond_dim=cond_dim, **kwargs)
        elif impl == 'pt':
            self.model = PTDecoder(latent_dim=latent_dim, cond_dim=cond_dim, **kwargs)
        else:
            raise ValueError("impl must be 'fold' or 'pt'")

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.model(z, cond)
