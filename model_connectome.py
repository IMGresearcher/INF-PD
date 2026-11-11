"""Connectome-guided graph Transformer and motor-aware fusion model.

Research implementation; use deidentified inputs and review evaluation protocol.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def preprocess_connectome(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    A: (B, R, R) or (R, R)
    Steps:
      1) clamp + log1p
      2) symmetrize
      3) add self-loop
      4) D^{-1/2} A D^{-1/2}
    """
    squeeze = False
    if A.ndim == 2:
        A = A.unsqueeze(0)
        squeeze = True

    A = torch.nan_to_num(A.float(), nan=0.0, posinf=0.0, neginf=0.0)
    A = torch.clamp(A, min=0.0)
    A = torch.log1p(A)
    A = 0.5 * (A + A.transpose(-1, -2))

    B, R, _ = A.shape
    I = torch.eye(R, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, -1, -1)
    A = A + I

    deg = A.sum(dim=-1).clamp_min(eps)
    d_inv_sqrt = deg.rsqrt()
    A_norm = d_inv_sqrt.unsqueeze(-1) * A * d_inv_sqrt.unsqueeze(-2)

    if squeeze:
        A_norm = A_norm.squeeze(0)
    return A_norm


class EdgeAwareSelfAttention(nn.Module):
    """
    Multi-head self-attention with additive edge bias from connectome edge weight.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.edge_bias = nn.Sequential(
            nn.Linear(1, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_heads),
        )
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        x: (B, R, D)
        A: (B, R, R) normalized adjacency / connectome
        """
        B, R, D = x.shape

        q = self.q_proj(x).view(B, R, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,R,Hd)
        k = self.k_proj(x).view(B, R, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, R, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B,H,R,R)

        edge_b = self.edge_bias(A.unsqueeze(-1))                  # (B,R,R,H)
        edge_b = edge_b.permute(0, 3, 1, 2)                       # (B,H,R,R)
        attn = attn + edge_b

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)                               # (B,H,R,Hd)
        out = out.transpose(1, 2).contiguous().view(B, R, D)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


class GraphTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EdgeAwareSelfAttention(dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), A)
        x = x + self.ffn(self.norm2(x))
        return x


class ConnectomeSubdomainNet(nn.Module):
    """
    Subject-level connectome -> five subdomain regression heads.

    Input:
      A      : (B, R, R) connectome matrix
      lap_pe : (B, R, K) optional Laplacian positional encoding

    Output:
      y      : (B, T) normalized regression outputs
    """

    def __init__(
        self,
        num_rois: int,
        num_tasks: int = 5,
        embed_dim: int = 128,
        depth: int = 5,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        lap_pe_dim: int = 16,
    ):
        super().__init__()
        self.num_rois = num_rois
        self.num_tasks = num_tasks
        self.embed_dim = embed_dim
        self.lap_pe_dim = lap_pe_dim

        node_in_dim = num_rois + 1 + lap_pe_dim  # row fingerprint + node strength + lap_pe

        self.row_encoder = nn.Sequential(
            nn.LayerNorm(node_in_dim),
            nn.Linear(node_in_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.row_norm = nn.LayerNorm(embed_dim)

        self.blocks = nn.ModuleList([
            GraphTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        # shared global branch
        global_in_dim = embed_dim * 3  # mean + std + max
        self.global_proj = nn.Sequential(
            nn.LayerNorm(global_in_dim),
            nn.Linear(global_in_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.global_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_tasks),
        )

        # task-specific pooling
        self.task_queries = nn.Parameter(torch.randn(num_tasks, embed_dim) * 0.02)
        self.task_bias = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, 1),
            )
            for _ in range(num_tasks)
        ])

        task_feat_dim = embed_dim * 3  # local_mean + local_std + global
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(task_feat_dim),
                nn.Linear(task_feat_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, 1),
            )
            for _ in range(num_tasks)
        ])

    @staticmethod
    def _global_context(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1)
        second = (x * x).mean(dim=1)
        std = torch.sqrt(torch.clamp(second - mean * mean, min=1e-6))
        vmax = x.max(dim=1).values
        return torch.cat([mean, std, vmax], dim=-1)

    def forward(self, A: torch.Tensor, lap_pe: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        A: (B, R, R)
        lap_pe: (B, R, K) or None
        """
        if A.ndim != 3:
            raise ValueError(f"Expected A with shape (B,R,R), got {tuple(A.shape)}")
        B, R, C = A.shape
        if R != C:
            raise ValueError(f"Connectome must be square, got {tuple(A.shape)}")
        if R != self.num_rois:
            raise ValueError(f"num_rois mismatch: got {R}, expected {self.num_rois}")

        A_norm = preprocess_connectome(A)  # (B,R,R)
        strengths = A_norm.sum(dim=-1, keepdim=True)  # (B,R,1)

        if lap_pe is None:
            lap_pe = torch.zeros(B, R, self.lap_pe_dim, device=A.device, dtype=A.dtype)
        if lap_pe.shape[:2] != (B, R):
            raise ValueError(f"lap_pe shape mismatch: got {tuple(lap_pe.shape)}, expected (B,{R},K)")

        node_feat = torch.cat([A_norm, strengths, lap_pe], dim=-1)   # (B,R,R+1+K)
        x = self.row_encoder(node_feat)
        x = self.row_norm(x)

        for blk in self.blocks:
            x = blk(x, A_norm)

        global_ctx = self.global_proj(self._global_context(x))       # (B,D)
        global_out = self.global_head(global_ctx)                    # (B,T)

        outs = []
        for t in range(self.num_tasks):
            q = self.task_queries[t].view(1, 1, -1)                 # (1,1,D)
            logits = (x * q).sum(dim=-1) / math.sqrt(self.embed_dim)
            logits = logits + self.task_bias[t](x).squeeze(-1)
            attn = F.softmax(logits, dim=1)                         # (B,R)

            local_mean = torch.sum(attn.unsqueeze(-1) * x, dim=1)   # (B,D)
            second = torch.sum(attn.unsqueeze(-1) * (x * x), dim=1)
            local_std = torch.sqrt(torch.clamp(second - local_mean * local_mean, min=1e-6))

            feat = torch.cat([local_mean, local_std, global_ctx], dim=-1)
            y_t = self.heads[t](feat).squeeze(-1)                   # (B,)
            outs.append(y_t)

        task_out = torch.stack(outs, dim=-1)                        # (B,T)
        out = task_out + global_out
        return out