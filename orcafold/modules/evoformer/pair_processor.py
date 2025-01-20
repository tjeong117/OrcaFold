"""
Pair representation processing module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class TriangleAttention(nn.Module):
    """
    Triangle attention for pair representations.
    Can operate in two modes: starting node or ending node.
    """

    def __init__(
        self,
        pair_dim: int,
        num_heads: int,
        starting_node: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        assert pair_dim % num_heads == 0, "pair_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = pair_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.starting_node = starting_node

        self.to_qkv = nn.Linear(pair_dim, 3 * pair_dim, bias=False)
        self.to_out = nn.Linear(pair_dim, pair_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        pair: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pair: [batch, seq_len, seq_len, pair_dim]
            mask: [batch, seq_len, seq_len] attention mask
        Returns:
            Updated pair representation
        """
        batch, seq_len, _, _ = pair.shape

        # Generate Q, K, V
        qkv = self.to_qkv(pair)
        qkv = qkv.view(batch, seq_len, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=3)

        # Rearrange based on attention mode
        if self.starting_node:
            q = q.transpose(1, 2)  # Swap i, j dimensions for starting node attention
            mask = mask.transpose(1, 2) if mask is not None else None

        # Scaled dot-product attention
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # Apply mask if provided
        if mask is not None:
            dots = dots.masked_fill(~mask.unsqueeze(1), float('-inf'))

        # Attention weights
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Restore original shape
        if self.starting_node:
            out = out.transpose(1, 2)

        out = out.contiguous().view(batch, seq_len, seq_len, -1)
        return self.to_out(out)

class TriangleMultiplication(nn.Module):
    """
    Triangle multiplication layer for pair representations.
    Can operate in two modes: outgoing or incoming.
    """

    def __init__(
        self,
        pair_dim: int,
        dropout: float = 0.1,
        outgoing: bool = True
    ):
        super().__init__()

        self.outgoing = outgoing
        intermediate_dim = pair_dim // 2

        # Gates
        self.g1 = nn.Linear(pair_dim, intermediate_dim)
        self.g2 = nn.Linear(pair_dim, intermediate_dim)

        # Output projection
        self.projection = nn.Linear(intermediate_dim, pair_dim)

        self.layer_norm = nn.LayerNorm(pair_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        pair: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pair: [batch, seq_len, seq_len, pair_dim]
            mask: [batch, seq_len, seq_len] attention mask
        Returns:
            Updated pair representation
        """
        batch, seq_len, _, _ = pair.shape

        # Apply gates
        g1 = self.g1(pair)
        g2 = self.g2(pair)

        # Perform triangle multiplication
        if self.outgoing:
            # Einstein notation for matrix multiplication along sequence dimension
            p = torch.einsum('bikd,bkjd->bijd', g1, g2)
        else:
            p = torch.einsum('bkid,bjkd->bijd', g1, g2)

        # Apply mask if provided
        if mask is not None:
            p = p * mask.unsqueeze(-1)

        # Project back to pair dimension
        p = self.projection(p)
        p = self.dropout(p)

        return self.layer_norm(pair + p)

class PairTransition(nn.Module):
    """
    Transition layer for pair representations.
    Applies position-wise feed-forward network to each pair independently.
    """

    def __init__(
        self,
        pair_dim: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(pair_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, pair_dim)
        )

        self.norm = nn.LayerNorm(pair_dim)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        return self.ff(self.norm(pair))
