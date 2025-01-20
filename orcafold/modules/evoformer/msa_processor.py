"""
Multiple Sequence Alignment (MSA) processing module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

class MSARowAttentionWithPairBias(nn.Module):
    """
    MSA row-wise attention with pair bias.
    Processes each sequence in the MSA independently while incorporating
    pair representation information.
    """

    def __init__(
        self,
        msa_dim: int,
        pair_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        assert msa_dim % num_heads == 0, "msa_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = msa_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        # Attention projections
        self.to_qkv = nn.Linear(msa_dim, 3 * msa_dim, bias=False)

        # Pair bias projection
        self.pair_bias = nn.Linear(pair_dim, num_heads)

        # Output projection
        self.to_out = nn.Linear(msa_dim, msa_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        msa: torch.Tensor,
        pair: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            msa: [batch, num_seqs, seq_len, msa_dim]
            pair: [batch, seq_len, seq_len, pair_dim]
            mask: [batch, num_seqs, seq_len] attention mask
        Returns:
            Updated MSA representation
        """
        batch, num_seqs, seq_len, _ = msa.shape

        # Generate Q, K, V
        qkv = self.to_qkv(msa)
        qkv = qkv.view(batch, num_seqs, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=3)

        # Generate attention bias from pair representation
        pair_bias = self.pair_bias(pair)  # [batch, seq_len, seq_len, num_heads]
        pair_bias = pair_bias.permute(0, 3, 1, 2)  # [batch, num_heads, seq_len, seq_len]

        # Scaled dot-product attention
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        dots = dots + pair_bias.unsqueeze(1)  # Add pair bias

        # Apply mask if provided
        if mask is not None:
            dots = dots.masked_fill(~mask.unsqueeze(1), float('-inf'))

        # Attention weights
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Merge heads and project
        out = out.transpose(2, 3).contiguous()
        out = out.view(batch, num_seqs, seq_len, -1)
        return self.to_out(out)

class MSAColumnAttention(nn.Module):
    """
    MSA column-wise attention.
    Processes each position in the sequence by attending to all sequences at that position.
    """

    def __init__(
        self,
        msa_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        assert msa_dim % num_heads == 0, "msa_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = msa_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(msa_dim, 3 * msa_dim, bias=False)
        self.to_out = nn.Linear(msa_dim, msa_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        msa: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            msa: [batch, num_seqs, seq_len, msa_dim]
            mask: [batch, num_seqs, seq_len] attention mask
        Returns:
            Updated MSA representation
        """
        batch, num_seqs, seq_len, _ = msa.shape

        # Generate Q, K, V
        qkv = self.to_qkv(msa)
        qkv = qkv.view(batch, num_seqs, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=3)

        # Transpose for column attention
        q = q.transpose(1, 2)  # [batch, seq_len, num_seqs, num_heads, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # Apply mask if provided
        if mask is not None:
            mask = mask.transpose(1, 2)  # [batch, seq_len, num_seqs]
            dots = dots.masked_fill(~mask.unsqueeze(1), float('-inf'))

        # Attention weights
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Restore original shape and merge heads
        out = out.transpose(1, 2).contiguous()  # [batch, num_seqs, seq_len, num_heads, head_dim]
        out = out.view(batch, num_seqs, seq_len, -1)

        return self.to_out(out)

class MSATransition(nn.Module):
    """
    Transition layer for MSA representations.
    Applies position-wise feed-forward network to each MSA position independently.
    """

    def __init__(
        self,
        msa_dim: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(msa_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, msa_dim)
        )

        self.norm = nn.LayerNorm(msa_dim)

    def forward(self, msa: torch.Tensor) -> torch.Tensor:
        return self.ff(self.norm(msa))
