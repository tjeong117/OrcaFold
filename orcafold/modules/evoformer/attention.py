"""
Main Evoformer attention module and stack implementation.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from .msa_processor import MSARowAttentionWithPairBias, MSAColumnAttention, MSATransition
from .pair_processor import TriangleAttention, TriangleMultiplication, PairTransition

class EvoformerBlock(nn.Module):
    """
    Single Evoformer block combining MSA and pair processing.
    """

    def __init__(
        self,
        msa_dim: int,
        pair_dim: int,
        num_heads: int,
        ff_dim: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()

        # MSA processing layers
        self.msa_row_attention = MSARowAttentionWithPairBias(
            msa_dim=msa_dim,
            pair_dim=pair_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.msa_column_attention = MSAColumnAttention(
            msa_dim=msa_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.msa_transition = MSATransition(
            msa_dim=msa_dim,
            ff_dim=ff_dim,
            dropout=dropout
        )

        # Pair processing layers
        self.triangle_attention_starting = TriangleAttention(
            pair_dim=pair_dim,
            num_heads=num_heads,
            starting_node=True,
            dropout=dropout
        )
        self.triangle_attention_ending = TriangleAttention(
            pair_dim=pair_dim,
            num_heads=num_heads,
            starting_node=False,
            dropout=dropout
        )
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            pair_dim=pair_dim,
            dropout=dropout,
            outgoing=True
        )
        self.triangle_multiplication_incoming = TriangleMultiplication(
            pair_dim=pair_dim,
            dropout=dropout,
            outgoing=False
        )
        self.pair_transition = PairTransition(
            pair_dim=pair_dim,
            ff_dim=ff_dim,
            dropout=dropout
        )

        # Layer normalization
        self.msa_norm = nn.LayerNorm(msa_dim)
        self.pair_norm = nn.LayerNorm(pair_dim)

    def forward(
        self,
        msa: torch.Tensor,
        pair: torch.Tensor,
        msa_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process one Evoformer block.

        Args:
            msa: [batch, num_seqs, seq_len, msa_dim] MSA representation
            pair: [batch, seq_len, seq_len, pair_dim] pair representation
            msa_mask: [batch, num_seqs, seq_len] MSA attention mask
            pair_mask: [batch, seq_len, seq_len] pair attention mask

        Returns:
            Updated MSA and pair representations
        """
        # MSA processing
        msa_update = self.msa_norm(msa)
        msa = msa + self.msa_row_attention(msa_update, pair, msa_mask)
        msa = msa + self.msa_column_attention(self.msa_norm(msa), msa_mask)
        msa = msa + self.msa_transition(self.msa_norm(msa))

        # Pair processing
        pair_update = self.pair_norm(pair)
        pair = pair + self.triangle_attention_starting(pair_update, pair_mask)
        pair = pair + self.triangle_attention_ending(self.pair_norm(pair), pair_mask)
        pair = pair + self.triangle_multiplication_outgoing(self.pair_norm(pair), pair_mask)
        pair = pair + self.triangle_multiplication_incoming(self.pair_norm(pair), pair_mask)
        pair = pair + self.pair_transition(self.pair_norm(pair))

        return msa, pair

class EvoformerStack(nn.Module):
    """
    Stack of Evoformer blocks for iterative refinement of MSA and pair representations.
    """

    def __init__(
        self,
        num_blocks: int,
        msa_dim: int,
        pair_dim: int,
        num_heads: int,
        ff_dim: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            EvoformerBlock(
                msa_dim=msa_dim,
                pair_dim=pair_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        msa: torch.Tensor,
        pair: torch.Tensor,
        msa_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process the complete Evoformer stack.

        Args:
            msa: [batch, num_seqs, seq_len, msa_dim] MSA representation
            pair: [batch, seq_len, seq_len, pair_dim] pair representation
            msa_mask: [batch, num_seqs, seq_len] MSA attention mask
            pair_mask: [batch, seq_len, seq_len] pair attention mask

        Returns:
            Final MSA and pair representations
        """
        for block in self.blocks:
            msa, pair = block(msa, pair, msa_mask, pair_mask)

        return msa, pair
