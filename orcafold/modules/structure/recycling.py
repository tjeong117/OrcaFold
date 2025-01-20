"""
Recycling module for iterative structure refinement.
Implements the recycling mechanism used in AlphaFold-style models.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Union

class RecyclingModule(nn.Module):
    """
    Handles recycling of structure predictions for iterative refinement.
    Combines information from previous iterations with current predictions.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_recycling_iters: int = 3,
        use_structure_recycling: bool = True,
        use_sequence_recycling: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_recycling_iters = num_recycling_iters
        self.use_structure_recycling = use_structure_recycling
        self.use_sequence_recycling = use_sequence_recycling

        # Structure recycling components
        if use_structure_recycling:
            self.structure_embedding = nn.Sequential(
                nn.Linear(9, hidden_dim // 2),  # 9 = 3 (coords) * 3 (N, CA, C)
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim)
            )

            self.structure_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout
            )

            self.structure_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )

        # Sequence recycling components
        if use_sequence_recycling:
            self.sequence_embedding = nn.Linear(hidden_dim, hidden_dim)

            self.sequence_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout
            )

            self.sequence_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )

        # Final processing
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

        # Confidence prediction for recycling
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def _process_structure(
        self,
        hidden: torch.Tensor,
        coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process structure information for recycling.

        Args:
            hidden: [batch, num_residues, hidden_dim] Current hidden representations
            coords: [batch, num_residues, 3, 3] Backbone coordinates (N, CA, C)
            mask: [batch, num_residues] Attention mask

        Returns:
            Updated hidden representations incorporating structural information
        """
        # Flatten coordinates
        batch, num_residues = coords.shape[:2]
        coords_flat = coords.reshape(batch, num_residues, -1)

        # Embed coordinates
        struct_features = self.structure_embedding(coords_flat)

        # Apply attention
        struct_features = struct_features.transpose(0, 1)  # [num_residues, batch, hidden_dim]
        hidden_transposed = hidden.transpose(0, 1)

        key_padding_mask = ~mask if mask is not None else None

        struct_attn, _ = self.structure_attention(
            query=hidden_transposed,
            key=struct_features,
            value=struct_features,
            key_padding_mask=key_padding_mask
        )

        # Process attention output
        struct_attn = struct_attn.transpose(0, 1)  # [batch, num_residues, hidden_dim]
        combined = torch.cat([hidden, struct_attn], dim=-1)

        return self.structure_mlp(combined)

    def _process_sequence(
        self,
        hidden: torch.Tensor,
        prev_hidden: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process sequence information from previous iteration.

        Args:
            hidden: [batch, num_residues, hidden_dim] Current hidden representations
            prev_hidden: [batch, num_residues, hidden_dim] Previous hidden representations
            mask: [batch, num_residues] Attention mask

        Returns:
            Updated hidden representations incorporating sequence information
        """
        # Embed previous hidden states
        prev_features = self.sequence_embedding(prev_hidden)

        # Apply attention
        prev_features = prev_features.transpose(0, 1)
        hidden_transposed = hidden.transpose(0, 1)

        key_padding_mask = ~mask if mask is not None else None

        seq_attn, _ = self.sequence_attention(
            query=hidden_transposed,
            key=prev_features,
            value=prev_features,
            key_padding_mask=key_padding_mask
        )

        # Process attention output
        seq_attn = seq_attn.transpose(0, 1)
        combined = torch.cat([hidden, seq_attn], dim=-1)

        return self.sequence_mlp(combined)

    def forward(
        self,
        hidden: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        prev_hidden: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process one recycling iteration.

        Args:
            hidden: [batch, num_residues, hidden_dim] Current hidden representations
            coords: [batch, num_residues, 3, 3] Current backbone coordinates
            prev_hidden: [batch, num_residues, hidden_dim] Previous hidden representations
            mask: [batch, num_residues] Attention mask

        Returns:
            Dictionary containing:
            - 'hidden': Updated hidden representations
            - 'confidence': Recycling confidence scores
        """
        updated_hidden = hidden

        # Process structure if available and enabled
        if self.use_structure_recycling and coords is not None:
            struct_update = self._process_structure(updated_hidden, coords, mask)
            updated_hidden = updated_hidden + struct_update

        # Process sequence if available and enabled
        if self.use_sequence_recycling and prev_hidden is not None:
            seq_update = self._process_sequence(updated_hidden, prev_hidden, mask)
            updated_hidden = updated_hidden + seq_update

        # Final processing
        updated_hidden = self.layer_norm(updated_hidden)
        updated_hidden = self.output_projection(updated_hidden)

        # Predict confidence scores
        confidence = self.confidence_predictor(updated_hidden)

        if mask is not None:
            updated_hidden = updated_hidden * mask.unsqueeze(-1)
            confidence = confidence * mask.unsqueeze(-1)

        return {
            'hidden': updated_hidden,
            'confidence': confidence
        }
