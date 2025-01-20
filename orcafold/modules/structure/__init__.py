"""
Structure module for OrcaFold protein structure prediction.
Contains components for backbone and sidechain prediction with recycling support.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from .backbone import BackboneUpdate, InvariantPointAttention
from .sidechain import SidechainPredictor
from .recycling import RecyclingModule

class StructureModule(nn.Module):
    """
    Main structure prediction module combining backbone, sidechain,
    and recycling components.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 8,
        num_recycling_iters: int = 3,
        use_structure_recycling: bool = True,
        use_sequence_recycling: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        # Backbone prediction
        self.backbone = BackboneUpdate(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # Sidechain prediction
        self.sidechain = SidechainPredictor(
            hidden_dim=hidden_dim,
            num_angles_per_residue=4,
            num_bins=50,
            use_rotamers=True
        )

        # Recycling
        self.recycling = RecyclingModule(
            hidden_dim=hidden_dim,
            num_recycling_iters=num_recycling_iters,
            use_structure_recycling=use_structure_recycling,
            use_sequence_recycling=use_sequence_recycling,
            dropout=dropout
        )

        # Additional layers
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.frame_embedding = nn.Linear(6, hidden_dim)  # 6 = sin/cos of 3 angles

    def _get_frame_features(self, backbone_coords: torch.Tensor) -> torch.Tensor:
        """Convert backbone coordinates to frame features."""
        # Extract vectors between backbone atoms
        n_ca = backbone_coords[..., 1, :] - backbone_coords[..., 0, :]  # CA - N
        ca_c = backbone_coords[..., 2, :] - backbone_coords[..., 1, :]  # C - CA

        # Normalize vectors
        n_ca = nn.functional.normalize(n_ca, dim=-1)
        ca_c = nn.functional.normalize(ca_c, dim=-1)

        # Compute cross product for perpendicular vector
        perp = torch.cross(n_ca, ca_c, dim=-1)
        perp = nn.functional.normalize(perp, dim=-1)

        # Stack vectors to form frame features
        frames = torch.stack([n_ca, ca_c, perp], dim=-2)

        # Convert to angles (simplified representation)
        angles = torch.atan2(frames[..., 1], frames[..., 0])  # y, x components

        # Convert to sin/cos features
        sin_features = torch.sin(angles)
        cos_features = torch.cos(angles)

        frame_features = torch.cat([sin_features, cos_features], dim=-1)
        return frame_features

    def forward(
        self,
        hidden: torch.Tensor,
        prev_hidden: Optional[torch.Tensor] = None,
        prev_coords: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        residue_type: Optional[torch.Tensor] = None
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Predict protein structure given sequence representations.

        Args:
            hidden: [batch, num_residues, hidden_dim] Sequence representations
            prev_hidden: Optional previous hidden states for recycling
            prev_coords: Optional previous coordinates for recycling
            mask: [batch, num_residues] Sequence mask
            residue_type: [batch, num_residues] Amino acid types

        Returns:
            Dictionary containing:
            - 'backbone_coords': Predicted backbone coordinates
            - 'sidechain_coords': Predicted sidechain coordinates
            - 'angles': Predicted torsion angles
            - 'confidence': Structure confidence scores
            - 'hidden': Updated hidden representations
        """
        batch_size, num_residues, _ = hidden.shape
        device = hidden.device

        # Initialize coordinates if not provided
        if prev_coords is None:
            prev_coords = torch.zeros(
                batch_size, num_residues, 3, 3,
                device=device
            )

        # Extract frame features from previous coordinates
        frame_features = self._get_frame_features(prev_coords)
        frame_embedding = self.frame_embedding(frame_features)

        # Add frame information to hidden states
        hidden = hidden + frame_embedding

        # Initialize recycling_output as None
        recycling_output = None

        # Recycling
        if prev_hidden is not None:
            recycling_output = self.recycling(
                hidden=hidden,
                coords=prev_coords,
                prev_hidden=prev_hidden,
                mask=mask
            )
            hidden = recycling_output['hidden']

        # Predict backbone structure
        backbone_output = self.backbone(
            hidden=self.layer_norm(hidden),
            xyz=prev_coords,
            mask=mask
        )

        backbone_coords = backbone_output['xyz']
        angles = backbone_output['angles']

        # Predict sidechain structure
        if residue_type is not None:
            sidechain_output = self.sidechain(
                hidden=self.layer_norm(hidden),
                backbone_coords=backbone_coords,
                residue_type=residue_type,
                mask=mask
            )
        else:
            sidechain_output = {
                'coordinates': None,
                'angles': None,
                'uncertainties': None,
                'rotamers': None
            }

        return {
            'backbone_coords': backbone_coords,
            'sidechain_coords': sidechain_output['coordinates'],
            'backbone_angles': angles,
            'sidechain_angles': sidechain_output['angles'],
            'confidence': recycling_output['confidence'] if recycling_output is not None else None,
            'hidden': hidden,
            'uncertainties': sidechain_output['uncertainties'],
            'rotamers': sidechain_output['rotamers']
        }

__all__ = [
    'StructureModule',
    'BackboneUpdate',
    'InvariantPointAttention',
    'SidechainPredictor',
    'RecyclingModule'
]
