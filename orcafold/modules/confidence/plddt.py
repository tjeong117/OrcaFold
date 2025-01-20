"""
pLDDT (predicted Local Distance Test) confidence scoring module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

class PerResidueLDDTPredictor(nn.Module):
    """
    Predicts per-residue LDDT scores for confidence estimation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_bins: int = 50,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_bins = num_bins

        # Main prediction network
        self.prediction_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_bins)
        )

        # Distance thresholds for LDDT calculation (0.5, 1, 2, 4 Angstroms)
        self.register_buffer(
            'distance_thresholds',
            torch.tensor([0.5, 1.0, 2.0, 4.0])
        )

        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        representations: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict per-residue LDDT scores.

        Args:
            representations: [batch, num_residues, input_dim] hidden representations
            mask: [batch, num_residues] sequence mask

        Returns:
            Dictionary containing:
            - 'logits': Raw prediction logits
            - 'confidences': Predicted pLDDT scores
            - 'bins': Binned predictions
        """
        # Normalize inputs
        representations = self.layer_norm(representations)

        # Predict LDDT bins
        logits = self.prediction_net(representations)

        # Convert to probabilities
        bins = F.softmax(logits, dim=-1)

        # Calculate pLDDT scores (mean of bin values)
        bin_values = torch.linspace(0, 100, self.num_bins, device=logits.device)
        confidences = torch.sum(bins * bin_values, dim=-1)

        # Apply mask if provided
        if mask is not None:
            confidences = confidences * mask
            bins = bins * mask.unsqueeze(-1)

        return {
            'logits': logits,
            'confidences': confidences,
            'bins': bins
        }

class StructureQualityPredictor(nn.Module):
    """
    Predicts overall structure quality and per-residue confidence scores.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_bins: int = 50,
        use_structure_features: bool = True
    ):
        super().__init__()

        self.use_structure_features = use_structure_features
        feature_dim = input_dim * 2 if use_structure_features else input_dim

        # Per-residue LDDT predictor
        self.plddt_predictor = PerResidueLDDTPredictor(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_bins=num_bins
        )

        if use_structure_features:
            # Structure feature extractor
            self.structure_encoder = nn.Sequential(
                nn.Linear(9, hidden_dim),  # 9 = 3 (coords) * 3 (N, CA, C)
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )

        # Global quality predictor
        self.global_quality_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def _process_structure_features(
        self,
        coords: torch.Tensor
    ) -> torch.Tensor:
        """Process backbone coordinates into structure features."""
        batch_size, num_residues = coords.shape[:2]

        # Flatten coordinates
        coords_flat = coords.view(batch_size, num_residues, -1)

        # Extract structure features
        return self.structure_encoder(coords_flat)

    def forward(
        self,
        representations: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict structure quality scores.

        Args:
            representations: [batch, num_residues, input_dim] hidden representations
            coords: Optional [batch, num_residues, 3, 3] backbone coordinates
            mask: Optional [batch, num_residues] sequence mask

        Returns:
            Dictionary containing:
            - 'plddt': Per-residue confidence scores
            - 'global_quality': Global structure quality score
            - 'plddt_bins': Binned pLDDT predictions
        """
        # Combine sequence and structure features if available
        features = representations
        if self.use_structure_features and coords is not None:
            structure_features = self._process_structure_features(coords)
            features = torch.cat([features, structure_features], dim=-1)

        # Predict per-residue confidence
        plddt_outputs = self.plddt_predictor(features, mask)

        # Predict global quality
        global_quality = self.global_quality_predictor(
            representations.mean(dim=1)
        )

        return {
            'plddt': plddt_outputs['confidences'],
            'plddt_bins': plddt_outputs['bins'],
            'plddt_logits': plddt_outputs['logits'],
            'global_quality': global_quality
        }

def compute_plddt_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Compute loss for pLDDT prediction.

    Args:
        logits: [batch, num_residues, num_bins] prediction logits
        target: [batch, num_residues] true LDDT scores (0-100)
        mask: Optional [batch, num_residues] sequence mask
        eps: Small value for numerical stability

    Returns:
        Loss value
    """
    num_bins = logits.shape[-1]
    bin_values = torch.linspace(0, 100, num_bins, device=logits.device)

    # Convert target scores to bin indices
    target_bins = torch.bucketize(target, bin_values)

    # Compute cross entropy loss
    loss = F.cross_entropy(
        logits.view(-1, num_bins),
        target_bins.view(-1),
        reduction='none'
    ).view_as(target)

    # Apply mask if provided
    if mask is not None:
        loss = loss * mask
        loss = loss.sum() / (mask.sum() + eps)
    else:
        loss = loss.mean()

    return loss
