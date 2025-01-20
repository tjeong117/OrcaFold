"""
Confidence prediction module for OrcaFold.
Combines pLDDT and TM-score prediction for structure quality assessment.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Union

from .plddt import PerResidueLDDTPredictor, StructureQualityPredictor
from .tm_score import TMScorePredictor, AlignedErrorPredictor

class ConfidencePredictor(nn.Module):
    """
    Combined confidence prediction module integrating pLDDT and TM-score predictions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_bins_lddt: int = 50,
        num_bins_tm: int = 31,
        max_error: float = 30.0,
        dropout: float = 0.1
    ):
        super().__init__()

        # pLDDT prediction
        self.quality_predictor = StructureQualityPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_bins=num_bins_lddt,
            use_structure_features=True
        )

        # TM-score prediction
        self.tm_predictor = TMScorePredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_bins=num_bins_tm,
            max_dist=max_error,
            dropout=dropout
        )

        # Aligned error prediction
        self.error_predictor = AlignedErrorPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_bins=num_bins_tm,
            max_error=max_error,
            dropout=dropout
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        representations: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Predict structure confidence metrics.

        Args:
            representations: [batch, num_residues, input_dim] sequence representations
            coords: Optional [batch, num_residues, 3, 3] backbone coordinates
            mask: Optional [batch, num_residues] sequence mask

        Returns:
            Dictionary containing:
            - 'plddt': Per-residue pLDDT scores
            - 'tm_score': Predicted TM-score
            - 'aligned_error': Predicted aligned error
            - 'global_quality': Global structure quality score
            - 'confidence_metrics': Additional confidence metrics
        """
        # Normalize input
        representations = self.layer_norm(representations)

        # Get pLDDT and quality predictions
        quality_outputs = self.quality_predictor(
            representations=representations,
            coords=coords,
            mask=mask
        )

        # Get TM-score predictions
        tm_outputs = self.tm_predictor(
            representations=representations,
            mask=mask
        )

        # Get aligned error predictions
        error_outputs = self.error_predictor(
            representations=representations,
            mask=mask
        )

        # Combine all confidence metrics
        confidence_metrics = {
            'plddt_bins': quality_outputs['plddt_bins'],
            'distance_matrix': tm_outputs['distance_matrix'],
            'error_distribution': error_outputs['error_probs']
        }

        return {
            'plddt': quality_outputs['plddt'],
            'tm_score': tm_outputs['predicted_tm_score'],
            'aligned_error': error_outputs['predicted_error'],
            'global_quality': quality_outputs['global_quality'],
            'confidence_metrics': confidence_metrics
        }

# Export classes and functions
__all__ = [
    'ConfidencePredictor',
    'PerResidueLDDTPredictor',
    'StructureQualityPredictor',
    'TMScorePredictor',
    'AlignedErrorPredictor'
]
