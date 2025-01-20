"""
TM-score (Template Modeling Score) prediction module.
Implements both TM-score calculation and prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math

class TMScorePredictor(nn.Module):
    """
    Predicts TM-score for structure quality assessment.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_bins: int = 31,  # Number of bins for distance prediction
        max_dist: float = 30.0,  # Maximum distance in Angstroms
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_bins = num_bins
        self.max_dist = max_dist
        self.d0_default = 5.0  # Default d0 parameter for TM-score

        # Distance prediction network
        self.distance_predictor = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_bins)
        )

        # Pair feature network
        self.pair_embedder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)

        # Register distance bins
        self.register_buffer(
            'distance_bins',
            torch.linspace(0, max_dist, num_bins)
        )

    def _get_d0(self, sequence_length: int) -> float:
        """
        Compute d0 parameter based on sequence length.
        Formula from original TM-score paper.
        """
        return 1.24 * (sequence_length - 15) ** (1/3) - 1.8

    def _compute_pair_features(
        self,
        representations: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise features from sequence representations."""
        # Create pairs of representations
        n_seq = representations.size(1)
        rep_i = representations.unsqueeze(2).expand(-1, -1, n_seq, -1)
        rep_j = representations.unsqueeze(1).expand(-1, n_seq, -1, -1)

        # Concatenate pair representations
        pairs = torch.cat([rep_i, rep_j], dim=-1)

        return self.pair_embedder(pairs)

    def forward(
        self,
        representations: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict TM-score and pairwise distances.

        Args:
            representations: [batch, num_residues, input_dim] sequence representations
            mask: Optional [batch, num_residues] sequence mask

        Returns:
            Dictionary containing:
            - 'distance_logits': Distance prediction logits
            - 'distance_probs': Distance probabilities
            - 'predicted_tm_score': Predicted TM-score
            - 'distance_matrix': Predicted distance matrix
        """
        batch_size, num_residues, _ = representations.shape

        # Normalize input
        representations = self.layer_norm(representations)

        # Compute pair features
        pair_features = self._compute_pair_features(representations)

        # Predict distances
        logits = self.distance_predictor(pair_features)

        # Convert to probabilities
        distance_probs = F.softmax(logits, dim=-1)

        # Compute expected distances
        # Expand bins to match probability dimensions
        expanded_bins = self.distance_bins.view(1, 1, 1, -1).expand(
            distance_probs.shape[0],  # batch
            distance_probs.shape[1],  # num_residues
            distance_probs.shape[2],  # num_residues
            -1  # num_bins
        )
        distances = torch.sum(distance_probs * expanded_bins, dim=-1)

        # Compute TM-score
        d0 = self._get_d0(num_residues)
        tm_score = self._compute_tm_score(distances, d0, mask)

        return {
            'distance_logits': logits,
            'distance_probs': distance_probs,
            'predicted_tm_score': tm_score,
            'distance_matrix': distances
        }

    def _compute_tm_score(
        self,
        distances: torch.Tensor,
        d0: float,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute TM-score from predicted distances.

        Args:
            distances: [batch, num_residues, num_residues] predicted distances
            d0: Normalization parameter
            mask: Optional [batch, num_residues] sequence mask
        """
        # Compute TM-score components
        d0_squared = d0 * d0
        tm_components = 1 / (1 + (distances * distances) / d0_squared)

        # Apply mask if provided
        if mask is not None:
            mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
            tm_components = tm_components * mask_2d
            normalizer = mask_2d.sum(dim=(1, 2))
        else:
            # Create tensor directly using float
            normalizer = torch.full(
                (distances.size(0),),
                float(distances.size(1) * distances.size(2)),
                device=distances.device
            )

        # Average to get final TM-score
        tm_score = tm_components.sum(dim=(1, 2)) / normalizer

        return tm_score

class AlignedErrorPredictor(nn.Module):
    """
    Predicts aligned distance error between predicted and true structures.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_bins: int = 64,
        max_error: float = 30.0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_bins = num_bins
        self.max_error = max_error

        # Error prediction network
        self.error_predictor = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_bins)
        )

        # Register error bins
        self.register_buffer(
            'error_bins',
            torch.linspace(0, max_error, num_bins)
        )

        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        representations: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict aligned error between structures.

        Args:
            representations: [batch, num_residues, input_dim] sequence representations
            mask: Optional [batch, num_residues] sequence mask

        Returns:
            Dictionary containing:
            - 'error_logits': Error prediction logits
            - 'error_probs': Error probabilities
            - 'predicted_error': Expected error values
        """
        # Normalize input
        representations = self.layer_norm(representations)

        # Create pair representations
        n_seq = representations.size(1)
        rep_i = representations.unsqueeze(2).expand(-1, -1, n_seq, -1)
        rep_j = representations.unsqueeze(1).expand(-1, n_seq, -1, -1)
        pairs = torch.cat([rep_i, rep_j], dim=-1)

        # Predict errors
        logits = self.error_predictor(pairs)
        error_probs = F.softmax(logits, dim=-1)

        # Compute expected errors
        # Expand bins to match probability dimensions
        expanded_error_bins = self.error_bins.view(1, 1, 1, -1).expand(
            error_probs.shape[0],  # batch
            error_probs.shape[1],  # num_residues
            error_probs.shape[2],  # num_residues
            -1  # num_bins
        )
        predicted_error = torch.sum(error_probs * expanded_error_bins, dim=-1)

        # Apply mask if provided
        if mask is not None:
            mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
            predicted_error = predicted_error * mask_2d

        return {
            'error_logits': logits,
            'error_probs': error_probs,
            'predicted_error': predicted_error
        }

def compute_tm_score_loss(
    logits: torch.Tensor,
    target_distances: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Compute loss for TM-score prediction.

    Args:
        logits: [batch, num_residues, num_residues, num_bins] prediction logits
        target_distances: [batch, num_residues, num_residues] true distances
        mask: Optional [batch, num_residues] sequence mask
        eps: Small value for numerical stability
    """
    num_bins = logits.shape[-1]
    device = logits.device

    # Create distance bins
    bins = torch.linspace(0, 30.0, num_bins, device=device)

    # Convert target distances to bin indices
    target_bins = torch.bucketize(target_distances, bins)

    # Compute cross entropy loss
    loss = F.cross_entropy(
        logits.view(-1, num_bins),
        target_bins.view(-1),
        reduction='none'
    ).view_as(target_distances)

    # Apply mask if provided
    if mask is not None:
        mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
        loss = loss * mask_2d
        loss = loss.sum() / (mask_2d.sum() + eps)
    else:
        loss = loss.mean()

    return loss
