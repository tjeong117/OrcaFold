"""
Sidechain structure prediction module.
Predicts protein sidechain coordinates and rotamers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple

# Constants for amino acid sidechain properties
MAX_CHI_ANGLES = {
    'ALA': 0, 'GLY': 0, 'SER': 1, 'CYS': 1, 'VAL': 1, 'THR': 1, 'LEU': 2,
    'ILE': 2, 'ASP': 2, 'PHE': 2, 'HIS': 2, 'TYR': 2, 'TRP': 2, 'ASN': 2,
    'MET': 3, 'GLU': 3, 'GLN': 3, 'LYS': 4, 'ARG': 4
}

class SidechainPredictor(nn.Module):
    """
    Predicts protein sidechain conformations given backbone coordinates and sequence features.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_angles_per_residue: int = 4,  # Maximum number of chi angles
        num_bins: int = 50,  # Number of bins for angle discretization
        use_rotamers: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_angles_per_residue = num_angles_per_residue
        self.num_bins = num_bins
        self.use_rotamers = use_rotamers

        # Angle prediction network
        self.angle_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_angles_per_residue * num_bins)
        )

        # Optional rotamer prediction
        if use_rotamers:
            self.rotamer_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 50)  # Number of common rotamer states
            )

        # Position refinement
        self.position_refiner = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

        # Uncertainty prediction
        self.uncertainty_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_angles_per_residue)
        )

        # Local structure attention
        self.local_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )

        # Final layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def _get_residue_chi_mask(self, residue_type: torch.Tensor) -> torch.Tensor:
        """Create mask for valid chi angles based on residue type."""
        batch, num_residues = residue_type.shape
        chi_mask = torch.zeros(
            batch, num_residues, self.num_angles_per_residue,
            device=residue_type.device, dtype=torch.bool
        )

        for aa, num_chi in MAX_CHI_ANGLES.items():
            mask = (residue_type == aa)
            if num_chi > 0:
                chi_mask[mask, :num_chi] = True

        return chi_mask

    def _convert_angles_to_coordinates(
        self,
        angles: torch.Tensor,
        backbone_coords: torch.Tensor,
        residue_type: torch.Tensor,
        chi_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert predicted angles to 3D coordinates using residue-specific templates.

        Args:
            angles: [batch, num_residues, num_chi] predicted chi angles
            backbone_coords: [batch, num_residues, 3] backbone coordinates (N, CA, C)
            residue_type: [batch, num_residues] amino acid types
            chi_mask: [batch, num_residues, num_chi] mask for valid chi angles

        Returns:
            [batch, num_residues, max_atoms, 3] predicted sidechain coordinates
        """
        batch, num_residues = angles.shape[:2]
        max_atoms = 14  # Maximum number of atoms in any amino acid

        # Initialize coordinate tensor
        coords = torch.zeros(batch, num_residues, max_atoms, 3, device=angles.device)

        # Copy backbone coordinates
        coords[:, :, :3] = backbone_coords

        # Place CB atoms (beta carbon)
        cb_coords = self._place_cb_atoms(backbone_coords)
        coords[:, :, 3] = cb_coords

        # Apply chi angle rotations for each residue type
        for aa, num_chi in MAX_CHI_ANGLES.items():
            if num_chi == 0:
                continue

            aa_mask = (residue_type == aa)
            if not aa_mask.any():
                continue

            # Get template coordinates for this amino acid
            template = self._get_aa_template(aa)

            # Apply chi angle rotations
            aa_coords = self._apply_chi_rotations(
                angles[aa_mask, :num_chi],
                cb_coords[aa_mask],
                template,
                chi_mask[aa_mask, :num_chi]
            )

            # Store coordinates
            coords[aa_mask, 4:4+template.shape[0]] = aa_coords

        return coords

    def _place_cb_atoms(self, backbone_coords: torch.Tensor) -> torch.Tensor:
        """Place beta carbon atoms based on backbone geometry."""
        # Extract N, CA, C coordinates
        n_coords = backbone_coords[:, :, 0]
        ca_coords = backbone_coords[:, :, 1]
        c_coords = backbone_coords[:, :, 2]

        # Calculate CB position using ideal geometry
        # (simplified - in practice would use more precise geometric calculations)
        cb_direction = F.normalize(n_coords - c_coords, dim=-1)
        cb_coords = ca_coords + 1.5 * cb_direction  # 1.5Å is approximate CB distance

        return cb_coords

    def _get_aa_template(self, aa: str) -> torch.Tensor:
        """Get template coordinates for an amino acid sidechain."""
        # In practice, would load pre-computed templates for each amino acid
        # Here's a simplified version
        template = torch.zeros(10, 3)  # Maximum 10 atoms per sidechain
        # Would populate with ideal sidechain geometry
        return template

    def _apply_chi_rotations(
        self,
        angles: torch.Tensor,
        cb_coords: torch.Tensor,
        template: torch.Tensor,
        chi_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply chi angle rotations to template coordinates."""
        coords = template.clone()

        for i, angle in enumerate(angles.unbind(-1)):
            if not chi_mask[:, i].any():
                continue

            rotation = self._get_rotation_matrix(angle)
            coords[chi_mask[:, i]] = torch.matmul(
                rotation[chi_mask[:, i]],
                coords[chi_mask[:, i]].unsqueeze(-1)
            ).squeeze(-1)

        return coords + cb_coords.unsqueeze(1)

    def _get_rotation_matrix(self, angle: torch.Tensor) -> torch.Tensor:
        """Generate rotation matrices from angles."""
        cos_theta = torch.cos(angle)
        sin_theta = torch.sin(angle)
        zeros = torch.zeros_like(angle)
        ones = torch.ones_like(angle)

        R = torch.stack([
            torch.stack([cos_theta, -sin_theta, zeros], dim=-1),
            torch.stack([sin_theta, cos_theta, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1)
        ], dim=-2)

        return R

    def forward(
        self,
        hidden: torch.Tensor,
        backbone_coords: torch.Tensor,
        residue_type: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Predict sidechain conformations.

        Args:
            hidden: [batch, num_residues, hidden_dim] hidden representations
            backbone_coords: [batch, num_residues, 3, 3] backbone coordinates (N, CA, C)
            residue_type: [batch, num_residues] residue type indices
            mask: [batch, num_residues] prediction mask

        Returns:
            Dictionary containing:
            - 'angles': Predicted chi angles
            - 'coordinates': Predicted sidechain coordinates
            - 'uncertainties': Predicted uncertainties
            - 'rotamers': Rotamer predictions (if enabled)
        """
        batch, num_residues = hidden.shape[:2]

        # Apply local attention to capture neighborhood information
        hidden = hidden.transpose(0, 1)  # [num_residues, batch, hidden_dim]
        hidden_attn, _ = self.local_attention(hidden, hidden, hidden)
        hidden = self.layer_norm(hidden + hidden_attn)
        hidden = hidden.transpose(0, 1)  # Restore original shape

        # Predict chi angles
        logits = self.angle_predictor(hidden)
        logits = logits.view(batch, num_residues, self.num_angles_per_residue, self.num_bins)

        # Convert to angles
        bin_width = 2 * torch.pi / self.num_bins
        bin_centers = torch.arange(self.num_bins, device=hidden.device) * bin_width
        angles = F.softmax(logits, dim=-1) @ bin_centers

        # Get chi angle mask based on residue types
        chi_mask = self._get_residue_chi_mask(residue_type)

        # Convert angles to coordinates
        coordinates = self._convert_angles_to_coordinates(
            angles, backbone_coords, residue_type, chi_mask
        )

        # Predict uncertainties
        uncertainties = self.uncertainty_predictor(hidden)
        uncertainties = torch.sigmoid(uncertainties)  # Scale to [0, 1]

        # Predict rotamers if enabled
        rotamers = None
        if self.use_rotamers:
            rotamers = self.rotamer_predictor(hidden)

        # Apply mask if provided
        if mask is not None:
            angles = angles * mask.unsqueeze(-1)
            coordinates = coordinates * mask.unsqueeze(-1).unsqueeze(-1)
            uncertainties = uncertainties * mask.unsqueeze(-1)
            if rotamers is not None:
                rotamers = rotamers * mask.unsqueeze(-1)

        return {
            'angles': angles,
            'coordinates': coordinates,
            'uncertainties': uncertainties,
            'rotamers': rotamers,
            'chi_mask': chi_mask
        }
