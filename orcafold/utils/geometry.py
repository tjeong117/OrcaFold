"""
Geometric transformation utilities for protein structure prediction.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import math

def get_rotation_matrix(angles: torch.Tensor) -> torch.Tensor:
    """
    Generate rotation matrices from angles.

    Args:
        angles: [..., 3] tensor of rotation angles (in radians)

    Returns:
        [..., 3, 3] tensor of rotation matrices
    """
    # Extract individual angles
    theta_x, theta_y, theta_z = torch.unbind(angles, dim=-1)

    # Compute sin and cos
    cos_x, sin_x = torch.cos(theta_x), torch.sin(theta_x)
    cos_y, sin_y = torch.cos(theta_y), torch.sin(theta_y)
    cos_z, sin_z = torch.cos(theta_z), torch.sin(theta_z)

    # Rotation matrices for each axis
    Rx = torch.stack([
        torch.stack([torch.ones_like(cos_x), torch.zeros_like(cos_x), torch.zeros_like(cos_x)], dim=-1),
        torch.stack([torch.zeros_like(cos_x), cos_x, -sin_x], dim=-1),
        torch.stack([torch.zeros_like(cos_x), sin_x, cos_x], dim=-1)
    ], dim=-2)

    Ry = torch.stack([
        torch.stack([cos_y, torch.zeros_like(cos_y), sin_y], dim=-1),
        torch.stack([torch.zeros_like(cos_y), torch.ones_like(cos_y), torch.zeros_like(cos_y)], dim=-1),
        torch.stack([-sin_y, torch.zeros_like(cos_y), cos_y], dim=-1)
    ], dim=-2)

    Rz = torch.stack([
        torch.stack([cos_z, -sin_z, torch.zeros_like(cos_z)], dim=-1),
        torch.stack([sin_z, cos_z, torch.zeros_like(cos_z)], dim=-1),
        torch.stack([torch.zeros_like(cos_z), torch.zeros_like(cos_z), torch.ones_like(cos_z)], dim=-1)
    ], dim=-2)

    # Combined rotation matrix
    return torch.matmul(torch.matmul(Rz, Ry), Rx)

def compute_backbone_frames(
    coords: torch.Tensor,
    eps: float = 1e-7
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute backbone reference frames from atom coordinates.

    Args:
        coords: [..., num_residues, 3, 3] backbone atom coordinates (N, CA, C)
        eps: Small value for numerical stability

    Returns:
        Tuple of:
        - [..., num_residues, 3] rotation matrices
        - [..., num_residues, 3] translation vectors
    """
    # Extract individual atom coordinates
    N, CA, C = torch.unbind(coords, dim=-2)

    # Compute frame vectors
    t = CA  # Origin at CA
    n = F.normalize(N - CA, dim=-1)  # N-CA defines first axis
    c = F.normalize(C - CA, dim=-1)  # C-CA helps define plane

    # Compute orthonormal basis
    z = F.normalize(torch.cross(n, c, dim=-1), dim=-1)
    y = F.normalize(torch.cross(z, n, dim=-1), dim=-1)
    x = F.normalize(torch.cross(y, z, dim=-1), dim=-1)

    # Stack into rotation matrix
    R = torch.stack([x, y, z], dim=-1)

    return R, t

def compute_dihedral_angles(
    coords: torch.Tensor,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Compute backbone dihedral angles (phi, psi, omega) from coordinates.

    Args:
        coords: [..., num_residues, 3, 3] backbone atom coordinates (N, CA, C)
        eps: Small value for numerical stability

    Returns:
        [..., num_residues, 3] dihedral angles in radians
    """
    # Extract atom positions
    N, CA, C = torch.unbind(coords, dim=-2)

    # Compute displacement vectors
    v1 = CA - N
    v2 = C - CA
    v3 = torch.roll(N, shifts=-1, dims=-2) - C

    # Normalize vectors
    v1 = F.normalize(v1, dim=-1)
    v2 = F.normalize(v2, dim=-1)
    v3 = F.normalize(v3, dim=-1)

    # Compute cross products
    n1 = F.normalize(torch.cross(v1, v2, dim=-1), dim=-1)
    n2 = F.normalize(torch.cross(v2, v3, dim=-1), dim=-1)

    # Compute angle using dot product
    cos_angle = torch.sum(n1 * n2, dim=-1)
    cos_angle = torch.clamp(cos_angle, -1 + eps, 1 - eps)

    # Determine sign using cross product
    cross_prod = torch.cross(n1, n2, dim=-1)
    sign = torch.sign(torch.sum(cross_prod * v2, dim=-1))

    # Compute final angles
    angles = sign * torch.acos(cos_angle)

    return angles

def compute_rmsd(
    coords1: torch.Tensor,
    coords2: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute Root Mean Square Deviation (RMSD) between two sets of coordinates.

    Args:
        coords1: [..., num_points, 3] first set of coordinates
        coords2: [..., num_points, 3] second set of coordinates
        mask: Optional [..., num_points] boolean mask for valid points

    Returns:
        [...] RMSD values
    """
    # Compute squared differences
    squared_diff = torch.sum((coords1 - coords2) ** 2, dim=-1)

    if mask is not None:
        squared_diff = squared_diff * mask
        # Compute mean over valid points only
        rmsd = torch.sqrt(torch.sum(squared_diff, dim=-1) / (torch.sum(mask, dim=-1) + 1e-7))
    else:
        rmsd = torch.sqrt(torch.mean(squared_diff, dim=-1))

    return rmsd

def kabsch_algorithm(
    coords1: torch.Tensor,
    coords2: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute optimal rotation and translation to align two sets of coordinates.

    Args:
        coords1: [..., num_points, 3] first set of coordinates
        coords2: [..., num_points, 3] second set of coordinates
        mask: Optional [..., num_points] boolean mask for valid points

    Returns:
        Tuple of:
        - [..., 3, 3] optimal rotation matrix
        - [..., 3] optimal translation vector
    """
    # Center coordinates
    if mask is not None:
        mask = mask[..., None]  # Add dimension for broadcasting
        mean1 = torch.sum(coords1 * mask, dim=-2) / (torch.sum(mask, dim=-2) + 1e-7)
        mean2 = torch.sum(coords2 * mask, dim=-2) / (torch.sum(mask, dim=-2) + 1e-7)
    else:
        mean1 = torch.mean(coords1, dim=-2)
        mean2 = torch.mean(coords2, dim=-2)

    coords1_centered = coords1 - mean1[..., None, :]
    coords2_centered = coords2 - mean2[..., None, :]

    # Compute covariance matrix
    if mask is not None:
        covariance = torch.matmul(
            (coords1_centered * mask).transpose(-2, -1),
            coords2_centered * mask
        )
    else:
        covariance = torch.matmul(
            coords1_centered.transpose(-2, -1),
            coords2_centered
        )

    # Compute optimal rotation using SVD
    U, _, V = torch.svd(covariance)
    rotation = torch.matmul(V, U.transpose(-2, -1))

    # Ensure right-handed coordinate system
    det = torch.det(rotation)
    V_adjusted = V.clone()
    V_adjusted[..., :, -1] *= torch.sign(det)[..., None]
    rotation = torch.matmul(V_adjusted, U.transpose(-2, -1))

    # Compute translation
    translation = mean2 - torch.matmul(mean1[..., None, :], rotation.transpose(-2, -1)).squeeze(-2)

    return rotation, translation
