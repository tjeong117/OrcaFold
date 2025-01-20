"""
Visualization utilities for protein structures and prediction outputs.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Union, Tuple
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
import numpy as np

class StructureVisualizer:
    """
    Visualize protein structures and model predictions.
    """

    @staticmethod
    def plot_contact_map(
        coords: torch.Tensor,
        threshold: float = 8.0,
        mask: Optional[torch.Tensor] = None,
        ax: Optional[Axes] = None,
        cmap: str = 'viridis'
    ) -> Figure:
        """
        Plot protein contact map based on distance threshold.

        Args:
            coords: [batch, num_residues, 3] coordinates
            threshold: Distance threshold for contacts (Angstroms)
            mask: Optional attention mask
            ax: Optional matplotlib axes
            cmap: Color map for plotting
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.figure

        # Compute pairwise distances
        distances = torch.cdist(coords, coords)
        contacts = (distances < threshold).float()

        if mask is not None:
            mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            contacts = contacts * mask_2d

        # Plot contact map
        if ax is not None:
            im = ax.imshow(contacts.cpu().numpy(), cmap=cmap)
            plt.colorbar(im, ax=ax)
            ax.set_title('Contact Map')

        return fig

    @staticmethod
    def plot_plddt_scores(
        plddt: torch.Tensor,
        ax: Optional[Axes] = None
    ) -> Figure:
        """
        Plot per-residue pLDDT confidence scores.

        Args:
            plddt: [batch, num_residues] pLDDT scores
            ax: Optional matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        else:
            fig = ax.figure

        scores = plddt.cpu().numpy()
        x = np.arange(len(scores))

        if ax is not None:
            ax.bar(x, scores, alpha=0.7)
            ax.set_ylim(0, 100)
            ax.set_xlabel('Residue')
            ax.set_ylabel('pLDDT Score')
            ax.set_title('Per-residue Confidence (pLDDT)')

        return fig

    @staticmethod
    def plot_attention_matrix(
        attention: torch.Tensor,
        sequence: Optional[List[str]] = None,
        ax: Optional[Axes] = None
    ) -> Figure:
        """
        Plot attention matrix with optional sequence labels.

        Args:
            attention: [num_heads, seq_len, seq_len] or [seq_len, seq_len] attention weights
            sequence: Optional amino acid sequence for labels
            ax: Optional matplotlib axes
        """
        if len(attention.shape) == 3:
            # Average over heads if multiple heads
            attention = attention.mean(0)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.figure

        if ax is not None:
            im = ax.imshow(attention.cpu().numpy(), cmap='viridis')
            plt.colorbar(im, ax=ax)

            if sequence is not None:
                ax.set_xticks(np.arange(len(sequence)))
                ax.set_yticks(np.arange(len(sequence)))
                ax.set_xticklabels(sequence, rotation=45, ha='right')
                ax.set_yticklabels(sequence)

            ax.set_title('Attention Matrix')
            plt.tight_layout()

        return fig

    @staticmethod
    def plot_backbone_trace(
        coords: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        ax: Optional[Axes] = None
    ) -> Figure:
        """
        Plot 3D backbone trace with optional confidence coloring.

        Args:
            coords: [num_residues, 3] CA coordinates
            confidence: Optional [num_residues] confidence scores
            ax: Optional matplotlib axes
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure

        if ax is not None:
            coords_np = coords.cpu().numpy()
            x, y, z = coords_np.T

            if confidence is not None:
                confidence_np = confidence.cpu().numpy()
                scatter = ax.scatter(x, y, z, c=confidence_np, cmap='viridis',
                                   s=50, alpha=0.7)
                plt.colorbar(scatter, ax=ax, label='Confidence')
            else:
                ax.plot(x, y, z, '-', lw=2, alpha=0.7)

            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            ax.set_zlabel('Z (Å)')
            ax.set_title('Protein Backbone Trace')

        return fig

    @staticmethod
    def plot_rmsd_by_residue(
        pred_coords: torch.Tensor,
        true_coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        ax: Optional[Axes] = None
    ) -> Figure:
        """
        Plot per-residue RMSD between predicted and true coordinates.

        Args:
            pred_coords: [num_residues, 3] predicted coordinates
            true_coords: [num_residues, 3] true coordinates
            mask: Optional [num_residues] mask for valid positions
            ax: Optional matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        else:
            fig = ax.figure

        # Compute per-residue RMSD
        rmsd = torch.sqrt(torch.sum((pred_coords - true_coords) ** 2, dim=-1))
        if mask is not None:
            rmsd = rmsd * mask

        if ax is not None:
            rmsd_np = rmsd.cpu().numpy()
            x = np.arange(len(rmsd_np))

            ax.bar(x, rmsd_np, alpha=0.7)
            ax.set_xlabel('Residue')
            ax.set_ylabel('RMSD (Å)')
            ax.set_title('Per-residue RMSD')

        return fig
