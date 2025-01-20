from typing import Union, Optional, List, Dict, Tuple, cast
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import QuadMesh
from matplotlib.colorbar import Colorbar
import seaborn as sns

class StructureVisualizer:
    """
    Visualize protein structures and model predictions with type safety.
    """
    def __init__(self) -> None:
        """Initialize the visualizer."""
        self._fig: Optional[Figure] = None
        self._axes: Optional[Union[Axes, Axes3D]] = None

    @property
    def fig(self) -> Optional[Figure]:
        """Get the current figure."""
        return self._fig

    def create_figure(self, figsize: Tuple[float, float] = (10, 10)) -> Figure:
        """Create a new figure with the specified size."""
        self._fig = cast(Figure, plt.figure(figsize=figsize))
        return self._fig

    def plot_contact_map(
        self,
        coords: torch.Tensor,
        threshold: float = 8.0,
        mask: Optional[torch.Tensor] = None,
        title: str = "Contact Map",
        cmap: str = 'viridis'
    ) -> Figure:
        """
        Plot protein contact map based on distance threshold.

        Args:
            coords: [N, 3] tensor of atom coordinates
            threshold: Distance threshold for contacts
            mask: Optional mask for valid positions
            title: Plot title
            cmap: Colormap name
        """
        self._fig = self.create_figure()
        ax = cast(Axes, self._fig.add_subplot(111))

        # Compute pairwise distances
        distances = torch.cdist(coords, coords)
        contacts = (distances < threshold).float()

        if mask is not None:
            mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            contacts = contacts * mask_2d

        # Plot contact map
        im = cast(QuadMesh, ax.imshow(contacts.cpu().numpy(), cmap=cmap))
        if self._fig is not None:
            self._fig.colorbar(im, ax=ax)
        ax.set_title(title)

        return self._fig

    def plot_plddt_scores(
        self,
        plddt: torch.Tensor,
        title: str = "Per-residue Confidence (pLDDT)"
    ) -> Figure:
        """
        Plot per-residue pLDDT confidence scores.

        Args:
            plddt: [N] tensor of pLDDT scores
            title: Plot title
        """
        self._fig = self.create_figure(figsize=(12, 4))
        ax = cast(Axes, self._fig.add_subplot(111))

        scores = plddt.cpu().numpy()
        x = np.arange(len(scores))

        colors = ['red' if score < 70 else 'green' for score in scores]
        ax.bar(x, scores, color=colors, alpha=0.7)
        ax.axhline(y=70, color='r', linestyle='--', alpha=0.5)

        ax.set_ylim(0, 100)
        ax.set_xlabel('Residue')
        ax.set_ylabel('pLDDT Score')
        ax.set_title(title)

        return self._fig

    def plot_attention_matrix(
        self,
        attention: torch.Tensor,
        sequence: Optional[List[str]] = None,
        title: str = "Attention Matrix"
    ) -> Figure:
        """
        Plot attention matrix with optional sequence labels.

        Args:
            attention: [..., N, N] tensor of attention weights
            sequence: Optional list of sequence labels
            title: Plot title
        """
        if len(attention.shape) == 3:
            attention = attention.mean(0)

        self._fig = self.create_figure()
        ax = cast(Axes, self._fig.add_subplot(111))

        im = cast(QuadMesh, ax.imshow(attention.cpu().numpy(), cmap='viridis'))
        if self._fig is not None:
            self._fig.colorbar(im, ax=ax)

        if sequence is not None:
            ax.set_xticks(np.arange(len(sequence)))
            ax.set_yticks(np.arange(len(sequence)))
            ax.set_xticklabels(sequence, rotation=45, ha='right')
            ax.set_yticklabels(sequence)

        ax.set_title(title)
        plt.tight_layout()

        return self._fig

    def plot_backbone_trace(
        self,
        coords: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        title: str = "Protein Backbone Trace"
    ) -> Figure:
        """
        Plot 3D backbone trace with optional confidence coloring.

        Args:
            coords: [N, 3] tensor of coordinates
            confidence: Optional [N] tensor of confidence scores
            title: Plot title
        """
        self._fig = self.create_figure()
        ax = cast(Axes3D, self._fig.add_subplot(111, projection='3d'))

        coords_np = coords.cpu().numpy()
        x, y, z = coords_np.T

        if confidence is not None:
            confidence_np = confidence.cpu().numpy()
            scatter = ax.scatter(x, y, z, c=confidence_np, cmap='viridis',
                               s=50, alpha=0.7)
            if self._fig is not None:
                self._fig.colorbar(scatter, ax=ax, label='Confidence')
        else:
            ax.plot(x, y, z, '-', lw=2, alpha=0.7)

        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(title)

        return self._fig

    def plot_rmsd_by_residue(
        self,
        pred_coords: torch.Tensor,
        true_coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        title: str = "Per-residue RMSD"
    ) -> Figure:
        """
        Plot per-residue RMSD between predicted and true coordinates.

        Args:
            pred_coords: [N, 3] tensor of predicted coordinates
            true_coords: [N, 3] tensor of true coordinates
            mask: Optional [N] tensor mask for valid positions
            title: Plot title
        """
        self._fig = self.create_figure(figsize=(12, 4))
        ax = cast(Axes, self._fig.add_subplot(111))

        # Compute per-residue RMSD
        rmsd = torch.sqrt(torch.sum((pred_coords - true_coords) ** 2, dim=-1))
        if mask is not None:
            rmsd = rmsd * mask

        rmsd_np = rmsd.cpu().numpy()
        x = np.arange(len(rmsd_np))

        ax.bar(x, rmsd_np, alpha=0.7)
        ax.set_xlabel('Residue')
        ax.set_ylabel('RMSD (Å)')
        ax.set_title(title)

        return self._fig

    def save_figure(
        self,
        filename: str,
        dpi: int = 300,
        bbox_inches: str = 'tight'
    ) -> None:
        """
        Save the current figure to a file.

        Args:
            filename: Output filename
            dpi: Resolution in dots per inch
            bbox_inches: Bounding box specification
        """
        if self._fig is not None:
            self._fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        else:
            raise ValueError("No figure exists to save")

    def close_figure(self) -> None:
        """Close the current figure and clear references."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._axes = None
