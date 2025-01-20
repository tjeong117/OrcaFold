"""
OrcaFold: Advanced Protein Structure Prediction System
====================================================

A powerful protein structure prediction system implementing state-of-the-art
deep learning architectures for molecular structure analysis.
"""

import torch
import warnings
from typing import Optional, Dict, Union, Tuple

# Version information
__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Main class imports
from orcafold.modules.evoformer import EvoformerStack
from orcafold.modules.structure import StructureModule
from orcafold.modules.confidence import ConfidencePredictor
from orcafold.data import Pipeline
from orcafold.config import OrcaFoldConfig

class OrcaFold:
    """
    Main OrcaFold protein structure prediction system.

    This class serves as the main interface to the OrcaFold system,
    coordinating the Evoformer, Structure Module, and confidence prediction
    components.
    """

    def __init__(
        self,
        config: Optional[Union[Dict, OrcaFoldConfig]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize OrcaFold model.

        Args:
            config: Model configuration, either as a dictionary or OrcaFoldConfig object.
                   If None, default configuration will be used.
            device: Device to run the model on ('cuda' or 'cpu'). If None, will
                   automatically detect GPU availability.
        """
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # Load configuration
        if config is None:
            self.config = OrcaFoldConfig()
        elif isinstance(config, dict):
            self.config = OrcaFoldConfig(**config)
        else:
            self.config = config

        # Initialize components
        self.evoformer = EvoformerStack(self.config.evoformer)
        self.structure_module = StructureModule(self.config.structure)
        self.confidence_predictor = ConfidencePredictor()

        # Move to device
        self.to(device)

        # Pipeline for data processing
        self.pipeline = Pipeline(self.config.data)

    def to(self, device: str) -> 'OrcaFold':
        """Move model to specified device."""
        self.device = device
        self.evoformer.to(device)
        self.structure_module.to(device)
        self.confidence_predictor.to(device)
        return self

    def predict(
        self,
        sequence: str,
        *,
        use_templates: bool = True,
        num_recycles: Optional[int] = None
    ) -> Tuple[Dict, Dict]:
        """
        Predict protein structure from amino acid sequence.

        Args:
            sequence: Amino acid sequence as string
            use_templates: Whether to use template information
            num_recycles: Number of prediction recycling iterations.
                         If None, uses value from config.

        Returns:
            Tuple containing:
            - Dictionary with structure prediction results
            - Dictionary with confidence metrics
        """
        # Process input data
        features = self.pipeline.process_sequence(
            sequence,
            use_templates=use_templates
        )

        # Set number of recycling iterations
        if num_recycles is None:
            num_recycles = self.config.structure.recycling_steps

        # Initial representations
        msa_repr = None
        pair_repr = None

        # Recycling loop
        for recycle_step in range(num_recycles):
            # Evoformer processing
            msa_repr, pair_repr = self.evoformer(
                features,
                msa_repr=msa_repr,
                pair_repr=pair_repr
            )

            # Structure module
            structure_output = self.structure_module(
                msa_repr,
                pair_repr,
                features
            )

            # Update representations for next recycling iteration
            if recycle_step < num_recycles - 1:
                msa_repr = structure_output['msa_repr']
                pair_repr = structure_output['pair_repr']

        # Predict confidence scores
        confidence_metrics = self.confidence_predictor(
            structure_output,
            features
        )

        return structure_output, confidence_metrics

    @classmethod
    def load_model(cls, path: str, device: Optional[str] = None) -> 'OrcaFold':
        """
        Load a pretrained OrcaFold model.

        Args:
            path: Path to model checkpoint
            device: Device to load model to

        Returns:
            Loaded OrcaFold model
        """
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']

        model = cls(config=config, device=device)
        model.load_state_dict(checkpoint['model_state'])

        return model

    def save_model(self, path: str):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint to
        """
        checkpoint = {
            'model_state': self.state_dict(),
            'config': self.config,
            'version': __version__
        }
        torch.save(checkpoint, path)

# Configure warnings
warnings.filterwarnings('once', category=UserWarning)

# Convenience function for quick predictions
def predict_structure(
    sequence: str,
    device: Optional[str] = None,
    **kwargs
) -> Tuple[Dict, Dict]:
    """
    Quick prediction function using default settings.

    Args:
        sequence: Amino acid sequence
        device: Computation device
        **kwargs: Additional arguments passed to OrcaFold.predict()

    Returns:
        Structure prediction and confidence metrics
    """
    model = OrcaFold(device=device)
    return model.predict(sequence, **kwargs)
