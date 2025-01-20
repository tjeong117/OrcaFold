"""
Model configuration for OrcaFold.
Defines the architecture and model hyperparameters.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Union

@dataclass
class EvoformerConfig:
    """Configuration for the Evoformer stack."""
    num_layers: int = 48
    msa_dim: int = 256
    pair_dim: int = 128
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 1024
    max_msa_clusters: int = 512

    # MSA attention settings
    msa_row_attention_with_pair_bias: bool = True
    msa_column_attention: bool = True
    msa_transition: bool = True

    # Pair attention settings
    outer_product_mean: bool = True
    triangle_attention_starting_node: bool = True
    triangle_attention_ending_node: bool = True
    triangle_multiplication_outgoing: bool = True
    triangle_multiplication_incoming: bool = True
    pair_transition: bool = True

    # Layer sizes
    ff_dim: int = 1024
    pair_transition_n: int = 2
    msa_transition_n: int = 1

@dataclass
class StructureConfig:
    """Configuration for the Structure module."""
    num_layers: int = 8
    hidden_dim: int = 128
    num_channels: int = 384
    num_residual_blocks: int = 4
    num_angles: int = 4  # phi, psi, omega, chi
    num_bins: int = 50   # For angle predictions

    # IPA (Invariant Point Attention) settings
    num_points: int = 8
    num_scalar_qk: int = 16
    num_scalar_v: int = 16
    num_point_qk: int = 4
    num_point_v: int = 8

    # Structure refinement
    num_refinement_iterations: int = 1
    refinement_size: int = 128

    # Recycling
    recycle_features: bool = True
    recycle_positions: bool = True
    num_recycling_iters: int = 3

@dataclass
class ConfidenceConfig:
    """Configuration for confidence prediction."""
    enabled: bool = True
    num_channels: int = 128
    num_layers: int = 4

    # Metrics to compute
    compute_plddt: bool = True
    compute_ptm: bool = True
    compute_aligned_error: bool = True

    # Auxiliary losses
    predicted_lddt_weight: float = 0.1
    predicted_aligned_error_weight: float = 0.1

@dataclass
class ModelConfig:
    """Main model configuration."""
    evoformer: EvoformerConfig = EvoformerConfig()
    structure: StructureConfig = StructureConfig()
    confidence: ConfidenceConfig = ConfidenceConfig()

    # Global model settings
    max_templates: int = 4
    template_embedding_dim: int = 64
    extra_msa_dim: int = 64
    extra_sequences: bool = True

    # Runtime settings
    chunk_size: Optional[int] = None

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'evoformer': self.evoformer.__dict__,
            'structure': self.structure.__dict__,
            'confidence': self.confidence.__dict__,
            'max_templates': self.max_templates,
            'template_embedding_dim': self.template_embedding_dim,
            'extra_msa_dim': self.extra_msa_dim,
            'extra_sequences': self.extra_sequences,
            'chunk_size': self.chunk_size
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ModelConfig':
        """Create config from dictionary."""
        evoformer_config = EvoformerConfig(**config_dict.get('evoformer', {}))
        structure_config = StructureConfig(**config_dict.get('structure', {}))
        confidence_config = ConfidenceConfig(**config_dict.get('confidence', {}))

        return cls(
            evoformer=evoformer_config,
            structure=structure_config,
            confidence=confidence_config,
            max_templates=config_dict.get('max_templates', 4),
            template_embedding_dim=config_dict.get('template_embedding_dim', 64),
            extra_msa_dim=config_dict.get('extra_msa_dim', 64),
            extra_sequences=config_dict.get('extra_sequences', True),
            chunk_size=config_dict.get('chunk_size', None)
        )

# Preset configurations
DEFAULT_MONOMER_CONFIG = ModelConfig(
    evoformer=EvoformerConfig(
        num_layers=48,
        max_seq_len=1024
    ),
    structure=StructureConfig(
        num_recycling_iters=3
    )
)

FAST_PRESET = ModelConfig(
    evoformer=EvoformerConfig(
        num_layers=24,
        max_seq_len=512
    ),
    structure=StructureConfig(
        num_recycling_iters=1
    )
)
