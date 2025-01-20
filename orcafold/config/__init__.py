"""
Configuration module for OrcaFold.
"""

from .model_config import (
    ModelConfig,
    EvoformerConfig,
    StructureConfig,
    ConfidenceConfig,
    DEFAULT_MONOMER_CONFIG,
    FAST_PRESET
)

from .train_config import (
    TrainingConfig,
    OptimizerConfig,
    DataConfig,
    LossConfig,
    DEFAULT_TRAINING_CONFIG,
    FAST_TRAINING_CONFIG
)

__all__ = [
    'ModelConfig',
    'EvoformerConfig',
    'StructureConfig',
    'ConfidenceConfig',
    'DEFAULT_MONOMER_CONFIG',
    'FAST_PRESET',
    'TrainingConfig',
    'OptimizerConfig',
    'DataConfig',
    'LossConfig',
    'DEFAULT_TRAINING_CONFIG',
    'FAST_TRAINING_CONFIG'
]
