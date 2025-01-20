"""
Training configuration for OrcaFold.
Defines training hyperparameters and optimization settings.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Union
import torch.optim as optim

@dataclass
class OptimizerConfig:
    """Configuration for the optimizer."""
    name: str = "adam"
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.0

    # Learning rate scheduling
    warmup_steps: int = 1000
    decay_rate: float = 0.95
    decay_steps: int = 50000
    min_lr: float = 1e-5

    def get_optimizer(self, parameters) -> optim.Optimizer:
        """Create optimizer instance."""
        if self.name.lower() == "adam":
            return optim.Adam(
                parameters,
                lr=self.learning_rate,
                betas=(self.beta1, self.beta2),
                eps=self.epsilon,
                weight_decay=self.weight_decay
            )
        elif self.name.lower() == "adamw":
            return optim.AdamW(
                parameters,
                lr=self.learning_rate,
                betas=(self.beta1, self.beta2),
                eps=self.epsilon,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.name}")

@dataclass
class DataConfig:
    """Configuration for training data."""
    train_data_dir: str = "data/train"
    val_data_dir: str = "data/val"
    max_template_date: str = "2024-01-01"
    max_templates: int = 4

    # Sequence processing
    min_sequence_length: int = 16
    max_sequence_length: int = 1024

    # MSA processing
    max_msa_clusters: int = 512
    max_extra_msa: int = 1024

    # Template processing
    max_template_hits: int = 20

    # Data augmentation
    random_crop: bool = True
    crop_size: int = 256
    random_rotation: bool = True
    random_translation: float = 0.2  # Å

@dataclass
class LossConfig:
    """Configuration for loss functions."""
    # Main loss components
    fape_loss_weight: float = 1.0
    auxiliary_loss_weight: float = 0.1

    # FAPE (Frame Aligned Point Error) settings
    fape_clamp_distance: float = 10.0
    fape_loss_unit_distance: float = 10.0

    # Auxiliary losses
    predicted_lddt_weight: float = 0.01
    predicted_aligned_error_weight: float = 0.01

    # Violation losses
    violation_tolerance_factor: float = 12.0
    clash_overlap_tolerance: float = 1.5
    bond_angle_tolerance: float = 12.0

@dataclass
class TrainingConfig:
    """Main training configuration."""
    # Basic training settings
    batch_size: int = 1
    num_epochs: int = 100
    gradient_clip_norm: float = 0.1
    accumulation_steps: int = 1

    # Optimizer and data settings
    optimizer: OptimizerConfig = OptimizerConfig()
    data: DataConfig = DataConfig()
    loss: LossConfig = LossConfig()

    # Training features
    use_amp: bool = True  # Automatic Mixed Precision
    num_workers: int = 4
    prefetch_factor: int = 2

    # Checkpointing
    save_frequency: int = 1000
    checkpoint_dir: str = "checkpoints"
    keep_n_checkpoints: int = 5

    # Logging
    log_frequency: int = 100
    validation_frequency: int = 1000

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'gradient_clip_norm': self.gradient_clip_norm,
            'accumulation_steps': self.accumulation_steps,
            'optimizer': self.optimizer.__dict__,
            'data': self.data.__dict__,
            'loss': self.loss.__dict__,
            'use_amp': self.use_amp,
            'num_workers': self.num_workers,
            'prefetch_factor': self.prefetch_factor,
            'save_frequency': self.save_frequency,
            'checkpoint_dir': self.checkpoint_dir,
            'keep_n_checkpoints': self.keep_n_checkpoints,
            'log_frequency': self.log_frequency,
            'validation_frequency': self.validation_frequency
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'TrainingConfig':
        """Create config from dictionary."""
        optimizer_config = OptimizerConfig(**config_dict.get('optimizer', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        loss_config = LossConfig(**config_dict.get('loss', {}))

        return cls(
            batch_size=config_dict.get('batch_size', 1),
            num_epochs=config_dict.get('num_epochs', 100),
            gradient_clip_norm=config_dict.get('gradient_clip_norm', 0.1),
            accumulation_steps=config_dict.get('accumulation_steps', 1),
            optimizer=optimizer_config,
            data=data_config,
            loss=loss_config,
            use_amp=config_dict.get('use_amp', True),
            num_workers=config_dict.get('num_workers', 4),
            prefetch_factor=config_dict.get('prefetch_factor', 2),
            save_frequency=config_dict.get('save_frequency', 1000),
            checkpoint_dir=config_dict.get('checkpoint_dir', 'checkpoints'),
            keep_n_checkpoints=config_dict.get('keep_n_checkpoints', 5),
            log_frequency=config_dict.get('log_frequency', 100),
            validation_frequency=config_dict.get('validation_frequency', 1000)
        )

# Preset configurations
DEFAULT_TRAINING_CONFIG = TrainingConfig()

FAST_TRAINING_CONFIG = TrainingConfig(
    batch_size=4,
    num_epochs=50,
    optimizer=OptimizerConfig(
        learning_rate=2e-4,
        warmup_steps=500
    ),
    data=DataConfig(
        max_sequence_length=512,
        max_msa_clusters=256
    )
)
