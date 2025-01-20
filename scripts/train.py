#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Training script for OrcaFold.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Optional

from orcafold.config import ModelConfig, TrainingConfig
from orcafold import OrcaFold
from orcafold.data import ProteinDataset

def setup_logging(save_dir: Path) -> None:
    """Setup logging configuration."""
    log_file = save_dir / 'train.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_epoch(
    model: OrcaFold,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    accumulation_steps: int = 1
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    optimizer.zero_grad()

    for i, batch in enumerate(dataloader):
        # Move data to device
        sequences = batch['sequence'].to(device)
        coords = batch['coordinates'].to(device)
        mask = batch['mask'].to(device)

        # Forward pass
        outputs = model(sequences, mask=mask)

        # Compute loss
        loss = outputs['loss'] / accumulation_steps

        # Backward pass
        loss.backward()

        # Update weights if needed
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

        if (i + 1) % 10 == 0:
            logging.info(f'Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}')

    return total_loss / num_batches

def validate(
    model: OrcaFold,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_metrics = {
        'loss': 0.0,
        'plddt': 0.0,
        'tm_score': 0.0
    }

    with torch.no_grad():
        for batch in dataloader:
            sequences = batch['sequence'].to(device)
            coords = batch['coordinates'].to(device)
            mask = batch['mask'].to(device)

            outputs = model(sequences, mask=mask)

            total_metrics['loss'] += outputs['loss'].item()
            total_metrics['plddt'] += outputs['confidence']['plddt'].mean().item()
            total_metrics['tm_score'] += outputs['confidence']['tm_score'].mean().item()

    # Average metrics
    num_batches = len(dataloader)
    return {k: v / num_batches for k, v in total_metrics.items()}

def main():
    parser = argparse.ArgumentParser(description='Train OrcaFold')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data-dir', type=str, required=True, help='Data directory')
    parser.add_argument('--save-dir', type=str, required=True, help='Save directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(save_dir)

    # Load configuration
    config = load_config(args.config)
    model_config = ModelConfig(**config['model'])
    train_config = TrainingConfig(**config['training'])

    # Initialize model
    device = torch.device(args.device)
    model = OrcaFold(config=model_config)
    model = model.to(device)

    # Setup data
    train_dataset = ProteinDataset(
        data_dir=f"{args.data_dir}/train",
        max_seq_len=train_config.data.max_sequence_length
    )
    val_dataset = ProteinDataset(
        data_dir=f"{args.data_dir}/val",
        max_seq_len=train_config.data.max_sequence_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=True
    )

    # Setup optimizer
    optimizer = train_config.optimizer.get_optimizer(model.parameters())

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(train_config.num_epochs):
        logging.info(f"Epoch {epoch+1}/{train_config.num_epochs}")

        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train_config.accumulation_steps
        )
        logging.info(f"Training Loss: {train_loss:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, device)
        logging.info("Validation Metrics:")
        for metric, value in val_metrics.items():
            logging.info(f"{metric}: {value:.4f}")

        # Save checkpoint if best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint_path = save_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config
            }, checkpoint_path)
            logging.info(f"Saved best model to {checkpoint_path}")

        # Regular checkpoint
        if (epoch + 1) % train_config.save_frequency == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config
            }, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")

if __name__ == '__main__':
    main()
