#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Prediction script for OrcaFold.
"""

import torch
import argparse
import logging
from pathlib import Path
from typing import List, Optional
import json

from orcafold import OrcaFold
from orcafold.utils import StructureVisualizer
from orcafold.data import ProteinSequence

def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_sequences(input_path: str) -> List[ProteinSequence]:
    """Load protein sequences from file."""
    sequences = []
    with open(input_path, 'r') as f:
        current_seq = []
        current_id = None

        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(ProteinSequence(
                        ''.join(current_seq),
                        identifier=current_id
                    ))
                    current_seq = []
                current_id = line[1:].split()[0]
            elif line:
                current_seq.append(line)

        if current_seq:
            sequences.append(ProteinSequence(
                ''.join(current_seq),
                identifier=current_id
            ))

    return sequences

def predict_structure(
    model: OrcaFold,
    sequence: ProteinSequence,
    device: torch.device,
    visualizer: Optional[StructureVisualizer] = None,
    output_dir: Optional[Path] = None
) -> dict:
    """Predict structure for a single sequence."""
    # Prepare input
    seq_tensor = sequence.to_tensor().to(device)
    seq_mask = sequence.get_mask().to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(seq_tensor.unsqueeze(0), mask=seq_mask.unsqueeze(0))

    # Process outputs
    coords = outputs['coords'][0].cpu()
    confidence = outputs['confidence']
    plddt = confidence['plddt'][0].cpu()
    tm_score = confidence['tm_score'][0].cpu()

    # Visualize if requested
    if visualizer is not None and output_dir is not None:
        # Plot pLDDT scores
        fig = visualizer.plot_plddt_scores(plddt)
        fig.savefig(output_dir / f'{sequence.identifier}_plddt.png')

        # Plot contact map
        fig = visualizer.plot_contact_map(coords)
        fig.savefig(output_dir / f'{sequence.identifier}_contacts.png')

        # Plot 3D structure
        fig = visualizer.plot_backbone_trace(coords, confidence=plddt)
        fig.savefig(output_dir / f'{sequence.identifier}_structure.png')

    # Return prediction results
    return {
        'coords': coords.numpy().tolist(),
        'plddt': plddt.numpy().tolist(),
        'tm_score': tm_score.item(),
        'sequence_length': len(sequence)
    }

def main():
    parser = argparse.ArgumentParser(description='Predict protein structures using OrcaFold')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input FASTA file')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    args = parser.parse_args()

    # Setup
    setup_logging()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logging.info(f"Loading model from {args.model}")
    model = OrcaFold.load_model(args.model, device=device)
    model.eval()

    # Initialize visualizer if needed
    visualizer = StructureVisualizer() if args.visualize else None

    # Load sequences
    logging.info(f"Loading sequences from {args.input}")
    sequences = load_sequences(args.input)
    logging.info(f"Found {len(sequences)} sequences")

    # Process each sequence
    results = {}
    for seq in sequences:
        logging.info(f"Processing sequence {seq.identifier}")
        try:
            results[seq.identifier] = predict_structure(
                model,
                seq,
                device,
                visualizer,
                output_dir if args.visualize else None
            )
            logging.info(f"Successfully predicted structure for {seq.identifier}")
        except Exception as e:
            logging.error(f"Failed to process {seq.identifier}: {str(e)}")
            continue

        # Save PDB file
        pdb_path = output_dir / f'{seq.identifier}.pdb'
        seq.save_pdb(
            pdb_path,
            coords=results[seq.identifier]['coords'],
            bfactors=results[seq.identifier]['plddt']
        )

    # Save results summary
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(f"Saved results to {results_path}")

if __name__ == '__main__':
    main()
