#!/usr/bin/env python3
"""
Multiple Sequence Alignment (MSA) Tools Module

This module provides utilities for generating and processing multiple sequence alignments,
which are crucial for protein structure prediction and evolutionary analysis.
"""

import os
import subprocess
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

class MSATools:
    """
    A comprehensive utility class for multiple sequence alignment operations.

    Supports various MSA generation tools and alignment processing methods.
    """

    def __init__(self,
                 tool_paths: Optional[Dict[str, str]] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize MSA Tools with configurable tool paths and output directory.

        Args:
            tool_paths (Dict[str, str], optional): Dictionary of paths to MSA tools
            output_dir (str, optional): Directory to store alignment outputs
        """
        # Default tool paths (can be overridden)
        self.tool_paths = {
            'hmmer': '/usr/bin/hmmer',
            'clustalw': '/usr/bin/clustalw',
            'mafft': '/usr/bin/mafft',
            'muscle': '/usr/bin/muscle',
            'tcoffee': '/usr/bin/t_coffee'
        }

        # Update with provided paths
        if tool_paths:
            self.tool_paths.update(tool_paths)

        # Setup output directory
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'msa_outputs')
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def generate_msa(self,
                     input_sequences: List[str],
                     method: str = 'mafft',
                     output_format: str = 'fasta') -> Path:
        """
        Generate multiple sequence alignment using specified method.

        Args:
            input_sequences (List[str]): List of protein sequences to align
            method (str, optional): MSA tool to use. Defaults to 'mafft'
            output_format (str, optional): Output alignment format. Defaults to 'fasta'

        Returns:
            Path to generated alignment file
        """
        # Validate method
        supported_methods = ['mafft', 'muscle', 'clustalw', 'hmmer', 'tcoffee']
        if method.lower() not in supported_methods:
            raise ValueError(f"Unsupported alignment method. Choose from {supported_methods}")

        # Prepare input file
        input_path = os.path.join(self.output_dir, 'input_sequences.fasta')
        with open(input_path, 'w') as f:
            for i, seq in enumerate(input_sequences, 1):
                f.write(f">Sequence_{i}\n{seq}\n")

        # Prepare output path
        output_path = os.path.join(
            self.output_dir,
            f'alignment_{method.lower()}.{output_format}'
        )

        # Run alignment based on method
        try:
            if method.lower() == 'mafft':
                cmd = [
                    self.tool_paths['mafft'],
                    '--auto',
                    input_path
                ]
                with open(output_path, 'w') as outfile:
                    subprocess.run(cmd, stdout=outfile, check=True)

            elif method.lower() == 'muscle':
                cmd = [
                    self.tool_paths['muscle'],
                    '-in', input_path,
                    '-out', output_path
                ]
                subprocess.run(cmd, check=True)

            # Add more alignment method implementations as needed

            self.logger.info(f"MSA generated using {method}: {output_path}")
            return Path(output_path)

        except subprocess.CalledProcessError as e:
            self.logger.error(f"MSA generation failed: {e}")
            raise

    def analyze_msa(self,
                    msa_file: Path) -> Dict[str, Any]:
        """
        Analyze multiple sequence alignment for various properties.

        Args:
            msa_file (Path): Path to the multiple sequence alignment file

        Returns:
            Dictionary of MSA analysis metrics
        """
        # Placeholder for comprehensive MSA analysis
        analysis_results = {
            'num_sequences': 0,
            'alignment_length': 0,
            'conservation_score': 0.0,
            'gap_percentage': 0.0
        }

        try:
            # Read alignment file
            with open(msa_file, 'r') as f:
                sequences = [line.strip() for line in f if not line.startswith('>')]

            # Basic analysis
            analysis_results['num_sequences'] = len(sequences)
            analysis_results['alignment_length'] = len(sequences[0]) if sequences else 0

            # Calculate gap percentage
            total_gaps = sum(seq.count('-') for seq in sequences)
            total_chars = sum(len(seq) for seq in sequences)
            analysis_results['gap_percentage'] = (total_gaps / total_chars) * 100

            # Placeholder for more advanced conservation analysis
            self.logger.info("MSA analysis completed")
            return analysis_results

        except Exception as e:
            self.logger.error(f"MSA analysis failed: {e}")
            raise

    def convert_alignment_format(self,
                                 input_file: Path,
                                 output_format: str = 'stockholm') -> Path:
        """
        Convert alignment between different file formats.

        Args:
            input_file (Path): Input alignment file
            output_format (str, optional): Desired output format

        Returns:
            Path to converted alignment file
        """
        supported_formats = ['fasta', 'stockholm', 'clustal', 'phylip']
        if output_format.lower() not in supported_formats:
            raise ValueError(f"Unsupported format. Choose from {supported_formats}")

        output_file = os.path.join(
            self.output_dir,
            f'converted_alignment.{output_format.lower()}'
        )

        try:
            # Use Bio.AlignIO for format conversion
            from Bio import AlignIO

            # Read input alignment
            alignment = AlignIO.read(input_file, 'fasta')

            # Write to new format
            AlignIO.write(alignment, output_file, output_format.lower())

            self.logger.info(f"Alignment converted to {output_format}")
            return Path(output_file)

        except ImportError:
            self.logger.error("Biopython (Bio.AlignIO) is required for format conversion")
            raise
        except Exception as e:
            self.logger.error(f"Alignment conversion failed: {e}")
            raise

def main():
    """
    Example usage of MSA Tools
    """
    # Example sequences
    sequences = [
        "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETMKELKAYKSELEEQLTPVAEETRARLSKELQAAQARLGADVLASHGRLVQYRGEVQAMLGQSTEELRVRLASHLRKLRKRLLRDADDLQKRLAVYQAGAREGAERGLSAIRERLGPLVEQGRVRAATVGSLAGQPLQERAQAWGERLRARMEEMGSRTRDRLDEVKEQVAEVRAKLEEQAQQIRLVLASHQARLKSWFEPLVEDMQRQWAGLVEKVQAAVGTSAAPVPSDNH",
        "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETMKELKAYKSELEEQLTPVAEETRARLSKELQAAQARLGADVLASHGRLVQYRGEVQAMLGQSTEELRVRLASHLRKLRKRLLRDADDLQKRLAVYQAGAREGAERGLSAIRERLGPLVEQGRVRAATVGSLAGQPLQERAQAWGERLRARMEEMGSRTRDRLDEVKEQVAEVRAKLEEQAQQIRLVLASHQARLKSWFEPLVEDMQRQWAGLVEKVQAAVGTSAAPVPSDNH",
        "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETMKELKAYKSELEEQLTPVAEETRARLSKELQAAQARLGADVLASHGRLVQYRGEVQAMLGQSTEELRVRLASHLRKLRKRLLRDADDLQKRLAVYQAGAREGAERGLSAIRERLGPLVEQGRVRAATVGSLAGQPLQERAQAWGERLRARMEEMGSRTRDRLDEVKEQVAEVRAKLEEQAQQIRLVLASHQARLKSWFEPLVEDMQRQWAGLVEKVQAAVGTSAAPVPSDNH"
    ]

    # Initialize MSA Tools
    msa_tools = MSATools(output_dir='example_msa_output')

    try:
        # Generate MSA
        msa_file = msa_tools.generate_msa(
            input_sequences=sequences,
            method='mafft'
        )

        # Analyze MSA
        analysis_results = msa_tools.analyze_msa(msa_file)
        print("MSA Analysis Results:", analysis_results)

        # Convert alignment format
        converted_file = msa_tools.convert_alignment_format(
            msa_file,
            output_format='stockholm'
        )
        print(f"Converted alignment saved to: {converted_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
