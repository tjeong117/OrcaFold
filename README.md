# OrcaFold

A state-of-the-art protein structure prediction system implementing advanced deep learning architectures inspired by AlphaFold.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![CUDA Support](https://img.shields.io/badge/CUDA-11.3%2B-green)
![License](https://img.shields.io/badge/license-MIT-blue)

## Core Components

### Evoformer Module
- Advanced MSA (Multiple Sequence Alignment) processing
- Pair representation handling
- Multi-head attention mechanisms
- Iterative refinement pipeline

### Structure Module
- 3D coordinate prediction
- Backbone and side chain refinement
- Atomic structure assembly
- Iterative structure optimization

### Supporting Systems
- High-performance data pipeline
- Configuration management
- Confidence scoring (pLDDT & TM-score)

## Project Structure

```
OrcaFold/
├── orcafold/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── model_config.py
│   │   └── train_config.py
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── evoformer/
│   │   │   ├── __init__.py
│   │   │   ├── msa_processor.py
│   │   │   ├── pair_processor.py
│   │   │   └── attention.py
│   │   ├── structure/
│   │   │   ├── __init__.py
│   │   │   ├── backbone.py
│   │   │   ├── sidechain.py
│   │   │   └── recycling.py
│   │   └── confidence/
│   │       ├── __init__.py
│   │       ├── plddt.py
│   │       └── tm_score.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── msa_tools.py
│   │   └── templates.py
│   └── utils/
│       ├── __init__.py
│       ├── geometry.py
│       ├── cuda_utils.py
│       └── visualization.py
├── scripts/
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── tests/
│   ├── __init__.py
│   ├── test_evoformer.py
│   ├── test_structure.py
│   └── test_pipeline.py
├── examples/
│   ├── single_prediction.py
│   ├── batch_processing.py
│   └── visualization.py
├── docs/
│   ├── architecture.md
│   ├── installation.md
│   ├── pipeline.md
│   └── api.md
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   └── gpu.txt
├── setup.py
├── README.md
└── LICENSE
```

## Module Details

### 1. Evoformer (`orcafold/modules/evoformer/`)
- **MSA Processor**: Handles multiple sequence alignment processing
- **Pair Processor**: Manages residue pair representations
- **Attention Mechanisms**: Implements various attention patterns
  - Row-wise attention
  - Column-wise attention
  - Triangle multiplication updates

### 2. Structure Module (`orcafold/modules/structure/`)
- **Backbone Generator**: Creates initial backbone trace
- **Side Chain Placement**: Predicts side chain conformations
- **Structure Refinement**: Iteratively improves predictions
- **Recycling Handler**: Manages prediction recycling

### 3. Data Pipeline (`orcafold/data/`)
- **MSA Generation**: Interfaces with JackHMMER/HHblits
- **Template Search**: Finds and processes template structures
- **Feature Processing**: Prepares features for model input

### 4. Configuration (`orcafold/config/`)
- Model architecture settings
- Training parameters
- Runtime configurations
- Prediction modes (monomer/multimer)

### 5. Confidence Metrics (`orcafold/modules/confidence/`)
- pLDDT score calculation
- TM-score estimation
- Per-residue confidence metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/OrcaFold.git
cd OrcaFold

# Create conda environment
conda create -n orcafold python=3.8
conda activate orcafold

# Install dependencies
pip install -r requirements/base.txt
pip install -r requirements/gpu.txt  # for CUDA support

# Install in development mode
pip install -e .
```

## Quick Start

```python
from orcafold import OrcaFold
from orcafold.data import Pipeline

# Initialize pipeline and model
pipeline = Pipeline()
model = OrcaFold(device='cuda')

# Prepare input data
features = pipeline.process_sequence('SEQUENCE.fasta')

# Generate prediction
structure, confidence = model.predict(features)

# Save results
structure.save('predicted_structure.pdb')
```

## Performance Optimization

- CUDA-accelerated computations
- Mixed precision training
- Efficient MSA processing
- Optimized attention mechanisms

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.
Bugs: visualizations

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
