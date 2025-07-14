# Deep Natural Selection Model (DNSM) - Molecular Biology and Evolution Archive

This repository contains the code and data for reproducing the experiments from:

**"A sitewise model of natural selection on individual antibodies via a transformer-encoder"**  
*Accepted for publication in Molecular Biology and Evolution*

Authors: Frederick A. Matsen IV, Kevin Sung, Mackenzie M. Johnson, Will Dumm, David Rich, Tyler Starr, Yun S. Song, Philip Bradley, Julia Fukuyama, Hugh K. Haddox

**Netam Version**: This archive corresponds to [netam v0.2.2](https://github.com/matsengrp/netam/releases/tag/v0.2.2)

## Overview

This archive contains all the necessary notebooks, trained models, and supporting code to reproduce the figures and analyses from the paper. The Deep Natural Selection Model (DNSM) predicts the strength of selection on each amino acid site in antibody sequences using a transformer-encoder architecture.

## Key Notebooks

The main analyses from the paper can be reproduced using these notebooks in `notebooks/dnsm_paper/`:

- **`dnsm_oe.ipynb`** - Figures 2, S3, S4: Model fit for predicting per-site probability of nonsynonymous substitution, baseline and neutral model comparisons
- **`dnsm.ipynb`** - Figure 4: Model performance by model size and training-set size  
- **`dnsm_simulation_panel_eval.ipynb`** - Figures 3, S5, S6: Simulation validation showing selection factors agree well with ground truth
- **`along_tree.ipynb`** - Figures 5, 6, S7: Natural selection estimates change as antibodies evolve, selection differs between CDR3s of different lengths
- **`surprising_sites.ipynb`** - Figures 7, S8, S9: Counting analysis of synonymous vs nonsynonymous mutations at exposed conserved and buried divergent sites
- **`dnsm_asa.ipynb`** - Figure 8, Table S3, S4: Comparison of selection factors with solvent accessibility on SAbDAb
- **`wiggle.ipynb`** - Figure S10: The wiggle activation function

Additional supporting notebooks:
- **`data_prep.ipynb`** - Data preprocessing  
- **`dnsm_model_summaries.ipynb`** - Model architecture summaries
- **`dnsm_clustered_selection_factors.ipynb`** - Analysis of clustered selection factors

## Installation

### 1. Install netam

First, install the [netam](https://github.com/matsengrp/netam) package in a virtual environment:

```bash
# Create and activate virtual environment
python -m venv netam_env
source netam_env/bin/activate

# Clone and install netam v0.2.2
git clone https://github.com/matsengrp/netam.git
cd netam
git checkout v0.2.2
pip install -e .
cd ..
```

### 2. Install this package

```bash
# Clone this repository
git clone https://github.com/matsengrp/dnsm-experiments-mbe.git
cd dnsm-experiments-mbe

# Install in development mode
pip install -e .
```

### 3. Configure local paths

Edit `dnsmex/local.py` to set paths appropriate for your system. The key variables to configure are:

- `DNSM_TRAINED_MODELS_DIR` - Path to trained models directory
- `FIGURES_DIR` - Path where figures should be saved  
- `DATA_DIR` - Path to data files

## Training Models (Optional)

The repository includes pre-trained models in `dnsm-train/trained_models/`. If you want to retrain models:

### Prerequisites

For parallel branch length optimization, increase file descriptor limits:

```bash
ulimit -n 8192
```

### Training

```bash
cd dnsm-train
snakemake -n  # dry run to see what would be executed
snakemake -j4  # run with 4 parallel jobs
```

## Reproducing Paper Figures

To reproduce all figures from the paper:

```bash
# Run all DNSM paper notebooks
./run_notebooks.sh notebooks/dnsm_paper/
```

Or run individual notebooks using Jupyter:

```bash
jupyter notebook notebooks/dnsm_paper/dnsm_oe.ipynb
```

## Data

The `data/` directory contains:

- **`human_ighv_aa_seqs.csv`** - Human IGHV amino acid sequences for germline analysis
- **`sabdab_summary_2024-01-26_abid_info_resnums.tsv.gz`** - SAbDAb antibody structure data
- Additional supporting data files for specific analyses

## Visualizations

Interactive 3D visualizations of natural selection factors are available at:
[https://matsengrp.github.io/dnsm-viz/v1/](https://matsengrp.github.io/dnsm-viz/v1/)

Key examples from the paper:
- [6MTX](https://matsengrp.github.io/dnsm-viz/v1/?pdbid=6mtx) - Shows strong diversification at IMGT site 24
- [5I19](https://matsengrp.github.io/dnsm-viz/v1/?pdbid=5i19) - Examples of buried sites with nonsynonymous substitutions
- [3B2V](https://matsengrp.github.io/dnsm-viz/v1/?pdbid=3b2v) - IMGT site 24 diversifying selection

## Repository Structure

```
├── notebooks/dnsm_paper/     # Main analysis notebooks for the paper
├── dnsm-train/              # Model training pipeline and trained models  
├── dnsm-grid/               # Hyperparameter grid search experiments
├── dnsmex/                  # Python package with analysis utilities
├── data/                    # Supporting data files
└── README.md               # This file
```

## Citation

If you use this code or data, please cite:

```bibtex
@article{matsen2025dnsm,
  title={A sitewise model of natural selection on individual antibodies via a transformer-encoder},
  author={Matsen IV, Frederick A and Sung, Kevin and Johnson, Mackenzie M and Dumm, Will and Rich, David and Starr, Tyler and Song, Yun S and Bradley, Philip and Fukuyama, Julia and Haddox, Hugh K},
  journal={Molecular Biology and Evolution},
  year={2025},
  note={Accepted for publication}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions about the code or methods, please open an issue on the repository or contact the corresponding author.