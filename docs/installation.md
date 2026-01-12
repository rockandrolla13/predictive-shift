# Installation Guide

## Requirements

- Python 3.8 or higher
- pip or conda package manager

## Dependencies

### Required

```
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
```

### Optional

```
pyreadstat          # For reading SPSS .sav files
econml>=0.14.0      # For Causal Forest method
matplotlib>=3.5.0   # For visualizations
seaborn>=0.11.0     # For enhanced plots
jupyter             # For notebooks
```

---

## Installation Methods

### Method 1: pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/[username]/osrct-benchmark.git
cd osrct-benchmark

# Install dependencies
pip install -r requirements.txt

# Optional: Install development dependencies
pip install -r requirements-dev.txt
```

### Method 2: Conda

```bash
# Clone the repository
git clone https://github.com/[username]/osrct-benchmark.git
cd osrct-benchmark

# Create and activate environment
conda env create -f environment.yml
conda activate osrct-benchmark
```

### Method 3: Manual Installation

```bash
# Install core dependencies
pip install numpy pandas scipy scikit-learn

# Install optional dependencies
pip install pyreadstat econml matplotlib seaborn jupyter
```

---

## Verification

Verify your installation by running the validation script:

```bash
cd osrct_benchmark_v1.0
python validate_benchmark.py
```

Expected output:
```
============================================================
OSRCT Benchmark Validation
============================================================

  Directory structure: PASS
  Required files: PASS
  Datasets: PASS
  Ground truth: PASS
  Metadata: PASS
  Code: PASS
  Analysis results: PASS
  Checksum verification: PASS

  Datasets: 525
  Total size: 586.6 MB
  Errors: 0
  Warnings: 0

============================================================
VALIDATION PASSED
============================================================
```

---

## Quick Test

Test the core functionality:

```python
import pandas as pd
import sys
sys.path.append('osrct_benchmark_v1.0/code')

from osrct import OSRCTSampler, load_manylabs1_data
from causal_methods import CausalMethodEvaluator

# Load a sample dataset
data = pd.read_csv('osrct_benchmark_v1.0/confounded_datasets/by_study/anchoring1/anchoring1_age_beta0.5.csv')
print(f"Loaded dataset with {len(data)} observations")

# Evaluate methods
evaluator = CausalMethodEvaluator()
results = evaluator.evaluate_all(data, ground_truth_ate=1555.67, skip_causal_forest=True)
print(results[['method', 'ate', 'bias']])
```

---

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'pyreadstat'`

**Solution**: Install pyreadstat for SPSS file support:
```bash
pip install pyreadstat
```

---

**Issue**: `ModuleNotFoundError: No module named 'econml'`

**Solution**: Causal Forest requires econml:
```bash
pip install econml
```

Note: econml has additional dependencies. On some systems you may need:
```bash
pip install econml[all]
```

---

**Issue**: Memory errors with large datasets

**Solution**: Process datasets one at a time:
```python
# Instead of loading all datasets
for study in studies:
    data = pd.read_csv(f'path/to/{study}.csv')
    # Process
    del data  # Free memory
```

---

## Platform-Specific Notes

### Linux

No special requirements. All dependencies install via pip.

### macOS

For M1/M2 Macs, some packages may need Rosetta:
```bash
arch -x86_64 pip install econml
```

### Windows

Use Anaconda for easiest installation:
```bash
conda install numpy pandas scipy scikit-learn
pip install pyreadstat econml
```

---

## Development Setup

For contributing to the project:

```bash
# Clone with full history
git clone https://github.com/[username]/osrct-benchmark.git
cd osrct-benchmark

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 osrct_benchmark_v1.0/code/
black --check osrct_benchmark_v1.0/code/
```
