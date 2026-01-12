#!/usr/bin/env python3
"""
Phase 6.1: Package OSRCT Benchmark for Release

This script packages all generated datasets, analysis results, and documentation
into a standardized benchmark release structure.

Usage:
    python package_benchmark.py --output-dir osrct_benchmark_v1.0
"""

import os
import sys
import json
import hashlib
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent
CONFOUNDED_DIR = PROJECT_ROOT / "confounded_datasets"
ANALYSIS_DIR = PROJECT_ROOT / "analysis_results"
PREPROCESS_DIR = PROJECT_ROOT / "ManyLabs1" / "pre-process"


def compute_md5(filepath: Path) -> str:
    """Compute MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def create_directory_structure(output_dir: Path) -> Dict[str, Path]:
    """Create the benchmark release directory structure."""
    dirs = {
        'root': output_dir,
        'raw_data': output_dir / "raw_rct_data",
        'confounded': output_dir / "confounded_datasets",
        'confounded_by_study': output_dir / "confounded_datasets" / "by_study",
        'confounded_by_beta': output_dir / "confounded_datasets" / "by_confounding_strength",
        'ground_truth': output_dir / "ground_truth",
        'analysis': output_dir / "analysis_results",
        'analysis_figures': output_dir / "analysis_results" / "figures",
        'analysis_method': output_dir / "analysis_results" / "method_evaluation",
        'code': output_dir / "code",
        'code_examples': output_dir / "code" / "examples",
        'metadata': output_dir / "metadata",
    }

    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {path.relative_to(output_dir)}")

    return dirs


def create_data_dictionary() -> Dict[str, Any]:
    """Create comprehensive data dictionary for the benchmark."""
    return {
        "version": "1.0.0",
        "created": datetime.now().isoformat(),
        "description": "OSRCT Benchmark: Semi-synthetic confounded observational datasets from ManyLabs1 RCTs",

        "dataset_variables": {
            "id": {
                "type": "string",
                "description": "Unique participant identifier"
            },
            "iv": {
                "type": "binary (0/1)",
                "description": "Treatment indicator (0=control, 1=treated)"
            },
            "dv": {
                "type": "numeric",
                "description": "Outcome variable (study-specific)"
            },
            "site": {
                "type": "categorical",
                "description": "Lab/site identifier (36 unique sites)"
            },
            "original_study": {
                "type": "categorical",
                "description": "Study identifier (15 unique studies)"
            },
            "resp_age": {
                "type": "numeric",
                "description": "Participant age (years)"
            },
            "resp_gender": {
                "type": "binary (0/1)",
                "description": "Participant gender (0=female, 1=male)"
            },
            "resp_polideo": {
                "type": "ordinal (0-6)",
                "description": "Political ideology (0=Strongly Conservative to 6=Strongly Liberal)"
            },
            "resp_ethnicity_hisp": {
                "type": "binary (0/1)",
                "description": "Hispanic/Latino ethnicity indicator"
            },
            "resp_american": {
                "type": "ordinal (1-11)",
                "description": "American identity strength"
            },
            "_selection_prob": {
                "type": "numeric (0-1)",
                "description": "OSRCT selection probability for this observation"
            }
        },

        "studies": {
            "anchoring1": {
                "full_name": "Anchoring (NYC population)",
                "outcome_type": "continuous",
                "treatment": "High vs Low anchor",
                "expected_effect": "Large positive"
            },
            "anchoring2": {
                "full_name": "Anchoring (Chicago population)",
                "outcome_type": "continuous",
                "treatment": "High vs Low anchor",
                "expected_effect": "Large positive"
            },
            "anchoring3": {
                "full_name": "Anchoring (Mt. Everest height)",
                "outcome_type": "continuous",
                "treatment": "High vs Low anchor",
                "expected_effect": "Large positive"
            },
            "anchoring4": {
                "full_name": "Anchoring (Baby births)",
                "outcome_type": "continuous",
                "treatment": "High vs Low anchor",
                "expected_effect": "Large positive"
            },
            "gainloss": {
                "full_name": "Gain vs Loss framing",
                "outcome_type": "binary",
                "treatment": "Gain vs Loss frame",
                "expected_effect": "Medium"
            },
            "sunk": {
                "full_name": "Sunk cost effect",
                "outcome_type": "ordinal (1-9)",
                "treatment": "Sunk cost present vs absent",
                "expected_effect": "Medium"
            },
            "allowedforbidden": {
                "full_name": "Allowed/Forbidden framing",
                "outcome_type": "binary",
                "treatment": "Word choice framing",
                "expected_effect": "Medium"
            },
            "reciprocity": {
                "full_name": "Norm of reciprocity",
                "outcome_type": "binary",
                "treatment": "Question order manipulation",
                "expected_effect": "Small"
            },
            "flag": {
                "full_name": "Flag priming",
                "outcome_type": "continuous",
                "treatment": "Flag exposure",
                "expected_effect": "Small"
            },
            "quote": {
                "full_name": "Quote attribution",
                "outcome_type": "continuous",
                "treatment": "Source attribution",
                "expected_effect": "Medium"
            },
            "gambfal": {
                "full_name": "Gambler's fallacy",
                "outcome_type": "continuous",
                "treatment": "Scenario condition",
                "expected_effect": "Medium"
            },
            "scales": {
                "full_name": "Low vs High scales",
                "outcome_type": "continuous",
                "treatment": "Scale anchoring",
                "expected_effect": "Medium"
            },
            "contact": {
                "full_name": "Imagined contact",
                "outcome_type": "continuous",
                "treatment": "Contact imagination",
                "expected_effect": "Medium"
            },
            "money": {
                "full_name": "Currency priming",
                "outcome_type": "continuous",
                "treatment": "Money exposure",
                "expected_effect": "Small"
            },
            "iat": {
                "full_name": "Implicit Association Test",
                "outcome_type": "continuous",
                "treatment": "Gender (Female vs Male)",
                "expected_effect": "Medium"
            }
        },

        "confounding_patterns": {
            "age": {
                "covariates": ["resp_age"],
                "type": "single-covariate",
                "description": "Confounding through age only"
            },
            "gender": {
                "covariates": ["resp_gender"],
                "type": "single-covariate",
                "description": "Confounding through gender only"
            },
            "polideo": {
                "covariates": ["resp_polideo"],
                "type": "single-covariate",
                "description": "Confounding through political ideology only"
            },
            "demo_basic": {
                "covariates": ["resp_age", "resp_gender"],
                "type": "multi-covariate",
                "description": "Confounding through age and gender"
            },
            "demo_full": {
                "covariates": ["resp_age", "resp_gender", "resp_polideo"],
                "type": "multi-covariate",
                "description": "Confounding through age, gender, and political ideology"
            }
        },

        "beta_values": {
            "0.1": "Very weak confounding",
            "0.25": "Weak confounding",
            "0.5": "Moderate confounding",
            "0.75": "Moderate-strong confounding",
            "1.0": "Strong confounding",
            "1.5": "Very strong confounding",
            "2.0": "Extreme confounding"
        },

        "sites": [
            "abington", "brasilia", "charles", "conncoll", "csun", "help",
            "ithaca", "jmu", "ku", "laurier", "lse", "luc", "mcdaniel", "msvu",
            "mturk", "osu", "oxy", "pi", "psu", "qccuny", "qccuny2", "sdsu",
            "swps", "swpson", "tamu", "tamuc", "tamuon", "tilburg", "ufl",
            "unipd", "uva", "vcu", "wisc", "wku", "wl", "wpi"
        ]
    }


def generate_metadata_catalog(confounded_dir: Path) -> pd.DataFrame:
    """Generate comprehensive metadata catalog for all datasets."""
    print("\nGenerating metadata catalog...")

    catalog = []

    # Load generation summary if available
    gen_summary_path = confounded_dir / "generation_summary.csv"
    if gen_summary_path.exists():
        gen_summary = pd.read_csv(gen_summary_path)
    else:
        gen_summary = None

    # Load ground truth ATEs
    gt_path = confounded_dir / "ground_truth_ates.csv"
    if gt_path.exists():
        ground_truth = pd.read_csv(gt_path)
        gt_dict = ground_truth.set_index('study').to_dict('index')
    else:
        gt_dict = {}

    # Iterate through all dataset files
    studies = [d for d in confounded_dir.iterdir()
               if d.is_dir() and d.name not in ['metadata', 'site_stratified']]

    total_files = sum(1 for s in studies for f in s.glob("*.csv"))
    processed = 0

    for study_dir in sorted(studies):
        study_name = study_dir.name
        gt_info = gt_dict.get(study_name, {})

        for csv_file in sorted(study_dir.glob("*.csv")):
            processed += 1
            if processed % 50 == 0:
                print(f"  Processed {processed}/{total_files} files...")

            # Parse filename
            filename = csv_file.stem
            parts = filename.split("_")

            # Extract pattern and beta
            if len(parts) >= 3:
                # Handle patterns like "demo_basic_beta0.5_seed42"
                beta_idx = next((i for i, p in enumerate(parts) if p.startswith("beta")), None)
                if beta_idx:
                    pattern = "_".join(parts[:beta_idx])
                    beta_str = parts[beta_idx].replace("beta", "")
                    try:
                        beta = float(beta_str)
                    except:
                        beta = None
                else:
                    pattern = parts[0]
                    beta = None
            else:
                pattern = filename
                beta = None

            # Load dataset to get stats
            try:
                df = pd.read_csv(csv_file)
                n_total = len(df)
                n_treated = (df['iv'] == 1).sum() if 'iv' in df.columns else None
                n_control = (df['iv'] == 0).sum() if 'iv' in df.columns else None
                treatment_rate = n_treated / n_total if n_total > 0 else None

                # Compute naive ATE
                if 'iv' in df.columns and 'dv' in df.columns:
                    treated_mean = df[df['iv'] == 1]['dv'].mean()
                    control_mean = df[df['iv'] == 0]['dv'].mean()
                    naive_ate = treated_mean - control_mean
                else:
                    naive_ate = None

            except Exception as e:
                n_total = n_treated = n_control = treatment_rate = naive_ate = None

            # Compute checksum
            md5 = compute_md5(csv_file)

            # Get ground truth ATE
            gt_ate = gt_info.get('ate', None)

            # Compute bias if possible
            confounding_bias = naive_ate - gt_ate if naive_ate and gt_ate else None

            catalog.append({
                'dataset_id': filename,
                'study': study_name,
                'pattern': pattern,
                'beta': beta,
                'filename': str(csv_file.relative_to(confounded_dir)),
                'n_total': n_total,
                'n_treated': n_treated,
                'n_control': n_control,
                'treatment_rate': treatment_rate,
                'ground_truth_ate': gt_ate,
                'naive_ate': naive_ate,
                'confounding_bias': confounding_bias,
                'md5_checksum': md5,
                'file_size_bytes': csv_file.stat().st_size
            })

    print(f"  Processed {processed}/{total_files} files")
    return pd.DataFrame(catalog)


def copy_datasets(source_dir: Path, dest_dirs: Dict[str, Path]) -> None:
    """Copy datasets to release structure."""
    print("\nCopying datasets...")

    # Copy by study
    studies = [d for d in source_dir.iterdir()
               if d.is_dir() and d.name not in ['metadata', 'site_stratified']]

    for study_dir in sorted(studies):
        study_name = study_dir.name
        dest_study = dest_dirs['confounded_by_study'] / study_name
        dest_study.mkdir(exist_ok=True)

        for csv_file in study_dir.glob("*.csv"):
            shutil.copy2(csv_file, dest_study / csv_file.name)

        print(f"  Copied {study_name}")

    # Copy ground truth
    gt_file = source_dir / "ground_truth_ates.csv"
    if gt_file.exists():
        shutil.copy2(gt_file, dest_dirs['ground_truth'] / "rct_ates.csv")
        print("  Copied ground_truth_ates.csv")

    # Copy generation summary
    gen_file = source_dir / "generation_summary.csv"
    if gen_file.exists():
        shutil.copy2(gen_file, dest_dirs['metadata'] / "generation_summary.csv")
        print("  Copied generation_summary.csv")

    # Copy validation report
    val_file = source_dir / "validation_report.json"
    if val_file.exists():
        shutil.copy2(val_file, dest_dirs['metadata'] / "validation_report.json")
        print("  Copied validation_report.json")

    # Copy site-stratified summary
    ss_file = source_dir / "site_stratified_summary.csv"
    if ss_file.exists():
        shutil.copy2(ss_file, dest_dirs['ground_truth'] / "site_stratified_summary.csv")
        print("  Copied site_stratified_summary.csv")


def copy_analysis_results(source_dir: Path, dest_dirs: Dict[str, Path]) -> None:
    """Copy analysis results to release structure."""
    print("\nCopying analysis results...")

    if not source_dir.exists():
        print("  Analysis results directory not found, skipping")
        return

    # Copy figures
    figures_dir = source_dir / "figures"
    if figures_dir.exists():
        for fig in figures_dir.glob("*"):
            shutil.copy2(fig, dest_dirs['analysis_figures'] / fig.name)
        print(f"  Copied {len(list(figures_dir.glob('*')))} figures")

    # Copy method evaluation
    method_dir = source_dir / "method_evaluation"
    if method_dir.exists():
        for f in method_dir.glob("*"):
            shutil.copy2(f, dest_dirs['analysis_method'] / f.name)
        print(f"  Copied method evaluation results")

    # Copy findings
    findings_file = source_dir / "phase5_findings.md"
    if findings_file.exists():
        shutil.copy2(findings_file, dest_dirs['analysis'] / "findings_summary.md")
        print("  Copied findings summary")


def copy_code(dest_dirs: Dict[str, Path]) -> None:
    """Copy relevant code files to release."""
    print("\nCopying code files...")

    code_files = [
        ("osrct.py", "Core OSRCT sampling algorithm"),
        ("generate_confounded_datasets.py", "Batch dataset generation"),
        ("causal_methods.py", "Causal inference method implementations"),
        ("experimental_grid.py", "Experimental grid configuration"),
    ]

    for filename, desc in code_files:
        src = PROJECT_ROOT / filename
        if src.exists():
            shutil.copy2(src, dest_dirs['code'] / filename)
            print(f"  Copied {filename}")


def create_readme(dest_dirs: Dict[str, Path], catalog: pd.DataFrame) -> None:
    """Create comprehensive README for the benchmark release."""
    print("\nCreating README...")

    n_datasets = len(catalog)
    n_studies = catalog['study'].nunique()
    n_patterns = catalog['pattern'].nunique()
    total_size_mb = catalog['file_size_bytes'].sum() / (1024 * 1024)

    readme = f"""# OSRCT Benchmark v1.0

**Semi-Synthetic Confounded Observational Datasets for Causal Inference Evaluation**

Generated: {datetime.now().strftime('%Y-%m-%d')}

---

## Overview

This benchmark provides **{n_datasets} confounded observational datasets** derived from real
randomized controlled trials (ManyLabs1) using the OSRCT algorithm (Gentzel et al., 2021).

### Key Features

- **Ground-truth ATEs**: True treatment effects known from original RCTs
- **Controlled confounding**: Systematic confounding via OSRCT sampling
- **Multiple studies**: {n_studies} psychological experiments
- **Multiple confounding patterns**: {n_patterns} covariate configurations
- **7 confounding strengths**: beta from 0.1 (weak) to 2.0 (extreme)

---

## Quick Start

### Load a Dataset (Python)

```python
import pandas as pd

# Load a confounded dataset
data = pd.read_csv('confounded_datasets/by_study/anchoring1/age_beta0.5_seed42.csv')

# Load ground truth
ground_truth = pd.read_csv('ground_truth/rct_ates.csv')
true_ate = ground_truth[ground_truth['study'] == 'anchoring1']['ate'].values[0]

# Your causal inference method
estimated_ate = your_method(data, treatment='iv', outcome='dv', covariates=['resp_age'])

# Evaluate
bias = estimated_ate - true_ate
print(f"True ATE: {{true_ate:.3f}}, Estimated: {{estimated_ate:.3f}}, Bias: {{bias:.3f}}")
```

### Load a Dataset (R)

```r
library(readr)

# Load confounded dataset
data <- read_csv('confounded_datasets/by_study/anchoring1/age_beta0.5_seed42.csv')

# Load ground truth
ground_truth <- read_csv('ground_truth/rct_ates.csv')
true_ate <- ground_truth$ate[ground_truth$study == 'anchoring1']
```

---

## Directory Structure

```
osrct_benchmark_v1.0/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── CITATION.cff                       # Citation metadata
│
├── raw_rct_data/                      # Original preprocessed RCT data
│   └── data_dictionary.json           # Variable descriptions
│
├── confounded_datasets/               # Main benchmark datasets
│   ├── by_study/                      # Organized by psychological study
│   │   ├── anchoring1/
│   │   ├── anchoring2/
│   │   └── ...
│   └── by_confounding_strength/       # Alternative organization
│
├── ground_truth/                      # True treatment effects
│   ├── rct_ates.csv                   # Study-level ATEs
│   └── site_stratified_summary.csv    # Site-level ATEs
│
├── analysis_results/                  # Pre-computed method evaluations
│   ├── figures/
│   ├── method_evaluation/
│   └── findings_summary.md
│
├── metadata/                          # Dataset metadata
│   ├── dataset_catalog.csv            # Full dataset catalog
│   ├── generation_summary.csv         # Generation details
│   └── checksums.txt                  # MD5 checksums
│
└── code/                              # Source code
    ├── osrct.py                       # OSRCT algorithm
    └── examples/                      # Usage examples
```

---

## Dataset Naming Convention

```
{{pattern}}_beta{{strength}}_seed{{seed}}.csv

Examples:
  age_beta0.5_seed42.csv        - Age confounding, moderate strength
  gender_beta1.0_seed42.csv     - Gender confounding, strong
  demo_full_beta0.1_seed42.csv  - Full demographics, weak confounding
```

---

## Studies Included

| Study | Outcome Type | True ATE | Description |
|-------|--------------|----------|-------------|
| anchoring1 | Continuous | ~1556 | NYC population estimation |
| anchoring2 | Continuous | ~2029 | Chicago population estimation |
| anchoring3 | Continuous | ~2418 | Mt. Everest height estimation |
| anchoring4 | Continuous | ~2495 | Daily baby births estimation |
| gainloss | Binary | ~0.29 | Gain vs Loss framing |
| sunk | Ordinal | ~0.61 | Sunk cost effect |
| flag | Continuous | ~0.03 | Flag priming on conservatism |
| quote | Continuous | ~0.70 | Quote attribution |
| reciprocity | Binary | ~0.13 | Norm of reciprocity |
| gambfal | Continuous | ~1.70 | Gambler's fallacy |
| scales | Continuous | ~0.17 | Scale anchoring |
| contact | Continuous | ~0.25 | Imagined contact |
| money | Continuous | ~-0.02 | Currency priming |
| iat | Continuous | ~0.26 | Implicit Association Test |
| allowedforbidden | Binary | ~-0.17 | Allowed/Forbidden framing |

---

## Confounding Patterns

| Pattern | Covariates | Type |
|---------|------------|------|
| age | resp_age | Single continuous |
| gender | resp_gender | Single binary |
| polideo | resp_polideo | Single ordinal |
| demo_basic | resp_age, resp_gender | Multi-covariate |
| demo_full | resp_age, resp_gender, resp_polideo | Multi-covariate |

---

## Confounding Strength (Beta)

| Beta | Interpretation | Expected Naive Bias |
|------|----------------|---------------------|
| 0.1 | Very weak | Minimal |
| 0.25 | Weak | Small |
| 0.5 | Moderate | Moderate |
| 0.75 | Moderate-strong | Substantial |
| 1.0 | Strong | Large |
| 1.5 | Very strong | Very large |
| 2.0 | Extreme | Extreme |

---

## Method Evaluation Results

From our Phase 5 analysis (5 methods evaluated on {n_datasets} datasets):

| Rank | Method | RMSE | Mean Bias |
|------|--------|------|-----------|
| 1 | IPW | 23.24 | -4.03 |
| 2 | Outcome Regression | 24.20 | +5.67 |
| 3 | AIPW (Doubly Robust) | 24.33 | -5.15 |
| 4 | Naive | 28.72 | +0.90 |
| 5 | PSM | 79.49 | +9.75 |

---

## Citation

If you use this benchmark, please cite:

```bibtex
@misc{{osrct_benchmark_2025,
  title={{OSRCT Benchmark: Semi-Synthetic Datasets for Causal Inference Evaluation}},
  author={{[Authors]}},
  year={{2025}},
  note={{Based on ManyLabs1 (Klein et al., 2014) and OSRCT (Gentzel et al., 2021)}}
}}
```

### Original Data Sources

- **ManyLabs1**: Klein, R. A., et al. (2014). Investigating variation in replicability. *Social Psychology*, 45(3), 142-152.
- **OSRCT Algorithm**: Gentzel, M., Garant, D., & Jensen, D. (2021). The case for evaluating causal models using interventional measures and empirical data. *NeurIPS 2021*.

---

## License

This benchmark is released under the MIT License.
The original ManyLabs1 data is released under CC0 license.

---

## Contact

For questions or issues, please open an issue on the GitHub repository.
"""

    readme_path = dest_dirs['root'] / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme)

    print(f"  Created README.md")


def create_license(dest_dirs: Dict[str, Path]) -> None:
    """Create MIT License file."""
    license_text = f"""MIT License

Copyright (c) {datetime.now().year} OSRCT Benchmark Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

Note: The original ManyLabs1 dataset is released under the CC0 license.
See: https://osf.io/wx7ck/
"""

    with open(dest_dirs['root'] / "LICENSE", 'w') as f:
        f.write(license_text)

    print("  Created LICENSE")


def create_citation(dest_dirs: Dict[str, Path]) -> None:
    """Create CITATION.cff file."""
    citation = f"""cff-version: 1.2.0
message: "If you use this benchmark, please cite it as below."
title: "OSRCT Benchmark: Semi-Synthetic Datasets for Causal Inference Evaluation"
version: 1.0.0
date-released: {datetime.now().strftime('%Y-%m-%d')}
authors:
  - name: "OSRCT Benchmark Authors"
keywords:
  - causal inference
  - benchmark
  - observational data
  - treatment effects
  - OSRCT
  - ManyLabs1
license: MIT
repository-code: "https://github.com/[username]/osrct-benchmark"
references:
  - type: article
    authors:
      - family-names: Klein
        given-names: Richard A.
    title: "Investigating variation in replicability"
    journal: "Social Psychology"
    year: 2014
    volume: 45
    issue: 3
    start: 142
    end: 152
  - type: conference-paper
    authors:
      - family-names: Gentzel
        given-names: Michael
    title: "The case for evaluating causal models using interventional measures and empirical data"
    conference: "NeurIPS 2021"
    year: 2021
"""

    with open(dest_dirs['root'] / "CITATION.cff", 'w') as f:
        f.write(citation)

    print("  Created CITATION.cff")


def generate_checksums(dest_dirs: Dict[str, Path], catalog: pd.DataFrame) -> None:
    """Generate checksums file for all datasets."""
    print("\nGenerating checksums...")

    checksums = []
    for _, row in catalog.iterrows():
        checksums.append(f"{row['md5_checksum']}  {row['filename']}")

    checksum_path = dest_dirs['metadata'] / "checksums.txt"
    with open(checksum_path, 'w') as f:
        f.write("\n".join(checksums))

    print(f"  Generated checksums for {len(checksums)} files")


def validate_package(dest_dirs: Dict[str, Path]) -> Dict[str, bool]:
    """Validate the packaged benchmark."""
    print("\nValidating package...")

    checks = {}

    # Check required directories exist
    required_dirs = ['confounded_by_study', 'ground_truth', 'metadata', 'code']
    for dir_name in required_dirs:
        checks[f'dir_{dir_name}'] = dest_dirs[dir_name].exists()

    # Check required files exist
    required_files = [
        (dest_dirs['root'] / "README.md", 'readme'),
        (dest_dirs['root'] / "LICENSE", 'license'),
        (dest_dirs['ground_truth'] / "rct_ates.csv", 'ground_truth'),
        (dest_dirs['metadata'] / "dataset_catalog.csv", 'catalog'),
        (dest_dirs['metadata'] / "data_dictionary.json", 'data_dict'),
    ]

    for filepath, name in required_files:
        checks[f'file_{name}'] = filepath.exists()

    # Check dataset count
    csv_count = sum(1 for _ in dest_dirs['confounded_by_study'].rglob("*.csv"))
    checks['datasets_present'] = csv_count > 500
    checks['dataset_count'] = csv_count

    # Print validation results
    all_passed = True
    for check, result in checks.items():
        if isinstance(result, bool):
            status = "PASS" if result else "FAIL"
            if not result:
                all_passed = False
        else:
            status = str(result)
        print(f"  {check}: {status}")

    checks['all_passed'] = all_passed
    return checks


def main():
    parser = argparse.ArgumentParser(
        description='Package OSRCT Benchmark for release'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='osrct_benchmark_v1.0',
        help='Output directory for packaged benchmark'
    )
    parser.add_argument(
        '--skip-copy',
        action='store_true',
        help='Skip copying datasets (for testing)'
    )

    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir

    print("=" * 60)
    print("OSRCT Benchmark Packaging")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")

    # Step 1: Create directory structure
    print("\n[1/8] Creating directory structure...")
    dirs = create_directory_structure(output_dir)

    # Step 2: Generate metadata catalog
    print("\n[2/8] Generating metadata catalog...")
    catalog = generate_metadata_catalog(CONFOUNDED_DIR)
    catalog.to_csv(dirs['metadata'] / "dataset_catalog.csv", index=False)
    print(f"  Catalog contains {len(catalog)} datasets")

    # Step 3: Create data dictionary
    print("\n[3/8] Creating data dictionary...")
    data_dict = create_data_dictionary()
    with open(dirs['metadata'] / "data_dictionary.json", 'w') as f:
        json.dump(data_dict, f, indent=2)
    print("  Created data_dictionary.json")

    # Step 4: Copy datasets
    if not args.skip_copy:
        print("\n[4/8] Copying datasets...")
        copy_datasets(CONFOUNDED_DIR, dirs)
    else:
        print("\n[4/8] Skipping dataset copy (--skip-copy)")

    # Step 5: Copy analysis results
    print("\n[5/8] Copying analysis results...")
    copy_analysis_results(ANALYSIS_DIR, dirs)

    # Step 6: Copy code
    print("\n[6/8] Copying code...")
    copy_code(dirs)

    # Step 7: Create documentation
    print("\n[7/8] Creating documentation...")
    create_readme(dirs, catalog)
    create_license(dirs)
    create_citation(dirs)
    generate_checksums(dirs, catalog)

    # Step 8: Validate
    print("\n[8/8] Validating package...")
    validation = validate_package(dirs)

    # Summary
    print("\n" + "=" * 60)
    print("PACKAGING COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {output_dir}")
    print(f"Datasets: {validation.get('dataset_count', 'N/A')}")
    print(f"Validation: {'PASSED' if validation.get('all_passed') else 'FAILED'}")

    # Calculate total size
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"Total size: {total_size / (1024*1024):.1f} MB")

    return 0 if validation.get('all_passed') else 1


if __name__ == "__main__":
    sys.exit(main())
