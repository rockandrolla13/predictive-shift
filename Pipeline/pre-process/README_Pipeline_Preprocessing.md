# Pipeline Project Python Preprocessing

**Date:** December 8, 2025
**Purpose:** Convert R preprocessing scripts to Python for Pipeline Project data

---

## Overview

This document describes the Python preprocessing pipeline for the Pipeline Project data, enabling integration with the OSRCT benchmark.

---

## Data Source

**Pipeline Project OSF Repository:** https://osf.io/s4ygw/

Required files:
- `PPIR_1.sav` - Dataset 1 (Studies 5, 7, 8)
- `PPIR_2.sav` - Dataset 2 (Studies 3, 6, 9)
- `PPIR_3.sav` - Dataset 3 (Studies 1, 2, 4, 10)
- `PPIR_Codebook.xlsx` - Variable codebook

---

## Studies Included

| Study ID | Name | Treatment | Suitable for OSRCT |
|----------|------|-----------|-------------------|
| 1 | Bigot Misanthrope | Binary (0/1) | Yes |
| 2 | Cold-hearted Prosociality | Binary (0/1) | Yes |
| 3 | Bad Tipper | Binary (0/1) | Yes |
| 4 | Belief-Act Consistency | 3 conditions | Yes (compare pairs) |
| 5 | Moral Inversion | 4 conditions | Yes (compare pairs) |
| 6 | Moral Cliff | Binary (0/1) | Yes |
| 7 | Intuitive Economics | Binary (1/2) | Yes |
| 8 | Burn-in-Hell | Within-subjects | **No** |
| 9 | Presumption of Guilt | 4 conditions | Yes (compare pairs) |
| 10 | Higher Standard | 6 conditions | Yes (compare pairs) |

**Note:** Study 8 (Burn-in-Hell) uses a within-subjects design with paired t-tests. It does not have a between-subjects treatment and is **not suitable for OSRCT sampling**.

---

## Usage

### Download Data

1. Go to https://osf.io/s4ygw/
2. Download `PPIR_1.sav`, `PPIR_2.sav`, `PPIR_3.sav`
3. Place in `Pipeline/pre-process/` directory

### Run Preprocessing

```bash
cd Pipeline/pre-process

# Basic usage
python pipeline_preprocess.py --input-dir . --output-dir ./clean_for_analysis

# With custom paths
python pipeline_preprocess.py \
    --input-dir /path/to/raw/data \
    --output-dir /path/to/output \
    --combined-output ./Pipeline_data.csv
```

### Output Files

```
clean_for_analysis/
├── 1_bigot_misanthrope.csv
├── 2_cold_heart.csv
├── 3_bad_tipper.csv
├── 5_moral_inversion.csv
├── 6_moral_cliff.csv
├── 7_intuitive_economics.csv
├── 8_burn_in_hell.csv      # Not for OSRCT
└── Pipeline_data.csv       # Combined OSRCT-suitable
```

---

## Study Details

### Study 7: Intuitive Economics (Recommended for OSRCT)

**Design:** Between-subjects, 2 conditions
**Treatment:** `condition` (1 or 2)
**Outcomes:**
- `htxfair` - How fair are high taxes (1-7 scale)
- `htxgood` - How good are high taxes (1-7 scale)

**Original analysis:** Contrast correlation between htxfair and htxgood across conditions

**For OSRCT:** Use difference-in-means on htxfair or htxgood between conditions

```python
from osrct import OSRCTSampler
import pandas as pd

# Load processed data
data = pd.read_csv('clean_for_analysis/7_intuitive_economics.csv')

# Binarize treatment (1 -> 0, 2 -> 1)
data['iv'] = (data['condition'] == 2).astype(int)
data['dv'] = data['htxfair']

# Create OSRCT sampler
sampler = OSRCTSampler(
    biasing_covariates=['gender', 'yearbirth', 'poltclid'],
    biasing_coefficients={'gender': 0.5, 'yearbirth': 0.01, 'poltclid': 0.3}
)

obs_data, probs = sampler.sample(data, treatment_col='iv')
```

### Study 3: Bad Tipper

**Design:** Between-subjects, 2 conditions
**Treatment:** `condition` (0=less tip, 1=pennies)
**Outcome:** `tipper_personjudg` - Person judgment composite (average of 4 items)

### Study 1: Bigot Misanthrope

**Design:** Between-subjects, 2 conditions
**Treatment:** `condition` (0 or 1)
**Outcome:** `bigot_personjudge` - Person judgment composite

### Study 5: Moral Inversion

**Design:** Between-subjects, 4 conditions
**Treatment:** `mi_condition` (1, 2, 3, or 4)
**Outcome:** `moralgood` = (mi_moral + mi_good) / 2

**For OSRCT:** Compare condition 1 vs 4, or create binary indicator

---

## Demographic Covariates

### Common Covariates Across Studies

| Variable | Description | Range/Values |
|----------|-------------|--------------|
| `gender` | Participant gender | 1=Male, 2=Female |
| `yearbirth` | Year of birth | 1930-2005 |
| `poltclid`/`pltclideo` | Political ideology | 1-7 scale |
| `ethnicity` | Ethnic background | Categorical |
| `parented` | Parent education | Categorical |
| `familyinc`/`faminc` | Family income | Categorical |

### Recommended Covariates for OSRCT

For confounding, use:
- `gender` - Binary demographic
- `yearbirth` - Continuous (can convert to age)
- `poltclid` - Political ideology (often correlated with outcomes)

---

## Integration with OSRCT Benchmark

### Option 1: Generate Confounded Datasets

```python
from osrct import OSRCTSampler
from pathlib import Path
import pandas as pd

# Process all OSRCT-suitable Pipeline studies
studies_config = {
    7: {'treatment': 'condition', 'outcome': 'htxfair', 'recode': {1: 0, 2: 1}},
    3: {'treatment': 'condition', 'outcome': 'tipper_personjudg', 'recode': None},
    1: {'treatment': 'condition', 'outcome': 'bigot_personjudge', 'recode': None},
}

for study_id, config in studies_config.items():
    data = pd.read_csv(f'clean_for_analysis/{study_id}_*.csv')

    # Recode treatment if needed
    if config['recode']:
        data['iv'] = data[config['treatment']].map(config['recode'])
    else:
        data['iv'] = data[config['treatment']]

    data['dv'] = data[config['outcome']]

    # Generate confounded sample
    sampler = OSRCTSampler(
        biasing_covariates=['gender', 'yearbirth'],
        biasing_coefficients={'gender': 0.5, 'yearbirth': 0.02},
        random_seed=42
    )

    obs_data, _ = sampler.sample(data, treatment_col='iv')
    obs_data.to_csv(f'confounded/pipeline_study{study_id}_confounded.csv')
```

### Option 2: Add to Benchmark Package

Extend `generate_confounded_datasets.py` to include Pipeline studies:

```python
# In generate_confounded_datasets.py
PIPELINE_STUDIES = {
    'pipeline_7': {
        'data_path': 'Pipeline/pre-process/clean_for_analysis/7_intuitive_economics.csv',
        'treatment_col': 'iv',  # After recoding
        'outcome_col': 'htxfair',
        'site_col': 'datacollection',
        'covariates': ['gender', 'yearbirth', 'poltclid']
    }
}
```

---

## Dependencies

```
numpy
pandas
pyreadstat
```

Install:
```bash
pip install numpy pandas pyreadstat
```

---

## Validation

After preprocessing, verify:

1. **Row counts** match expected (varies by study)
2. **Treatment is binary** for OSRCT-suitable studies
3. **No missing values** in treatment/outcome
4. **Site distribution** is reasonable

```python
import pandas as pd

data = pd.read_csv('clean_for_analysis/7_intuitive_economics.csv')
print(f"Rows: {len(data)}")
print(f"Treatment distribution:\n{data['condition'].value_counts()}")
print(f"Missing values:\n{data.isnull().sum()}")
print(f"Sites: {data['datacollection'].nunique()}")
```

---

## References

- Schweinsberg, M., et al. (2016). The pipeline project: Pre-publication independent replications of a single laboratory's research pipeline. Journal of Experimental Social Psychology, 66, 55-67.
- OSF Project: https://osf.io/s4ygw/
