---
name: Method Submission
about: Submit a new causal inference method for the leaderboard
title: '[METHOD] '
labels: method-submission
assignees: ''
---

## Method Information

**Method Name:**
**Method Type:** [propensity-based / outcome-based / doubly-robust / ML-based / other]

## Authors
- Name(s):
- Affiliation(s):
- Email:

## References
- Paper URL:
- Code Repository:

## Method Description
Brief description of the method and its key assumptions.

## Implementation

### Code
```python
def your_method(data, treatment_col, outcome_col, covariates):
    """
    Your causal inference method.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with treatment, outcome, and covariates
    treatment_col : str
        Name of treatment column ('iv')
    outcome_col : str
        Name of outcome column ('dv')
    covariates : list
        List of covariate column names

    Returns
    -------
    dict with keys:
        - 'ate': float, estimated ATE
        - 'se': float, standard error
        - 'ci_lower': float, lower CI bound
        - 'ci_upper': float, upper CI bound
    """
    # Your implementation here
    pass
```

### Dependencies
List any additional packages required:
-

## Validation Results

Have you tested your method on the sample datasets?
- [ ] Yes, results attached below
- [ ] No

### Sample Results (if available)
| Dataset | True ATE | Estimated ATE | Bias |
|---------|----------|---------------|------|
| anchoring1_age_beta0.5 | | | |
| gainloss_gender_beta1.0 | | | |

## Checklist
- [ ] Method returns dict with required keys (ate, se, ci_lower, ci_upper)
- [ ] Method handles missing covariates gracefully
- [ ] Method runs in reasonable time (<60 seconds per dataset)
- [ ] Code is documented
- [ ] Dependencies are listed
