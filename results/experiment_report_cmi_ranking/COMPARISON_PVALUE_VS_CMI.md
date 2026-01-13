# Comparison: P-Value vs CMI-Based Covariate Ranking

**Date:** January 13, 2026
**Study:** anchoring1
**Purpose:** Compare instrument selection between p-value-based and CMI-based EHS scoring

---

## 1. Summary of Change

### Code Change (ci_tests.py:322)

```python
# BEFORE (p-value based, sample-size sensitive)
score = test_i['p_value'] * (1 - test_ii['p_value'])

# AFTER (CMI based, sample-size invariant)
score = test_ii['cmi'] - test_i['cmi']
```

### Rationale

P-values conflate effect size with sample size:
- Large sample + tiny effect → low p-value → high score
- Small sample + large effect → high p-value → low score

CMI measures pure association strength regardless of sample size.

---

## 2. Results Comparison

### 2.1 Best Instrument Selection

| Pattern | P-Value Scoring | CMI Scoring | Changed |
|---------|-----------------|-------------|---------|
| **age** | `resp_gender` | `resp_age_cat` | **YES** |
| **gender** | `resp_gender` | `resp_age_cat` | **YES** |
| **polideo** | `resp_polideo_cat` | `resp_age_cat` | **YES** |

**Key Finding:** CMI scoring consistently selects `resp_age_cat` across all patterns.

### 2.2 Bounds Metrics (Unchanged)

| Metric | age | gender | polideo |
|--------|-----|--------|---------|
| Mean Width | 781.65 | 792.00 | 768.02 |
| Median Width | 311.83 | 311.96 | 312.13 |
| ATE Covered | Yes | Yes | Yes |
| CATE Coverage | 37.9% | 37.9% | 40.0% |

**Note:** Bounds are identical because they are computed for all strata regardless of instrument selection.

---

## 3. Interpretation

### 3.1 Why All Patterns Select `resp_age_cat`

With CMI scoring, `resp_age_cat` shows:
- **High CMI(test_ii)**: Strong association between age and outcome (Y)
- **Low CMI(test_i)**: Weak residual association when conditioning on treatment (X)

This gives a high `CMI(test_ii) - CMI(test_i)` score.

### 3.2 Why P-Value Scoring Differed

With p-value scoring:
- Larger sites dominated the aggregated scores
- `resp_gender` may have had lower p-values in larger sites due to sample size effects
- The p-value formula `p_i × (1 - p_ii)` penalized small p-values differently

### 3.3 Which is "Correct"?

Neither is definitively correct. The difference highlights:
- **P-value scoring** emphasizes statistical significance (sample-dependent)
- **CMI scoring** emphasizes effect magnitude (sample-independent)

For instrument selection across heterogeneous sites with varying sample sizes, CMI scoring is more theoretically appropriate.

---

## 4. Impact Assessment

| Aspect | Impact |
|--------|--------|
| **Bounds computation** | None (unchanged) |
| **Coverage rates** | None (unchanged) |
| **Instrument diagnostics** | Changed (affects reporting/interpretation) |
| **Reproducibility** | Results now consistent across sites with different sample sizes |

---

## 5. Recommendation

The CMI-based scoring is preferred because:
1. Sample-size invariant (fairer comparison across heterogeneous sites)
2. Directly measures information content (interpretable)
3. Avoids conflating statistical significance with practical relevance

However, p-values are still computed and available in the output for diagnostic purposes.

---

## 6. File References

| Report | Path |
|--------|------|
| Original (p-value) | `results/experiment_report/EXPERIMENT_REPORT.md` |
| New (CMI) | `results/experiment_report_cmi_ranking/EXPERIMENT_REPORT_CMI_RANKING.md` |
| This comparison | `results/experiment_report_cmi_ranking/COMPARISON_PVALUE_VS_CMI.md` |
