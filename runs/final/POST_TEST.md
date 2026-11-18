# Post-test exploration — C-LS (LLM listwise selection) vs frozen C-full

**Not part of the frozen results.** Designed after the test split was spent (plan.md, 'Post-test exploration'); dev gate fixed in advance and passed (pooled dev R@8k +0.048 [+0.004, +0.093], n=135); then run once on test (TEST_RUNS.md). Δ = C-LS − C-full, paired bootstrap 95% CI; bold = excludes 0.


## dev_tune (n = 90 answerable)

| budget | recall C-LS | recall C-full | Δ recall | Δ MRR |
|---|---|---|---|---|
| 2000 | 0.722 | 0.606 | **+0.117** [+0.039, +0.200] | **+0.128** [+0.053, +0.206] |
| 8000 | 0.850 | 0.811 | +0.039 [-0.017, +0.100] | **+0.101** [+0.037, +0.166] |
| 32000 | 0.894 | 0.894 | +0.000 [+0.000, +0.000] | **+0.094** [+0.040, +0.150] |

| category @8k | C-LS | C-full |
|---|---|---|
| cross_reference (14) | 0.679 | 0.571 |
| definitional (10) | 0.700 | 0.700 |
| jurisdictional (14) | 0.786 | 0.786 |
| numeric_threshold (10) | 0.900 | 0.900 |
| provision_lookup (15) | 1.000 | 1.000 |
| situational (27) | 0.926 | 0.852 |

| answers | C-LS | C-full | Δ |
|---|---|---|---|
| fact_coverage | 0.733 | 0.739 | -0.006 [-0.072, +0.056] |
| citation_recall | 0.806 | 0.789 | +0.017 [-0.050, +0.083] |
| citation_precision | 0.710 | 0.685 | +0.025 [-0.032, +0.083] |

## dev_holdout (n = 45 answerable)

| budget | recall C-LS | recall C-full | Δ recall | Δ MRR |
|---|---|---|---|---|
| 2000 | 0.689 | 0.644 | +0.044 [-0.056, +0.144] | +0.020 [-0.081, +0.120] |
| 8000 | 0.789 | 0.722 | **+0.067** [+0.011, +0.144] | +0.040 [-0.044, +0.128] |
| 32000 | 0.867 | 0.867 | +0.000 [+0.000, +0.000] | +0.032 [-0.047, +0.115] |

| category @8k | C-LS | C-full |
|---|---|---|
| cross_reference (7) | 0.929 | 0.786 |
| definitional (5) | 1.000 | 0.800 |
| jurisdictional (7) | 0.714 | 0.714 |
| numeric_threshold (5) | 0.600 | 0.600 |
| provision_lookup (8) | 1.000 | 1.000 |
| situational (13) | 0.615 | 0.538 |

| answers | C-LS | C-full | Δ |
|---|---|---|---|
| fact_coverage | 0.793 | 0.744 | +0.048 [-0.030, +0.126] |
| citation_recall | 0.789 | 0.778 | +0.011 [-0.022, +0.044] |
| citation_precision | 0.731 | 0.742 | -0.011 [-0.067, +0.044] |

## test (n = 317 answerable)

| budget | recall C-LS | recall C-full | Δ recall | Δ MRR |
|---|---|---|---|---|
| 2000 | 0.615 | 0.563 | **+0.052** [+0.009, +0.093] | **+0.053** [+0.011, +0.094] |
| 8000 | 0.740 | 0.716 | +0.024 [-0.005, +0.054] | **+0.057** [+0.021, +0.093] |
| 32000 | 0.819 | 0.815 | +0.003 [-0.006, +0.013] | **+0.059** [+0.025, +0.092] |

| category @8k | C-LS | C-full |
|---|---|---|
| cross_reference (51) | 0.657 | 0.627 |
| definitional (34) | 0.735 | 0.765 |
| jurisdictional (50) | 0.820 | 0.820 |
| numeric_threshold (36) | 0.917 | 0.889 |
| provision_lookup (53) | 0.943 | 0.943 |
| situational (93) | 0.559 | 0.495 |

| answers | C-LS | C-full | Δ |
|---|---|---|---|
| fact_coverage | 0.694 | 0.722 | -0.028 [-0.058, +0.002] |
| citation_recall | 0.678 | 0.655 | +0.024 [-0.011, +0.058] |
| citation_precision | 0.634 | 0.597 | **+0.037** [+0.002, +0.071] |
