"""Statistical testing utilities."""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import pandas as pd


def paired_ttest(x: np.ndarray, y: np.ndarray, 
                 alpha: float = 0.05) -> Dict:
    """Perform paired t-test with effect size.

    Args:
        x, y: Paired observations
        alpha: Significance level

    Returns:
        Dictionary with t-statistic, p-value, effect size, and interpretation
    """
    # Check normality
    _, p_norm_x = stats.shapiro(x[:min(5000, len(x))])
    _, p_norm_y = stats.shapiro(y[:min(5000, len(y))])

    use_parametric = p_norm_x > 0.05 and p_norm_y > 0.05

    if use_parametric:
        t_stat, p_val = stats.ttest_rel(x, y)
        test_name = "Paired t-test"
    else:
        t_stat, p_val = stats.wilcoxon(x, y)
        test_name = "Wilcoxon signed-rank"

    # Effect size (Cohen's d for paired samples)
    d = cohens_d(x, y)

    # Interpretation
    if abs(d) < 0.2:
        effect = "Negligible"
    elif abs(d) < 0.5:
        effect = "Small"
    elif abs(d) < 0.8:
        effect = "Medium"
    else:
        effect = "Large"

    return {
        "test": test_name,
        "t_statistic": t_stat,
        "p_value": p_val,
        "significant": p_val < alpha,
        "cohens_d": d,
        "effect_size": effect,
        "alpha": alpha
    }


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size for paired samples."""
    diff = x - y
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    if std_diff == 0:
        return 0.0

    return mean_diff / std_diff


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values
        alpha: Family-wise error rate

    Returns:
        List of booleans indicating significance after correction
    """
    m = len(p_values)
    corrected_alpha = alpha / m

    return [p < corrected_alpha for p in p_values]


def generate_significance_heatmap(results_df: pd.DataFrame,
                                  metric: str = "PDE") -> np.ndarray:
    """Generate pairwise significance matrix for heatmap visualization.

    Args:
        results_df: DataFrame with columns [model, example_id, metric_value]
        metric: Column name of metric to compare

    Returns:
        Matrix of -log10(p-values)
    """
    models = results_df['model'].unique()
    n_models = len(models)

    heatmap = np.zeros((n_models, n_models))

    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                heatmap[i, j] = 0
            else:
                x = results_df[results_df['model'] == m1][metric].values
                y = results_df[results_df['model'] == m2][metric].values

                _, p_val = stats.ttest_rel(x, y)
                heatmap[i, j] = -np.log10(p_val + 1e-300)

    return heatmap


def compute_confidence_interval(data: np.ndarray, 
                                confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval for mean."""
    mean = np.mean(data)
    sem = stats.sem(data)
    interval = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)
    return interval
