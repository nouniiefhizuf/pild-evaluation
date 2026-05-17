"""Unit tests for PDE metric."""

import numpy as np
import pytest
from evaluation.pde_metric import compute_pde, compute_trajectory_divergence


def test_perfect_match():
    """PDE should be 0 when trajectories match exactly."""
    def ground_truth(t, y0, params):
        return np.column_stack([np.sin(t), np.cos(t)])

    code = """
import numpy as np
def simulate(t_eval, y0, params):
    return np.column_stack([np.sin(t_eval), np.cos(t_eval)])
"""

    pde = compute_pde(code, ground_truth, (0, 2*np.pi), 
                     np.array([0.0, 1.0]), {})
    assert pde < 0.01  # Near zero


def test_max_error():
    """PDE should be high when trajectories diverge."""
    def ground_truth(t, y0, params):
        return np.column_stack([np.zeros_like(t), np.zeros_like(t)])

    code = """
import numpy as np
def simulate(t_eval, y0, params):
    return np.column_stack([np.ones(len(t_eval)), np.ones(len(t_eval))])
"""

    pde = compute_pde(code, ground_truth, (0, 1), 
                     np.array([0.0, 0.0]), {})
    assert pde > 0.5  # High error


def test_syntax_error():
    """PDE should be 1.0 for non-executable code."""
    def ground_truth(t, y0, params):
        return np.zeros((len(t), 2))

    code = "this is not valid python!!!"

    pde = compute_pde(code, ground_truth, (0, 1), 
                     np.array([0.0, 0.0]), {})
    assert pde == 1.0


def test_trajectory_divergence_stats():
    """Test detailed divergence statistics."""
    gen = np.array([[0, 0], [1, 1], [2, 2]])
    true = np.array([[0, 0], [1, 2], [2, 4]])

    stats = compute_trajectory_divergence(gen, true)

    assert stats["mean_l2"] > 0
    assert stats["max_l2"] >= stats["mean_l2"]
    assert stats["rmse"] > stats["mae"]  # RMSE >= MAE
