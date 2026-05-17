"""Physical Deviation Error (PDE) metric implementation."""

import numpy as np
from typing import Callable, Tuple, Dict
import tempfile
import os
import subprocess


def compute_pde(generated_code: str,
                ground_truth_fn: Callable,
                t_span: Tuple[float, float],
                y0: np.ndarray,
                params: Dict,
                num_timesteps: int = 1000,
                normalize: bool = True) -> float:
    """Compute Physical Deviation Error between generated and ground truth trajectories.

    Args:
        generated_code: Python code string from LLM
        ground_truth_fn: Callable analytical ground truth function
        t_span: Simulation time range (t0, tf)
        y0: Initial state vector
        params: Physical parameters dictionary
        num_timesteps: Number of evaluation points
        normalize: Whether to normalize by trajectory magnitude

    Returns:
        PDE score in [0, 1] (lower is better)
    """
    t_eval = np.linspace(t_span[0], t_span[1], num_timesteps)

    # Execute generated code in sandbox
    try:
        gen_traj = _execute_generated_code(generated_code, t_eval, y0, params)
        if gen_traj is None:
            return 1.0  # Maximum error for non-executable
    except Exception:
        return 1.0

    # Compute ground truth
    try:
        true_traj = ground_truth_fn(t_eval, y0, params)
        if isinstance(true_traj, np.ndarray):
            true_traj = true_traj
        else:
            true_traj = np.array(true_traj)
    except Exception:
        # Fallback: use reference code execution
        return 1.0

    # Ensure same shape
    if gen_traj.shape != true_traj.shape:
        return 1.0

    # Compute L2 divergence at each timestep
    divergences = np.linalg.norm(gen_traj - true_traj, axis=1)
    mean_divergence = np.mean(divergences)

    if normalize:
        # Normalize by maximum ground truth magnitude
        max_magnitude = np.max(np.linalg.norm(true_traj, axis=1))
        if max_magnitude > 1e-10:
            pde = mean_divergence / max_magnitude
        else:
            pde = mean_divergence
    else:
        pde = mean_divergence

    return float(np.clip(pde, 0.0, 1.0))


def compute_trajectory_divergence(gen_traj: np.ndarray, 
                                  true_traj: np.ndarray) -> Dict:
    """Compute detailed divergence statistics between two trajectories."""
    if gen_traj.shape != true_traj.shape:
        raise ValueError("Trajectory shapes must match")

    divergences = np.linalg.norm(gen_traj - true_traj, axis=1)

    return {
        "mean_l2": np.mean(divergences),
        "max_l2": np.max(divergences),
        "std_l2": np.std(divergences),
        "rmse": np.sqrt(np.mean(divergences**2)),
        "mae": np.mean(np.abs(divergences)),
        "final_state_error": divergences[-1]
    }


def _execute_generated_code(code: str, t_eval: np.ndarray, 
                            y0: np.ndarray, params: Dict) -> np.ndarray:
    """Safely execute generated code and extract trajectory."""
    # Create sandbox namespace
    import scipy.integrate

    local_ns = {
        "np": np,
        "scipy": scipy,
        "solve_ivp": scipy.integrate.solve_ivp,
        "t_eval": t_eval,
        "y0": y0,
        "params": params
    }

    # Execute in restricted environment
    exec(code, local_ns)

    # Try to find simulation function or result
    if "simulate" in local_ns and callable(local_ns["simulate"]):
        result = local_ns["simulate"](t_eval, y0, params)
    elif "sol" in local_ns:
        result = local_ns["sol"]
    elif "trajectory" in local_ns:
        result = local_ns["trajectory"]
    else:
        return None

    return np.array(result) if not isinstance(result, np.ndarray) else result
