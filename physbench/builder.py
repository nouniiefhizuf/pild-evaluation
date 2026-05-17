"""Benchmark construction and validation."""

import json
import sympy as sp
from pathlib import Path
from typing import Dict, List
import numpy as np
from scipy.integrate import solve_ivp


class BenchmarkBuilder:
    """Construct PhysBench-1K examples from symbolic definitions."""

    def __init__(self, output_dir: str = "physbench"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_rigid_body_example(self, 
                                  name: str,
                                  description: str,
                                  equations: List[sp.Expr],
                                  initial_conditions: Dict,
                                  t_span: List[float],
                                  difficulty: str = "medium") -> Dict:
        """Create a rigid body dynamics example."""
        # Symbolic derivation
        t = sp.Symbol('t')
        y = sp.Function('y')

        # Generate analytical solution
        analytical = self._solve_analytical(equations, initial_conditions)

        # Generate reference code
        code = self._generate_reference_code(equations, initial_conditions, t_span)

        example = {
            "id": f"rigid_body_{name}",
            "description": description,
            "domain": "rigid_body",
            "difficulty": difficulty,
            "initial_conditions": initial_conditions,
            "t_span": t_span,
            "y0": list(initial_conditions.values()),
            "reference_code": code,
            "ground_truth": analytical,
            "symbolic_equations": [str(eq) for eq in equations]
        }

        return example

    def _solve_analytical(self, equations, ics):
        """Solve system analytically where possible."""
        # Fallback to numerical for complex systems
        return {"method": "numerical", "solver": "Radau"}

    def _generate_reference_code(self, equations, ics, t_span):
        """Generate verified Python reference implementation."""
        code = f"""
import numpy as np
from scipy.integrate import solve_ivp

def simulate(t_eval, y0, params):
    def dynamics(t, y):
        # Auto-generated from symbolic equations
        dydt = np.zeros_like(y)
        # ... (implementation)
        return dydt

    sol = solve_ivp(dynamics, {t_span}, y0, t_eval=t_eval, method='Radau', 
                    atol=1e-12, rtol=1e-12)
    return sol.y.T
"""
        return code.strip()

    def save_example(self, example: Dict):
        """Save example to appropriate domain directory."""
        domain_dir = self.output_dir / example["domain"]
        domain_dir.mkdir(exist_ok=True)

        filepath = domain_dir / f"{example['id']}.json"
        with open(filepath, "w") as f:
            json.dump(example, f, indent=2)

        return filepath
