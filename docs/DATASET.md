# PhysBench-1K Dataset Documentation

## Overview

PhysBench-1K is a curated benchmark of 1,000 physics simulation tasks designed to evaluate Large Language Models as actionable physics simulators.

## Domains

### Rigid Body Dynamics (300 examples)
- Projectile motion with/without air resistance
- Simple and double pendulum
- Elastic and inelastic collisions
- Rotational dynamics (spinning top, gyroscope)

### Fluid Dynamics (400 examples)
- Incompressible Navier-Stokes (2D)
- Bernoulli principle applications
- Reynolds number transitions
- Vortex shedding (von Karman street)

### Multi-Object Interactions (300 examples)
- Gravitational N-body problems (2D/3D)
- Coupled spring-mass systems
- Friction-dominated sliding
- Buoyancy and Archimedes principle

## Difficulty Levels

| Level | Criteria | Example Count |
|-------|----------|---------------|
| Easy | Single equation, analytic solution | 350 |
| Medium | Coupled ODEs, requires numerical integration | 450 |
| Hard | Chaotic systems, PDEs, multi-scale | 200 |

## Data Format

Each example is a JSON file with the following structure:

```json
{
  "id": "rigid_body_001",
  "description": "Natural language problem description",
  "domain": "rigid_body",
  "difficulty": "easy",
  "initial_conditions": {"param": value},
  "t_span": [0.0, 10.0],
  "y0": [1.0, 0.0],
  "reference_code": "Python implementation",
  "ground_truth": {"analytical_results": values}
}
```

## Ground Truth Generation

All ground truth trajectories are computed using:
- **Solver:** SciPy `solve_ivp` with Radau method
- **Tolerances:** Absolute 1e-12, Relative 1e-12
- **Validation:** Energy/momentum conservation checks

## Citation

```bibtex
@dataset{physbench1k2026,
  title={PhysBench-1K: A Benchmark for Physics Simulation Evaluation},
  author={Soltani, Ahmed and Chanchah, Ryan and Darghouth, Skander and Ben Rejeb, Khalil},
  year={2026},
  publisher={GitHub},
  url={https://github.com/medtech-tn/pild-evaluation}
}
```
