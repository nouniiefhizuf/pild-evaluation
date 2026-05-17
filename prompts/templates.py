"""Ten graded prompt templates (L1-L10) for physics simulation."""

from pathlib import Path
from typing import Dict, List
import re

PROMPT_DIR = Path(__file__).parent


class PromptTemplate:
    """A single prompt template with variable interpolation."""

    def __init__(self, level: str, name: str, template: str, 
                 clarity: float, naturalness: float, physics_relevance: float):
        self.level = level
        self.name = name
        self.template = template
        self.clarity = clarity
        self.naturalness = naturalness
        self.physics_relevance = physics_relevance

    def fill(self, **kwargs) -> str:
        """Fill template variables."""
        result = self.template
        for key, value in kwargs.items():
            placeholder = f"{{{key}}}"
            if isinstance(value, dict):
                value = json.dumps(value, indent=2)
            result = result.replace(placeholder, str(value))
        return result

    def __repr__(self):
        return f"PromptTemplate({self.level}: {self.name})"


# Define all 10 prompt levels
PROMPTS = {
    "L1": PromptTemplate(
        "L1", "Basic Zero-Shot",
        """Write Python code to simulate the following physics problem.
Return only the complete, runnable Python script.

Problem: {physics_problem_description}

Requirements:
- Use NumPy and SciPy
- Include initial conditions: {initial_conditions}
- Simulate for T={simulation_time}s with dt={time_step}s
- Plot the trajectory
""",
        4.2, 4.0, 2.1
    ),

    "L2": PromptTemplate(
        "L2", "Zero-Shot Chain-of-Thought",
        """Write Python code to simulate the following physics problem.

Problem: {physics_problem_description}

First, explain your physical reasoning step-by-step:
1. What physical principles apply?
2. What are the governing equations?
3. What numerical method is appropriate?

Then, return the complete Python implementation.
""",
        4.5, 4.3, 2.8
    ),

    "L3": PromptTemplate(
        "L3", "Role: Expert",
        """You are a professor of classical mechanics with 30 years of experience.

Problem: {physics_problem_description}

Derive the equations of motion using the Euler-Lagrange formalism:
L = T - V

Show your derivation, then implement the simulation in Python using symplectic integrators.
""",
        4.3, 4.1, 3.5
    ),

    "L4": PromptTemplate(
        "L4", "Few-Shot",
        """Here are two example physics simulations:

Example 1 (Projectile Motion):
{example_1_code}

Example 2 (Simple Pendulum):
{example_2_code}

Now write Python code for:
Problem: {physics_problem_description}

Follow the same structure and style as the examples.
""",
        4.7, 4.5, 3.2
    ),

    "L5": PromptTemplate(
        "L5", "Structured CoT",
        """Solve this physics simulation problem using the following structured approach:

PHYSICAL ANALYSIS:
1. Identify all forces and conserved quantities
2. Write the governing differential equations
3. Identify boundary conditions

NUMERICAL METHOD:
4. Choose integration scheme (Euler, RK4, Verlet)
5. Determine stability constraints (CFL condition)

IMPLEMENTATION:
6. Write Python code with type hints
7. Add validation checks for physical invariants
8. Include visualization

Problem: {physics_problem_description}
""",
        4.6, 4.4, 3.8
    ),

    "L6": PromptTemplate(
        "L6", "Self-Consistency",
        """Generate THREE different approaches to simulate:
Problem: {physics_problem_description}

Approach A: Use Lagrangian mechanics
Approach B: Use Newtonian force balance
Approach C: Use energy conservation principles

Compare the physical assumptions of each approach.
Select the most accurate and implement it in Python.
Explain your selection criteria.
""",
        4.4, 4.2, 3.6
    ),

    "L7": PromptTemplate(
        "L7", "Analogical",
        """This problem is analogous to a well-known physical system.

Problem: {physics_problem_description}

Analogy: {provided_analogy} (e.g., "This is like a mass on a spring")

Use the known solution structure from the analogy to derive your implementation.
Map each variable from the analogy to the target problem.
""",
        4.1, 3.9, 3.0
    ),

    "L8": PromptTemplate(
        "L8", "Constraint-Explicit",
        """Simulate the following problem with MANDATORY physical constraints:

Problem: {physics_problem_description}

CONSTRAINTS (must be verified in code):
1. Total energy conservation within 1% tolerance
2. Linear momentum conservation for isolated system
3. Angular momentum conservation (if applicable)
4. No rigid body penetration (collision detection)
5. CFL stability condition: dt < dx/v_max

Include assert statements or checks for each constraint.
If any constraint is violated, raise an exception.
""",
        4.8, 4.6, 4.5
    ),

    "L9": PromptTemplate(
        "L9", "Error-Correction",
        """Common errors when simulating this type of problem:

Problem: {physics_problem_description}

Known Error 1: Using explicit Euler for energy-conserving systems (causes drift)
Known Error 2: Ignoring Coriolis force in rotating reference frames
Known Error 3: Incorrect collision restitution coefficient application

Write Python code that explicitly AVOIDS these errors.
Include comments explaining how each error is prevented.
""",
        4.5, 4.3, 4.2
    ),

    "L10": PromptTemplate(
        "L10", "Physics-Informed",
        """Using the Euler-Lagrange formalism, derive the equations of motion.

Problem: {physics_problem_description}

STEP 1 - LAGRANGIAN:
Define T (kinetic energy) and V (potential energy)
L = T - V

STEP 2 - EULER-LAGRANGE:
d/dt(dL/dq_dot) - dL/dq = 0

STEP 3 - DISCRETIZATION:
Use symplectic integrator (Verlet or leapfrog):
p_{{n+1/2}} = p_n - dt/2 * dV/dq(q_n)
q_{{n+1}} = q_n + dt * p_{{n+1/2}}/m
p_{{n+1}} = p_{{n+1/2}} - dt/2 * dV/dq(q_{{n+1}})

STEP 4 - IMPLEMENTATION:
Write Python code with real-time energy monitoring.
Verify energy oscillation is bounded (symplectic property).
""",
        4.9, 4.7, 4.9
    ),
}


def load_prompt(level: str) -> PromptTemplate:
    """Load a prompt template by level (L1-L10)."""
    if level not in PROMPTS:
        raise ValueError(f"Unknown prompt level: {level}. Choose from {list(PROMPTS.keys())}")
    return PROMPTS[level]


def list_prompts() -> List[Dict]:
    """List all available prompts with metadata."""
    return [
        {
            "level": p.level,
            "name": p.name,
            "clarity": p.clarity,
            "naturalness": p.naturalness,
            "physics_relevance": p.physics_relevance
        }
        for p in PROMPTS.values()
    ]
