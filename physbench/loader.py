"""Dataset loading utilities for PhysBench-1K."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Callable
import numpy as np

DATASET_DIR = Path(__file__).parent


def load_dataset(domain: Optional[str] = None, 
                 difficulty: Optional[str] = None,
                 num_examples: Optional[int] = None,
                 seed: int = 42) -> List[Dict]:
    """Load PhysBench-1K examples with optional filtering.

    Args:
        domain: Filter by domain ('rigid_body', 'fluid_dynamics', 'multi_object')
        difficulty: Filter by difficulty ('easy', 'medium', 'hard')
        num_examples: Subsample to N examples (balanced across domains)
        seed: Random seed for subsampling

    Returns:
        List of example dictionaries
    """
    examples = []
    domains = [domain] if domain else ["rigid_body", "fluid_dynamics", "multi_object"]

    for dom in domains:
        domain_dir = DATASET_DIR / dom
        if not domain_dir.exists():
            continue

        for json_file in sorted(domain_dir.glob("*.json")):
            with open(json_file, "r") as f:
                ex = json.load(f)

            if difficulty and ex.get("difficulty") != difficulty:
                continue

            examples.append(ex)

    if num_examples and len(examples) > num_examples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(examples), size=num_examples, replace=False)
        examples = [examples[i] for i in indices]

    return examples


def load_example(domain: str, example_id: str) -> Dict:
    """Load a single example by ID."""
    filepath = DATASET_DIR / domain / f"{example_id}.json"
    with open(filepath, "r") as f:
        return json.load(f)


def list_domains() -> List[str]:
    """Return available domains."""
    return ["rigid_body", "fluid_dynamics", "multi_object"]


def list_difficulties() -> List[str]:
    """Return available difficulty levels."""
    return ["easy", "medium", "hard"]


class PhysicsExample:
    """Object-oriented interface for a single example."""

    def __init__(self, data: Dict):
        self.id = data["id"]
        self.description = data["description"]
        self.domain = data["domain"]
        self.difficulty = data["difficulty"]
        self.initial_conditions = data["initial_conditions"]
        self.t_span = tuple(data["t_span"])
        self.y0 = np.array(data["y0"])
        self.reference_code = data["reference_code"]
        self.ground_truth = data.get("ground_truth", {})

    def get_ground_truth_fn(self) -> Callable:
        """Compile and return the analytical ground truth function."""
        local_ns = {"np": np, "scipy": __import__("scipy")}
        exec(self.reference_code, local_ns)
        return local_ns.get("simulate", local_ns.get("ground_truth"))
