"""Prompt validation survey analysis."""

import numpy as np
from typing import Dict, List
import json


class PromptSurvey:
    """Analyze subjective prompt quality ratings."""

    def __init__(self, ratings_file: str = None):
        """Load survey ratings from JSON file or use default."""
        if ratings_file:
            with open(ratings_file, 'r') as f:
                self.ratings = json.load(f)
        else:
            # Default ratings from paper (N=12)
            self.ratings = self._default_ratings()

    def _default_ratings(self) -> Dict:
        """Default survey results from paper."""
        return {
            "participants": 12,
            "demographics": {
                "physics_students": 6,
                "cs_students": 6,
                "native_english_speakers": 12,
                "age_range": "22-28"
            },
            "cronbach_alpha": {
                "clarity": 0.87,
                "naturalness": 0.84,
                "physics_relevance": 0.91
            },
            "ratings_by_level": {
                "L1": {"clarity": 4.2, "naturalness": 4.0, "physics_relevance": 2.1, "std": 0.4},
                "L2": {"clarity": 4.5, "naturalness": 4.3, "physics_relevance": 2.8, "std": 0.3},
                "L3": {"clarity": 4.3, "naturalness": 4.1, "physics_relevance": 3.5, "std": 0.5},
                "L4": {"clarity": 4.7, "naturalness": 4.5, "physics_relevance": 3.2, "std": 0.3},
                "L5": {"clarity": 4.6, "naturalness": 4.4, "physics_relevance": 3.8, "std": 0.4},
                "L6": {"clarity": 4.4, "naturalness": 4.2, "physics_relevance": 3.6, "std": 0.5},
                "L7": {"clarity": 4.1, "naturalness": 3.9, "physics_relevance": 3.0, "std": 0.6},
                "L8": {"clarity": 4.8, "naturalness": 4.6, "physics_relevance": 4.5, "std": 0.2},
                "L9": {"clarity": 4.5, "naturalness": 4.3, "physics_relevance": 4.2, "std": 0.3},
                "L10": {"clarity": 4.9, "naturalness": 4.7, "physics_relevance": 4.9, "std": 0.2},
            }
        }

    def get_correlation_with_pde(self, pde_by_level: Dict[str, float]) -> Dict:
        """Compute correlation between survey ratings and observed PDE."""
        levels = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10"]

        physics_rel = [self.ratings["ratings_by_level"][l]["physics_relevance"] for l in levels]
        pde_vals = [pde_by_level[l] for l in levels]

        corr = np.corrcoef(physics_rel, pde_vals)[0, 1]

        return {
            "correlation": corr,
            "interpretation": "Strong negative" if corr < -0.7 else 
                            "Moderate negative" if corr < -0.4 else "Weak"
        }

    def generate_latex_table(self) -> str:
        """Generate LaTeX table for paper."""
        lines = [
            "\begin{table}[htbp]",
            "\centering",
            "\caption{Prompt validation survey results (N=12, 5-point Likert scale).}",
            "\begin{tabular}{clccc}",
            "\toprule",
            "\textbf{Level} & \textbf{Prompt Type} & \textbf{Clarity} & \textbf{Naturalness} & \textbf{Physics Rel.} \\",
            "\midrule"
        ]

        for level in ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10"]:
            r = self.ratings["ratings_by_level"][level]
            lines.append(
                f"{level} & {self._level_name(level)} & {r['clarity']:.1f} ({r['std']:.1f}) & "
                f"{r['naturalness']:.1f} ({r['std']:.1f}) & {r['physics_relevance']:.1f} ({r['std']:.1f}) \\"
            )

        lines.extend(["\bottomrule", "\end{tabular}", "\end{table}"])
        return "
".join(lines)

    def _level_name(self, level: str) -> str:
        names = {
            "L1": "Basic", "L2": "Zero-Shot CoT", "L3": "Role: Expert",
            "L4": "Few-Shot", "L5": "Structured CoT", "L6": "Self-Consistency",
            "L7": "Analogical", "L8": "Constraint-Explicit", "L9": "Error-Correction",
            "L10": "Physics-Informed"
        }
        return names.get(level, level)
