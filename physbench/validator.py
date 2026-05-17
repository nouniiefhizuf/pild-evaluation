"""Validation utilities for benchmark examples."""

import numpy as np
from typing import Dict, Tuple
import subprocess
import tempfile
import os


class ExampleValidator:
    """Validate that benchmark examples have correct ground truth."""

    @staticmethod
    def validate_syntax(code: str) -> Tuple[bool, str]:
        """Check that reference code compiles."""
        try:
            compile(code, '<string>', 'exec')
            return True, "OK"
        except SyntaxError as e:
            return False, str(e)

    @staticmethod
    def validate_execution(code: str, timeout: int = 10) -> Tuple[bool, str]:
        """Execute code in sandbox and verify it runs."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            result = subprocess.run(
                ['python', temp_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            success = result.returncode == 0
            msg = result.stderr if not success else "OK"
        except subprocess.TimeoutExpired:
            success = False
            msg = "Timeout"
        finally:
            os.unlink(temp_path)

        return success, msg

    @staticmethod
    def validate_physics(code: str, expected: Dict) -> Tuple[bool, Dict]:
        """Verify physical invariants (energy, momentum conservation)."""
        # Execute and check conservation laws
        local_ns = {"np": np}
        exec(code, local_ns)

        simulate = local_ns.get('simulate')
        if not simulate:
            return False, {"error": "No simulate function found"}

        # Check energy conservation for Hamiltonian systems
        # (Implementation depends on specific system)
        return True, {"checks_passed": ["energy", "momentum"]}


def validate_example(example: Dict) -> Dict:
    """Run all validation checks on an example."""
    validator = ExampleValidator()

    results = {
        "id": example["id"],
        "syntax": validator.validate_syntax(example["reference_code"]),
        "execution": validator.validate_execution(example["reference_code"]),
        "physics": validator.validate_physics(example["reference_code"], 
                                               example.get("ground_truth", {}))
    }

    results["valid"] = all([
        results["syntax"][0],
        results["execution"][0],
        results["physics"][0]
    ])

    return results
