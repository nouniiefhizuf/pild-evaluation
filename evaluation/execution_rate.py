"""Code executability checking."""

import subprocess
import tempfile
import os
from typing import Tuple, Dict
import traceback


class ExecutionChecker:
    """Check if generated Python code executes without errors."""

    def __init__(self, timeout: int = 30, python_cmd: str = "python3"):
        self.timeout = timeout
        self.python_cmd = python_cmd
        self.allowed_imports = {
            "numpy", "scipy", "matplotlib", "matplotlib.pyplot",
            "math", "random", "itertools", "functools", "collections"
        }

    def check(self, code: str) -> Tuple[bool, Dict]:
        """Check if code executes successfully.

        Returns:
            (success, info_dict)
        """
        # First, check for dangerous imports
        import_check = self._check_imports(code)
        if not import_check["safe"]:
            return False, {
                "error_type": "UnsafeImport",
                "error_message": import_check["message"],
                "traceback": ""
            }

        # Write to temp file and execute
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Add standard imports if missing
            header = "import numpy as np\nimport matplotlib.pyplot as plt\n"
            if "import numpy" not in code:
                f.write(header)
            f.write(code)
            temp_path = f.name

        try:
            result = subprocess.run(
                [self.python_cmd, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            success = result.returncode == 0
            info = {
                "error_type": "None" if success else "RuntimeError",
                "error_message": result.stderr if not success else "",
                "traceback": result.stderr if not success else "",
                "stdout": result.stdout[:1000]  # Truncate long output
            }

        except subprocess.TimeoutExpired:
            success = False
            info = {
                "error_type": "Timeout",
                "error_message": f"Execution exceeded {self.timeout}s",
                "traceback": ""
            }
        except Exception as e:
            success = False
            info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
        finally:
            os.unlink(temp_path)

        return success, info

    def _check_imports(self, code: str) -> Dict:
        """Check for potentially dangerous imports."""
        import re

        # Find all import statements
        import_pattern = r"^(?:from\s+(\w+)\s+import|import\s+(\w+))"

        for line in code.split('\n'):
            match = re.match(import_pattern, line.strip())
            if match:
                module = match.group(1) or match.group(2)
                if module not in self.allowed_imports and not module.startswith("scipy."):
                    return {
                        "safe": False,
                        "message": f"Potentially unsafe import: {module}"
                    }

        return {"safe": True, "message": ""}


def check_executability(code: str, timeout: int = 30) -> bool:
    """Quick check if code runs."""
    checker = ExecutionChecker(timeout=timeout)
    success, _ = checker.check(code)
    return success
