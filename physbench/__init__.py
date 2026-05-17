"""PhysBench-1K: Physics Simulation Benchmark Dataset."""

from .loader import load_dataset, load_example, list_domains, list_difficulties
from .builder import BenchmarkBuilder
from .validator import validate_example

__version__ = "1.0.0"
__all__ = ["load_dataset", "load_example", "list_domains", "list_difficulties", 
           "BenchmarkBuilder", "validate_example"]
