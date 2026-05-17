#!/usr/bin/env python3
"""Reproduce all paper results from pre-computed outputs."""

import argparse
import subprocess
from pathlib import Path
import sys


def run_notebook(notebook_path: Path, output_dir: Path):
    """Execute a Jupyter notebook and save outputs."""
    cmd = [
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--output", str(output_dir / notebook_path.name),
        str(notebook_path)
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Executed {notebook_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Reproduce all paper results")
    parser.add_argument("--results_dir", default="results", help="Directory with raw results")
    parser.add_argument("--output_dir", default="analysis/figures", help="Output directory for figures")
    parser.add_argument("--skip_notebooks", action="store_true", help="Skip notebook execution")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PhysBench-1K: Full Paper Reproduction Pipeline")
    print("=" * 60)

    # Check results exist
    if not results_dir.exists():
        print(f"ERROR: Results directory {results_dir} not found.")
        print("Run: python scripts/download_results.py")
        sys.exit(1)

    # Run all analysis notebooks
    if not args.skip_notebooks:
        notebook_dir = Path("analysis/notebooks")
        notebooks = sorted(notebook_dir.glob("*.ipynb"))

        print(f"\nFound {len(notebooks)} analysis notebooks")

        for nb in notebooks:
            print(f"\nRunning {nb.name}...")
            try:
                run_notebook(nb, output_dir.parent / "notebooks")
            except Exception as e:
                print(f"ERROR in {nb.name}: {e}")
                continue

    # Generate summary report
    print("\n" + "=" * 60)
    print("Reproduction Complete!")
    print("=" * 60)
    print(f"Figures saved to: {output_dir}")
    print(f"Notebooks saved to: {output_dir.parent / 'notebooks'}")
    print("\nGenerated artifacts:")
    for fig in output_dir.glob("*.png"):
        print(f"  - {fig.name}")


if __name__ == "__main__":
    main()
