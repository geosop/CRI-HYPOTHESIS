# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 14:06:36 2025

@author: ADMIN

generate_figures.py: one-click script to run all figure-making modules in the figures/ directory.
"""
#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path
from time import perf_counter

def main():
    # Determine script and figures directory
    script_dir = Path(__file__).resolve().parent
    figures_dir = script_dir / "figures"

    # Verify figures directory exists
    if not figures_dir.exists():
        print(f"Error: Figures directory not found at {figures_dir}")
        sys.exit(1)

    # Debug: list directory contents
    print(f"Contents of {figures_dir}:")
    for item in sorted(figures_dir.iterdir()):
        print("  ", item.name)

    # Collect all figure scripts matching naming convention
    figure_scripts = sorted(figures_dir.glob("make_*_figure.py"), key=lambda p: p.name.lower())
    if not figure_scripts:
        print("No figure scripts matching 'make_*_figure.py' found in figures/ directory.")
        sys.exit(0)

    # Run each figure-generation script sequentially
    start_all = perf_counter()
    for script in figure_scripts:
        print(f"\n=== Running {script.name} ===")
        start = perf_counter()
        result = subprocess.run([sys.executable, str(script)], check=False)
        elapsed = perf_counter() - start
        if result.returncode != 0:
            print(f"Error: {script.name} exited with code {result.returncode} after {elapsed:.2f}s.")
            sys.exit(result.returncode)
        print(f"✓ Completed {script.name} in {elapsed:.2f}s")

    print(f"\nAll figures have been generated successfully in {perf_counter() - start_all:.2f}s.")

if __name__ == "__main__":
    main()
