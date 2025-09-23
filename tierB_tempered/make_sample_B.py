# tierB_tempered/make_sample_B.py
# Draw lifetimes from the fitted 2-exp mixture and write output/sample_B.csv

from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"

def draw_two_exp(n: int, eta: float, tau_fast: float, tau_slow: float, seed: int = 20250701) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.random(n)
    t = np.empty(n, dtype=float)
    slow = (z < eta)                      # fraction η → slow component
    t[slow] = rng.exponential(tau_slow, slow.sum())
    t[~slow] = rng.exponential(tau_fast, (~slow).sum())
    return t

def main():
    f2 = pd.read_csv(OUT / "fit_2exp.csv")   # expects columns: eta, tau_fast, tau_slow
    eta = float(f2["eta"])
    tau_fast = float(f2["tau_fast"])         # seconds
    tau_slow = float(f2["tau_slow"])         # seconds

    n = int(os.environ.get("TIERB_N", "20000"))
    t = draw_two_exp(n, eta, tau_fast, tau_slow)
    df = pd.DataFrame({"t": t})
    df.to_csv(OUT / "sample_B.csv", index=False)
    print(f"Wrote {OUT/'sample_B.csv'} with N={n} "
          f"(η={eta:.2f}, τ_fast={tau_fast*1e3:.1f} ms, τ_slow={tau_slow*1e3:.1f} ms)")

if __name__ == "__main__":
    main()
