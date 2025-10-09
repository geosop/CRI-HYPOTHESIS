# ADMIN

# tierB_tempered/make_sample_B.py
# Draw lifetimes from the fitted 2-exp mixture (η = FAST weight)
# and write output/sample_B.csv (times in seconds).

from __future__ import annotations
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)


def draw_two_exp(
    n: int,
    eta_fast: float,
    tau_fast: float,
    tau_slow: float,
    seed: int | None = 20250701,
) -> np.ndarray:
    """
    Draw lifetimes from a two-exponential mixture where
    - eta_fast is the FAST-component weight (0<eta_fast<1),
    - tau_fast <= tau_slow are component means (seconds).
    """
    if not (0.0 < eta_fast < 1.0):
        raise ValueError("eta_fast must be in (0,1).")
    if tau_fast <= 0 or tau_slow <= 0:
        raise ValueError("tau_fast and tau_slow must be positive.")

    # Enforce ordering to avoid label confusion
    if tau_fast > tau_slow:
        tau_fast, tau_slow = tau_slow, tau_fast
        eta_fast = 1.0 - eta_fast  # keep semantics: weight of the (now) fast component

    rng = np.random.default_rng(seed)
    z = rng.random(n) < eta_fast
    t = np.empty(n, dtype=float)
    # FAST draw where z is True, SLOW otherwise
    t[z] = rng.exponential(tau_fast, z.sum())
    t[~z] = rng.exponential(tau_slow, (~z).sum())
    return t


def main() -> None:
    # Read fitted 2-exp parameters (as written by simulate_and_fit.py)
    f2_path = OUT / "fit_2exp.csv"
    if not f2_path.exists():
        raise FileNotFoundError(
            f"Missing {f2_path}. Run tierB_tempered/simulate_and_fit.py first."
        )

    f2 = pd.read_csv(f2_path)  # expects columns: eta, tau_fast, tau_slow
    # Be robust to either column layout (single-row wide or melted)
    if {"eta", "tau_fast", "tau_slow"}.issubset(set(f2.columns)):
        eta_fast = float(f2.loc[0, "eta"])
        tau_fast = float(f2.loc[0, "tau_fast"])   # seconds
        tau_slow = float(f2.loc[0, "tau_slow"])   # seconds
    else:
        # fallback: try key/value style
        try:
            eta_fast = float(f2.query("param=='eta'")["value"].iloc[0])
            tau_fast = float(f2.query("param=='tau_fast'")["value"].iloc[0])
            tau_slow = float(f2.query("param=='tau_slow'")["value"].iloc[0])
        except Exception as e:
            raise ValueError("Unrecognized fit_2exp.csv format.") from e

    # Env overrides
    n = int(os.environ.get("TIERB_N", "20000"))
    seed = int(os.environ.get("TIERB_SEED", "20250701"))

    # Draw and save
    t = draw_two_exp(n, eta_fast, tau_fast, tau_slow, seed=seed)
    df = pd.DataFrame({"t": t})
    out_csv = OUT / "sample_B.csv"
    df.to_csv(out_csv, index=False)

    # Optional meta for reproducibility
    meta = {
        "n": n,
        "seed": seed,
        "eta_fast": eta_fast,
        "tau_fast_s": tau_fast,
        "tau_slow_s": tau_slow,
        "units": "seconds",
        "note": "η is the FAST-component weight; times drawn from two-exponential mixture.",
    }
    (OUT / "sample_B_meta.json").write_text(json.dumps(meta, indent=2))

    print(
        f"Wrote {out_csv} with N={n} "
        f"(η_fast={eta_fast:.2f}, τ_fast={tau_fast*1e3:.1f} ms, τ_slow={tau_slow*1e3:.1f} ms)"
    )


if __name__ == "__main__":
    main()
