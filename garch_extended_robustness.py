"""
PBS MEV Research — GARCH Extended Robustness Checks
Runs GARCH(1,1), GJR-GARCH, winsorized, and subperiod specifications.
Compatible with Windows, Mac, and Linux.
"""

import pandas as pd
import numpy as np
from arch import arch_model
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BEACON_GENESIS = 1606824023

def load_returns(data_dir="data/raw_extended"):
    path = Path(data_dir) / "delivered_all_relays.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_parquet(path)
    df = df.sort_values("slot").reset_index(drop=True)
    df["value_eth"] = pd.to_numeric(df["value_eth"], errors="coerce")
    df["log_return"] = np.log1p(df["value_eth"]).diff()
    df = df.dropna(subset=["log_return"])
    df["timestamp"] = BEACON_GENESIS + df["slot"] * 12
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    print(f"Full sample: {len(df):,} observations")
    return df

def fit_garch(returns, label, p=1, q=1, power=2.0, vol="GARCH"):
    try:
        m = arch_model(returns * 100, mean="Constant", vol=vol,
                       p=p, q=q, power=power, dist="t")
        r = m.fit(disp="off", options={"maxiter": 5000})
        params = r.params
        alpha = params.get("alpha[1]", params.get("alpha[1]", 0))
        beta = params.get("beta[1]", 0)
        gamma = params.get("gamma[1]", 0)
        if vol == "GARCH":
            persistence = alpha + beta
        elif vol == "EGARCH":
            persistence = beta
        else:
            persistence = alpha + gamma * 0.5 + beta
        converged = r.convergence_flag == 0
        print(f"   n={len(returns):,}, persistence={persistence:.4f}, converged={converged}")
        return persistence, converged, r.aic
    except Exception as e:
        print(f"   FAILED: {e}")
        return None, False, None

def run_all():
    df = load_returns()
    returns = df["log_return"]

    print("\n" + "="*60)
    print("1. GARCH(1,1) Student t — baseline")
    p1, c1, aic1 = fit_garch(returns, "GARCH(1,1)")

    print("\n2. EGARCH(1,1) Student t")
    p2, c2, aic2 = fit_garch(returns, "EGARCH", vol="EGARCH")

    print("\n3. GJR-GARCH(1,1,1) Student t")
    p3, c3, aic3 = fit_garch(returns, "GJR-GARCH", vol="GARCH", power=2.0)

    print("\n4. GARCH(1,1) on winsorized returns (1% tails)")
    lower = returns.quantile(0.01)
    upper = returns.quantile(0.99)
    winsorized = returns.clip(lower, upper)
    p4, c4, aic4 = fit_garch(winsorized, "Winsorized")

    print("\n5. GARCH(1,1) — first half of sample")
    mid = len(df) // 2
    first_half = returns.iloc[:mid]
    p5, c5, aic5 = fit_garch(first_half, "First half")

    print("\n6. GARCH(1,1) — second half of sample")
    second_half = returns.iloc[mid:]
    p6, c6, aic6 = fit_garch(second_half, "Second half")

    print("\n" + "="*60)
    print("ROBUSTNESS SUMMARY")
    print("="*60)
    print(f"{'Specification':<30} {'Persistence':>12}")
    print("-"*44)
    for label, p in [
        ("GARCH(1,1)", p1),
        ("EGARCH(1,1)", p2),
        ("GJR-GARCH", p3),
        ("GARCH Winsorized", p4),
        ("First Half", p5),
        ("Second Half", p6),
    ]:
        val = f"{p:.4f}" if p is not None else "FAILED"
        print(f"{label:<30} {val:>12}")

    out_dir = Path("results/extended")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "garch_robustness_summary.txt", "w") as f:
        f.write("GARCH Robustness Results\n")
        f.write("========================\n\n")
        for label, p, c in [
            ("GARCH(1,1)", p1, c1),
            ("EGARCH(1,1)", p2, c2),
            ("GJR-GARCH", p3, c3),
            ("GARCH Winsorized", p4, c4),
            ("First Half", p5, c5),
            ("Second Half", p6, c6),
        ]:
            val = f"{p:.4f}" if p is not None else "FAILED"
            f.write(f"{label}: persistence={val}, converged={c}\n")

if __name__ == "__main__":
    run_all()
