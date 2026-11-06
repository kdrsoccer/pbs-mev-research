"""
PBS MEV Research — Main Analysis
Runs all statistical analyses on the delivered payload dataset.
Compatible with Windows, Mac, and Linux.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BEACON_GENESIS = 1606824023

def load_data(data_dir="data/raw_extended"):
    path = Path(data_dir) / "delivered_all_relays.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}\nRun 01_data_collector.py first.")
    df = pd.read_parquet(path)
    df = df.sort_values("slot").reset_index(drop=True)
    df["value_eth"] = pd.to_numeric(df["value_eth"], errors="coerce")
    df["slot"] = pd.to_numeric(df["slot"], errors="coerce")
    df["timestamp"] = BEACON_GENESIS + df["slot"] * 12
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["log_value"] = np.log1p(df["value_eth"])
    df["log_return"] = df["log_value"].diff()
    df["realized_vol"] = df["log_return"].rolling(300).std() * np.sqrt(300)
    df = df.dropna(subset=["realized_vol"])
    print(f"Loaded {len(df):,} observations")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    return df

def volatility_regime_analysis(df, out_dir):
    vol_33 = df["realized_vol"].quantile(0.33)
    vol_67 = df["realized_vol"].quantile(0.67)
    print(f"\nVol thresholds: low={vol_33:.4f}, high={vol_67:.4f}")

    df["vol_regime"] = pd.cut(
        df["realized_vol"],
        bins=[-np.inf, vol_33, vol_67, np.inf],
        labels=["Low Vol", "Mid Vol", "High Vol"]
    )

    stats_df = df.groupby("vol_regime", observed=True)["value_eth"].agg(
        mean="mean", median="median", std="std",
        p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75),
        p95=lambda x: x.quantile(0.95),
        count="count"
    ).reset_index()
    print(stats_df.to_string())
    stats_df.to_csv(out_dir / "volatility_regime_stats.csv", index=False)

    groups = [df[df["vol_regime"] == r]["value_eth"].values
              for r in ["Low Vol", "Mid Vol", "High Vol"]]
    kw_stat, kw_p = stats.kruskal(*groups)
    mw_stat, mw_p = stats.mannwhitneyu(groups[0], groups[2], alternative="less")
    rho, rho_p = stats.spearmanr(df["realized_vol"], df["value_eth"])

    results = (
        f"Volatility Regime Analysis\n"
        f"==========================\n\n"
        f"Vol thresholds: low={vol_33:.4f}, high={vol_67:.4f}\n\n"
        f"Kruskal-Wallis: H={kw_stat:.4f}, p={kw_p:.2e}\n"
        f"Mann-Whitney (low < high): U={mw_stat:.0f}, p={mw_p:.2e}\n"
        f"Spearman (vol vs value): rho={rho:.4f}, p={rho_p:.2e}\n\n"
    )
    results += stats_df.to_string()
    with open(out_dir / "volatility_regime_tests.txt", "w") as f:
        f.write(results)
    print(f"\nKW H={kw_stat:.1f}, p={kw_p:.2e}")
    print(f"Spearman rho={rho:.4f}")
    return df

def temporal_analysis(df, out_dir):
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.day_name()

    hourly = df.groupby("hour")["value_eth"].mean().reset_index()
    hourly.columns = ["hour", "mean_value_eth"]
    hourly.to_csv(out_dir / "hourly_patterns.csv", index=False)

    days_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    daily = df.groupby("dayofweek")["value_eth"].mean().reindex(days_order).reset_index()
    daily.columns = ["day", "mean_value_eth"]
    daily.to_csv(out_dir / "daily_patterns.csv", index=False)

    hours = [df[df["hour"] == h]["value_eth"].values for h in range(24)]
    kw_h, kw_h_p = stats.kruskal(*hours)
    days = [df[df["dayofweek"] == d]["value_eth"].values for d in days_order]
    kw_d, kw_d_p = stats.kruskal(*days)

    results = (
        f"Temporal Pattern Tests\n"
        f"======================\n\n"
        f"Hourly (KW): H={kw_h:.4f}, p={kw_h_p:.2e}\n"
        f"Day-of-week (KW): H={kw_d:.4f}, p={kw_d_p:.2e}\n"
    )
    with open(out_dir / "temporal_test_results.txt", "w") as f:
        f.write(results)
    print(f"\nHourly KW H={kw_h:.0f}, p={kw_h_p:.2e}")
    print(f"Day-of-week KW H={kw_d:.0f}, p={kw_d_p:.2e}")

def concentration_analysis(df, out_dir):
    df["week"] = df["datetime"].dt.to_period("W")
    records = []
    for week, group in df.groupby("week"):
        shares = group["builder_pubkey"].value_counts(normalize=True)
        hhi = (shares**2).sum() * 10000
        n = len(shares)
        sorted_shares = shares.sort_values()
        cumulative = sorted_shares.cumsum()
        lorenz_area = cumulative.mean()
        gini = 1 - 2 * lorenz_area
        records.append({"week": str(week), "hhi": hhi, "gini": gini, "n_builders": n})
    hhi_df = pd.DataFrame(records)
    hhi_df.to_csv(out_dir / "builder_hhi.csv", index=False)
    print(f"\nMean HHI: {hhi_df['hhi'].mean():.0f}")
    print(f"Mean Gini: {hhi_df['gini'].mean():.3f}")

    shares = df["builder_pubkey"].value_counts(normalize=True).reset_index()
    shares.columns = ["builder_pubkey", "share"]
    shares.to_csv(out_dir / "builder_market_share.csv", index=False)

if __name__ == "__main__":
    out_dir = Path("results/extended")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    df = volatility_regime_analysis(df, out_dir)
    temporal_analysis(df, out_dir)

    if "builder_pubkey" in df.columns:
        concentration_analysis(df, out_dir)
    else:
        print("\nNo builder_pubkey column found — skipping concentration analysis")

    print("\nAnalysis complete. Results saved to results/extended/")
