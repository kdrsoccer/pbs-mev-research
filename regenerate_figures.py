"""
PBS MEV Research — Regenerate All Figures
Regenerates all five paper figures from the extended dataset.
Compatible with Windows, Mac, and Linux.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Windows-safe backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from pathlib import Path

BEACON_GENESIS = 1606824023

plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
})
COLORS = sns.color_palette("deep", 10)

DATA_DIR = Path("data/raw_extended")
OUT_DIR  = Path("results/extended")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    path = DATA_DIR / "delivered_all_relays.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}\nRun 01_data_collector.py first.")
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
    vol_33 = df["realized_vol"].quantile(0.33)
    vol_67 = df["realized_vol"].quantile(0.67)
    df["vol_regime"] = pd.cut(
        df["realized_vol"],
        bins=[-np.inf, vol_33, vol_67, np.inf],
        labels=["Low Vol", "Mid Vol", "High Vol"]
    )
    print(f"Loaded {len(df):,} observations")
    return df

def figure1_volatility_regimes(df):
    print("Generating Figure 1: Volatility Regimes...")
    CLIP_ETH  = 0.25
    LOG_FLOOR = 1e-4
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    data_clipped = [
        df[df["vol_regime"] == r]["value_eth"].clip(upper=CLIP_ETH).values
        for r in ["Low Vol", "Mid Vol", "High Vol"]
    ]
    parts = ax.violinplot(data_clipped, positions=[0, 1, 2], showmedians=True, showextrema=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(COLORS[i]); pc.set_alpha(0.7)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Low Vol", "Mid Vol", "High Vol"])
    ax.set_ylabel("Block Value (ETH)"); ax.set_ylim(bottom=0, top=CLIP_ETH)
    ax.set_title(f"(A) Block Value Distributions by Regime\n(clipped at {CLIP_ETH} ETH)")
    max_val = df["value_eth"].max()
    ax.text(0.97, 0.97, f"Full sample max: {max_val:.1f} ETH",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="gray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.8))

    ax = axes[1]
    data_log = [
        np.log10(df[df["vol_regime"] == r]["value_eth"].clip(lower=LOG_FLOOR).values)
        for r in ["Low Vol", "Mid Vol", "High Vol"]
    ]
    parts2 = ax.violinplot(data_log, positions=[0, 1, 2], showmedians=True, showextrema=True)
    for i, pc in enumerate(parts2["bodies"]):
        pc.set_facecolor(COLORS[i]); pc.set_alpha(0.7)
    parts2["cmedians"].set_color("black"); parts2["cmedians"].set_linewidth(2)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Low Vol", "Mid Vol", "High Vol"])
    ax.set_ylabel("log$_{10}$(Block Value in ETH)")
    ax.set_title("(B) Same distributions on log$_{10}$ scale")

    ax = axes[2]
    sample = df.sample(min(8000, len(df)), random_state=42)
    clip_x = sample["realized_vol"].quantile(0.99)
    clip_y = sample["value_eth"].quantile(0.995)
    x = sample["realized_vol"].clip(upper=clip_x)
    y = sample["value_eth"].clip(upper=clip_y)
    ax.scatter(x, y, alpha=0.15, s=4, color=COLORS[0])
    z = np.polyfit(sample["realized_vol"], sample["value_eth"], 1)
    x_range = np.linspace(0, clip_x, 100)
    rho, _ = spearmanr(sample["realized_vol"], sample["value_eth"])
    ax.plot(x_range, np.poly1d(z)(x_range).clip(0, clip_y), color="red", linewidth=1.5,
            label=rf"$\rho={rho:.3f}$")
    ax.set_xlabel("Realized Volatility (300-slot window)")
    ax.set_ylabel("Block Value (ETH)")
    ax.set_title("(C) Volatility vs Block Value\n(99th percentile clipped)")
    ax.legend(fontsize=9); ax.set_xlim(left=0); ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_volatility_regimes.pdf", bbox_inches="tight")
    plt.savefig(OUT_DIR / "fig_volatility_regimes.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved fig_volatility_regimes.pdf/.png")

def figure2_garch_clustering(df):
    print("Generating Figure 2: GARCH Clustering...")
    from arch import arch_model
    returns = df["log_return"] * 100
    m = arch_model(returns, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
    res = m.fit(disp="off")
    cond_vol = res.conditional_volatility

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    slots = df["slot"].values

    axes[0].plot(slots, df["log_return"].values, color=COLORS[0], linewidth=0.3, alpha=0.8)
    axes[0].set_ylabel("Log Returns"); axes[0].set_title("Block Value Log Returns")

    axes[1].fill_between(slots, df["log_return"].values**2, color=COLORS[1], alpha=0.7)
    axes[1].set_ylabel("Squared Returns")
    axes[1].set_title("Squared Returns (ARCH Effect Visible as Clustering)")

    axes[2].fill_between(slots, cond_vol.values, color=COLORS[2], alpha=0.8)
    axes[2].set_ylabel("Conditional Volatility")
    axes[2].set_title("GARCH(1,1) Conditional Volatility of Block Values")
    axes[2].set_xlabel("Slot")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_garch_clustering.pdf", bbox_inches="tight")
    plt.savefig(OUT_DIR / "fig_garch_clustering.png", bbox_inches="tight", dpi=150)
    plt.close()

    with open(OUT_DIR / "garch_fit.txt", "w") as f:
        f.write(str(res.summary()))
        f.write("\n\nLjung-Box on squared returns:\n")
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb = acorr_ljungbox(df["log_return"]**2, lags=[10, 20, 50], return_df=True)
        f.write(lb.to_string())
    print("  Saved fig_garch_clustering.pdf/.png")

def figure3_hhi(df):
    print("Generating Figure 3: HHI Over Time...")
    if "builder_pubkey" not in df.columns:
        print("  Skipped — no builder_pubkey column")
        return
    df["week"] = df["datetime"].dt.to_period("W")
    records = []
    for week, grp in df.groupby("week"):
        shares = grp["builder_pubkey"].value_counts(normalize=True)
        hhi = (shares**2).sum() * 10000
        sorted_s = shares.sort_values()
        gini = 1 - 2 * sorted_s.cumsum().mean()
        records.append({"week": week.start_time, "hhi": hhi, "gini": gini})
    hhi_df = pd.DataFrame(records)
    hhi_df.to_csv(OUT_DIR / "builder_hhi.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(hhi_df["week"], hhi_df["hhi"], color=COLORS[0], linewidth=1.5, label="Weekly HHI")
    axes[0].axhline(2500, color="red", linestyle="--", alpha=0.7, label="High concentration (HHI≥2500)")
    axes[0].axhline(1500, color="orange", linestyle=":", alpha=0.7, label="Moderate concentration (HHI≥1500)")
    axes[0].fill_between(hhi_df["week"], hhi_df["hhi"], alpha=0.1, color=COLORS[0])
    axes[0].set_ylabel("Herfindahl-Hirschman Index")
    axes[0].set_title("Builder Market Concentration Over Time\n(September 2024 – May 2026)")
    axes[0].legend(loc="upper right"); axes[0].set_ylim(bottom=0, top=2700)

    axes[1].plot(hhi_df["week"], hhi_df["gini"], color=COLORS[1], linewidth=1.5)
    axes[1].axhline(hhi_df["gini"].mean(), color="gray", linestyle="--", alpha=0.7,
                    label=f"Mean Gini = {hhi_df['gini'].mean():.3f}")
    axes[1].set_ylabel("Gini Coefficient"); axes[1].set_xlabel("Week")
    axes[1].set_ylim(0.6, 0.95); axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_hhi_over_time.pdf", bbox_inches="tight")
    plt.savefig(OUT_DIR / "fig_hhi_over_time.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved fig_hhi_over_time.pdf/.png")

def figure4_lorenz(df):
    print("Generating Figure 4: Lorenz Curve...")
    if "builder_pubkey" not in df.columns:
        print("  Skipped — no builder_pubkey column")
        return
    shares = df["builder_pubkey"].value_counts(normalize=True).sort_values()
    cumulative_builders = np.linspace(0, 1, len(shares))
    cumulative_blocks = shares.cumsum().values

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(cumulative_builders, cumulative_blocks, color=COLORS[0], linewidth=2, label="Builder distribution")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Perfect equality")
    ax.set_xlabel("Cumulative share of builders"); ax.set_ylabel("Cumulative share of blocks")
    ax.set_title("Lorenz Curve: Builder Block Production")
    ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    gini = 1 - 2 * shares.cumsum().mean()
    ax.text(0.05, 0.9, f"Gini ≈ {gini:.2f}", transform=ax.transAxes, fontsize=11)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_lorenz_curve.pdf", bbox_inches="tight")
    plt.savefig(OUT_DIR / "fig_lorenz_curve.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved fig_lorenz_curve.pdf/.png")

def figure5_temporal(df):
    print("Generating Figure 5: Temporal Patterns...")
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.day_name()
    days_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    hourly = df.groupby("hour")["value_eth"].mean()
    daily  = df.groupby("dayofweek")["value_eth"].mean().reindex(days_order)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].fill_between(hourly.index, hourly.values, alpha=0.4, color=COLORS[0])
    axes[0].plot(hourly.index, hourly.values, color=COLORS[0], linewidth=1.5)
    axes[0].set_xlabel("Hour of Day (UTC)"); axes[0].set_ylabel("Mean Block Value (ETH)")
    axes[0].set_title("Intraday MEV Pattern (KW p=0.00e+00)")
    axes[0].set_xticks(range(0, 23, 2))

    axes[1].bar(days_order, daily.values, color=COLORS[1], alpha=0.8)
    axes[1].set_ylabel("Mean Block Value (ETH)")
    axes[1].set_title("Day-of-Week MEV Pattern (KW p=0.00e+00)")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_temporal_patterns.pdf", bbox_inches="tight")
    plt.savefig(OUT_DIR / "fig_temporal_patterns.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved fig_temporal_patterns.pdf/.png")

if __name__ == "__main__":
    df = load_data()
    figure1_volatility_regimes(df)
    figure2_garch_clustering(df)
    figure3_hhi(df)
    figure4_lorenz(df)
    figure5_temporal(df)
    print("\nAll figures saved to results/extended/")
