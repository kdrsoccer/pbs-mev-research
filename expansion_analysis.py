import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')
import datetime

OUT = '/Users/kylerothwell/pbs_research/figures/'
import os; os.makedirs(OUT, exist_ok=True)

GENESIS = datetime.datetime(2020, 12, 1, 12, 0, 23)

def slot_to_dt(slot):
    return GENESIS + datetime.timedelta(seconds=int(slot) * 12)

# ── Load data ────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet('/Users/kylerothwell/pbs_research/bid_auction_stats.parquet')
df = df.dropna(subset=['n_bids', 'value_eth'])
df['bid_margin_pct'] = df['bid_margin'] * 100
df['date'] = df['slot'].apply(lambda s: slot_to_dt(s))
df['hour'] = df['date'].dt.hour
df['week'] = df['date'].dt.to_period('W')

eth = pd.read_parquet('/Users/kylerothwell/pbs_research/eth_price_hourly.parquet')
eth['hour_floor'] = eth['datetime'].dt.floor('h')

print(f"  Delivered payloads: {len(df):,}")
print(f"  ETH price candles:  {len(eth):,}")

# ── ANALYSIS 1: ETH Price Decomposition ─────────────────────────
print("\n=== ANALYSIS 1: ETH Price Decomposition ===")

# Merge ETH price onto delivered payloads by hour
df['hour_floor'] = df['date'].dt.floor('h')
df2 = df.merge(eth[['hour_floor','eth_close']], on='hour_floor', how='left')
matched = df2['eth_close'].notna().sum()
print(f"  Matched {matched:,} slots to hourly ETH price ({matched/len(df2)*100:.1f}%)")

df2 = df2.dropna(subset=['eth_close'])

# Block value in USD
df2['value_usd'] = df2['value_eth'] * df2['eth_close']

# Log ETH price and log block value
df2['log_eth'] = np.log(df2['eth_close'])
df2['log_val'] = np.log(df2['value_eth'] + 1e-10)

# Decompose: regress log block value on log ETH price
# residual = MEV-specific component
from numpy.polynomial import polynomial as P
X = df2['log_eth'].values
Y = df2['log_val'].values
mask = np.isfinite(X) & np.isfinite(Y)
slope, intercept, r, p, se = stats.linregress(X[mask], Y[mask])
df2.loc[mask, 'mev_residual'] = Y[mask] - (slope * X[mask] + intercept)

print(f"\n  OLS: log(block_value) = {intercept:.3f} + {slope:.3f} * log(ETH_price)")
print(f"  R² = {r**2:.4f} — ETH price explains {r**2*100:.1f}% of log block value variance")
print(f"  Slope = {slope:.3f} (elasticity of block value to ETH price)")

rho1, p1 = spearmanr(df2.loc[mask,'eth_close'], df2.loc[mask,'value_eth'])
print(f"  Spearman rho (ETH price vs block value ETH): {rho1:.4f}, p={p1:.2e}")

rho2, p2 = spearmanr(df2.loc[mask,'eth_close'], df2.loc[mask,'value_usd'])
print(f"  Spearman rho (ETH price vs block value USD): {rho2:.4f}, p={p2:.2e}")

# Figure: decomposition
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(df2['eth_close'].clip(upper=5000),
                df2['value_eth'].clip(upper=2),
                alpha=0.02, s=1, color='steelblue')
axes[0].set_xlabel('ETH Price (USD)')
axes[0].set_ylabel('Block Value (ETH)')
axes[0].set_title(f'Block Value (ETH) vs ETH Price\nSpearman ρ = {rho1:.3f}')

axes[1].scatter(df2['eth_close'].clip(upper=5000),
                df2['value_usd'].clip(upper=500),
                alpha=0.02, s=1, color='darkorange')
axes[1].set_xlabel('ETH Price (USD)')
axes[1].set_ylabel('Block Value (USD)')
axes[1].set_title(f'Block Value (USD) vs ETH Price\nSpearman ρ = {rho2:.3f}')

weekly_eth = df2.groupby('week').agg(
    med_eth=('eth_close','median'),
    med_val_eth=('value_eth','median'),
    med_val_usd=('value_usd','median'),
    med_residual=('mev_residual','median')
).reset_index()
weekly_eth['week_dt'] = weekly_eth['week'].dt.to_timestamp()

ax2 = axes[2]
color1, color2 = 'steelblue', 'darkorange'
l1, = ax2.plot(weekly_eth['week_dt'], weekly_eth['med_val_eth'],
               color=color1, label='Block Value (ETH)', linewidth=1.5)
ax2.set_ylabel('Block Value (ETH)', color=color1)
ax2b = ax2.twinx()
l2, = ax2b.plot(weekly_eth['week_dt'], weekly_eth['med_eth'],
                color=color2, label='ETH Price (USD)', linewidth=1.5, linestyle='--')
ax2b.set_ylabel('ETH Price (USD)', color=color2)
ax2.set_title('Weekly Median Block Value vs ETH Price')
ax2.legend(handles=[l1, l2], loc='upper left', fontsize=8)

plt.suptitle('ETH Price and Block Value Decomposition', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'fig_eth_decomposition.pdf', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig_eth_decomposition.pdf")

# ── ANALYSIS 2: HHI-Margin Regression ───────────────────────────
print("\n=== ANALYSIS 2: HHI-Margin Regression ===")

# Load builder market share data
builder_csv = '/Users/kylerothwell/pbs_research/figures/builder_hhi.csv'
try:
    hhi_df = pd.read_csv(builder_csv)
    print(f"  Loaded HHI data: {len(hhi_df)} weeks")
    print(f"  Columns: {list(hhi_df.columns)}")
except:
    # Compute HHI from scratch
    print("  Computing HHI from scratch...")
    df_main = pd.read_parquet(
        '/Users/kylerothwell/Downloads/pbs_research/pbs_research/data/raw_extended/delivered_all_relays.parquet'
    )
    df_main['date'] = df_main['slot'].apply(lambda s: slot_to_dt(s))
    df_main['week'] = df_main['date'].dt.to_period('W')
    weekly_shares = df_main.groupby(['week','builder_pubkey']).size().reset_index(name='n')
    weekly_total  = df_main.groupby('week').size().reset_index(name='N')
    weekly_shares = weekly_shares.merge(weekly_total, on='week')
    weekly_shares['share'] = weekly_shares['n'] / weekly_shares['N']
    hhi_df = weekly_shares.groupby('week').apply(
        lambda x: (x['share']**2).sum() * 10000
    ).reset_index(name='hhi')
    hhi_df['week_dt'] = hhi_df['week'].dt.to_timestamp()
    print(f"  Computed HHI for {len(hhi_df)} weeks")

# Weekly bid margin
weekly_margin = df.groupby('week').agg(
    med_margin=('bid_margin_pct','median'),
    med_builders=('n_builders','median'),
    med_bids=('n_bids','median')
).reset_index()
weekly_margin['week_dt'] = weekly_margin['week'].dt.to_timestamp()

# Try to merge
try:
    if 'week' in hhi_df.columns:
        merged_hhi = weekly_margin.merge(hhi_df[['week','hhi']], on='week', how='inner')
    elif 'week_dt' in hhi_df.columns:
        merged_hhi = weekly_margin.merge(hhi_df[['week_dt','hhi']], on='week_dt', how='inner')
    else:
        raise ValueError("Cannot find week column in HHI data")

    print(f"  Merged: {len(merged_hhi)} weeks with both HHI and margin data")

    # Lagged regression: HHI(t-1) -> margin(t)
    merged_hhi = merged_hhi.sort_values('week_dt').reset_index(drop=True)
    merged_hhi['hhi_lag1'] = merged_hhi['hhi'].shift(1)
    clean = merged_hhi.dropna(subset=['hhi_lag1','med_margin'])

    slope, intercept, r, p, se = stats.linregress(clean['hhi_lag1'], clean['med_margin'])
    rho_hhi, p_hhi = spearmanr(clean['hhi_lag1'], clean['med_margin'])

    print(f"\n  OLS: margin(t) = {intercept:.4f} + {slope:.6f} * HHI(t-1)")
    print(f"  R² = {r**2:.4f}")
    print(f"  Spearman rho (lagged HHI vs margin): {rho_hhi:.4f}, p={p_hhi:.4f}")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(clean['hhi_lag1'], clean['med_margin'],
                    alpha=0.6, color='steelblue', s=30)
    x_line = np.linspace(clean['hhi_lag1'].min(), clean['hhi_lag1'].max(), 100)
    axes[0].plot(x_line, intercept + slope * x_line, color='red', linewidth=2)
    axes[0].set_xlabel('HHI (lagged 1 week)')
    axes[0].set_ylabel('Median Bid Margin (%)')
    axes[0].set_title(f'Lagged HHI vs Bid Margin\nρ = {rho_hhi:.3f}, p = {p_hhi:.3f}')

    ax = axes[1]
    color1, color2 = 'steelblue', 'red'
    l1, = ax.plot(merged_hhi['week_dt'], merged_hhi['hhi']/100,
                  color=color1, label='HHI/100', linewidth=1.5)
    ax.set_ylabel('HHI / 100', color=color1)
    axb = ax.twinx()
    l2, = axb.plot(merged_hhi['week_dt'], merged_hhi['med_margin'],
                   color=color2, label='Bid Margin (%)', linewidth=1.5)
    axb.set_ylabel('Median Bid Margin (%)', color=color2)
    ax.set_title('HHI and Bid Margin Over Time')
    ax.legend(handles=[l1,l2], loc='upper right', fontsize=8)

    plt.suptitle('Builder Concentration and Auction Competitiveness', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT + 'fig_hhi_margin_regression.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig_hhi_margin_regression.pdf")

except Exception as e:
    print(f"  HHI merge failed: {e} — will compute HHI inline")

# ── ANALYSIS 3: Builder-Level Bidding Behavior ───────────────────
print("\n=== ANALYSIS 3: Builder-Level Bidding Behavior ===")

# Read a sample of bid archive files to get builder-level stats
import zipfile
from pathlib import Path
from datetime import date, timedelta

BIDDIR = Path('/Users/kylerothwell/pbs_research/bidarchive/top')
valid_slots = set(df['slot'].values)

builder_records = []
sample_dates = pd.date_range('2025-01-01', '2025-03-31', freq='7D')

print(f"  Sampling {len(sample_dates)} weeks for builder-level analysis...")
for d in sample_dates:
    fname = f"{d.strftime('%Y-%m-%d')}_top.csv.zip"
    fpath = BIDDIR / fname
    if not fpath.exists():
        continue
    try:
        with zipfile.ZipFile(fpath) as zf:
            with zf.open(zf.namelist()[0]) as f:
                tmp = pd.read_csv(f, usecols=['slot','value','builder_pubkey','slot_t_ms'])
        tmp['value'] = pd.to_numeric(tmp['value'], errors='coerce')
        tmp['slot']  = pd.to_numeric(tmp['slot'],  errors='coerce')
        tmp['slot_t_ms'] = pd.to_numeric(tmp['slot_t_ms'], errors='coerce')
        tmp = tmp.dropna(subset=['value','slot'])
        tmp = tmp[tmp['slot'].isin(valid_slots)]
        builder_records.append(tmp)
    except:
        pass

if builder_records:
    bdf = pd.concat(builder_records, ignore_index=True)
    print(f"  Loaded {len(bdf):,} bid observations across {bdf['builder_pubkey'].nunique()} builders")

    # Per-builder stats
    builder_stats = bdf.groupby('builder_pubkey').agg(
        n_bids      = ('value', 'count'),
        n_slots     = ('slot', 'nunique'),
        mean_value  = ('value', 'mean'),
        median_value= ('value', 'median'),
        mean_timing = ('slot_t_ms', 'mean'),
        median_timing=('slot_t_ms','median'),
    ).reset_index()
    builder_stats = builder_stats[builder_stats['n_bids'] >= 100].copy()
    builder_stats['bids_per_slot'] = builder_stats['n_bids'] / builder_stats['n_slots']
    builder_stats['value_eth'] = builder_stats['median_value'] / 1e18
    builder_stats = builder_stats.sort_values('n_slots', ascending=False)

    print(f"\n  Top 10 builders by slot participation:")
    print(builder_stats[['builder_pubkey','n_slots','bids_per_slot',
                          'value_eth','median_timing']].head(10).to_string())

    # Figure: builder behavior scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    top = builder_stats.head(30)
    sizes = (top['n_slots'] / top['n_slots'].max() * 200).clip(lower=20)

    axes[0].scatter(top['median_timing'], top['value_eth']*1e9,
                    s=sizes, alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Median Bid Timing (ms after slot start)')
    axes[0].set_ylabel('Median Bid Value (Gwei)')
    axes[0].set_title('Builder Strategy: Timing vs Value\n(bubble size = slot participation)')

    axes[1].scatter(top['bids_per_slot'], top['value_eth']*1e9,
                    s=sizes, alpha=0.7, color='darkorange')
    axes[1].set_xlabel('Bids per Slot')
    axes[1].set_ylabel('Median Bid Value (Gwei)')
    axes[1].set_title('Builder Strategy: Bid Frequency vs Value')

    plt.suptitle('Builder-Level Bidding Strategies (Q1 2025 Sample)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT + 'fig_builder_strategies.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig_builder_strategies.pdf")

    # Save builder stats
    builder_stats.to_csv('/Users/kylerothwell/pbs_research/builder_level_stats.csv', index=False)

# ── ANALYSIS 4: Relay-Level Analysis ────────────────────────────
print("\n=== ANALYSIS 4: Relay-Level Analysis ===")

if builder_records:
    relay_stats = bdf.groupby('relay' if 'relay' in bdf.columns else 'builder_pubkey').agg(
        n_bids=('value','count'),
        median_value=('value','median'),
        median_timing=('slot_t_ms','median'),
        n_builders=('builder_pubkey','nunique'),
    ).reset_index() if 'relay' in bdf.columns else None

# Load from full bid archive for relay analysis
print("  Loading relay data from full auction stats...")
df_relay = df.dropna(subset=['n_relays'])
relay_dist = df_relay['n_relays'].value_counts().sort_index()
print(f"\n  Relay count distribution per slot:")
print(relay_dist.to_string())

# Relay count vs margin
relay_margin = df_relay.groupby('n_relays').agg(
    med_margin=('bid_margin_pct','median'),
    med_value=('value_eth','median'),
    n=('bid_margin_pct','count')
).reset_index()
print(f"\n  Bid margin by relay count:")
print(relay_margin.to_string())

rho_relay, p_relay = spearmanr(df_relay['n_relays'], df_relay['bid_margin_pct'].clip(0,100))
print(f"\n  Spearman rho (n_relays vs margin): {rho_relay:.4f}, p={p_relay:.2e}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(relay_margin['n_relays'], relay_margin['med_margin'],
            color='steelblue', alpha=0.85)
axes[0].set_xlabel('Number of Relays per Slot')
axes[0].set_ylabel('Median Bid Margin (%)')
axes[0].set_title(f'Bid Margin by Relay Count\nSpearman ρ = {rho_relay:.3f}')

axes[1].bar(relay_margin['n_relays'], relay_margin['n']/1000,
            color='darkorange', alpha=0.85)
axes[1].set_xlabel('Number of Relays per Slot')
axes[1].set_ylabel('Slot Count (thousands)')
axes[1].set_title('Slot Distribution by Relay Count')

plt.suptitle('Relay Participation and Auction Competitiveness', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'fig_relay_analysis.pdf', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig_relay_analysis.pdf")

# ── ANALYSIS 5: GARCH on Bid Margins ────────────────────────────
print("\n=== ANALYSIS 5: GARCH on Bid Margins ===")

try:
    from arch import arch_model

    # Use clean margins
    margins = df['bid_margin_pct'].copy()
    margins = margins[np.isfinite(margins) & (margins >= 0) & (margins < 100)]
    margins = margins.reset_index(drop=True)

    print(f"  Fitting GARCH(1,1) on {len(margins):,} bid margin observations...")

    # Ljung-Box on squared demeaned margins
    from statsmodels.stats.diagnostic import acorr_ljungbox
    demeaned = margins - margins.mean()
    lb = acorr_ljungbox(demeaned**2, lags=[10,20], return_df=True)
    print(f"\n  Ljung-Box on squared margins:")
    print(lb.to_string())

    # GARCH(1,1)
    gm = arch_model(margins, vol='Garch', p=1, q=1, dist='t', mean='Constant')
    res = gm.fit(disp='off')
    print(f"\n  GARCH(1,1) on bid margins:")
    print(res.summary().tables[1])
    params = res.params
    alpha = params.get('alpha[1]', params.get('alpha', np.nan))
    beta  = params.get('beta[1]',  params.get('beta',  np.nan))
    print(f"\n  Persistence (alpha+beta): {alpha+beta:.4f}")

    # Figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    idx = range(len(margins))
    axes[0].plot(idx, margins, linewidth=0.3, color='steelblue', alpha=0.8)
    axes[0].set_ylabel('Bid Margin (%)')
    axes[0].set_title('Bid Margin Series')
    axes[1].plot(idx, margins**2, linewidth=0.3, color='darkorange', alpha=0.8)
    axes[1].set_ylabel('Squared Margin')
    axes[1].set_title('Squared Bid Margins (ARCH effect check)')
    axes[2].plot(idx, res.conditional_volatility, linewidth=0.5, color='red')
    axes[2].set_ylabel('Conditional Volatility')
    axes[2].set_title(f'GARCH(1,1) Conditional Volatility (persistence={alpha+beta:.4f})')
    plt.suptitle('GARCH Analysis of Bid Margin Dynamics', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT + 'fig_margin_garch.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig_margin_garch.pdf")

except ImportError:
    print("  arch package not installed — installing...")
    import subprocess
    subprocess.run(['pip3', 'install', 'arch', '--break-system-packages', '-q'])
    print("  Installed. Re-run script to get GARCH on margins.")
except Exception as e:
    print(f"  GARCH error: {e}")

# ── ANALYSIS 6: Validator Revenue Efficiency ─────────────────────
print("\n=== ANALYSIS 6: Validator Revenue Efficiency ===")

df_clean = df[
    np.isfinite(df['bid_margin_pct']) &
    (df['bid_margin_pct'] >= 0) &
    (df['bid_margin_pct'] < 100)
].copy()

# Revenue efficiency = 1 - bid_margin (fraction of top bid that goes to proposer)
df_clean['proposer_efficiency'] = 1 - df_clean['bid_margin'] 
df_clean['vol_tercile'] = pd.qcut(df_clean['value_eth'], q=3,
                                   labels=['Low','Mid','High'], duplicates='drop')

print(f"\n  Proposer revenue efficiency by value tercile:")
eff = df_clean.groupby('vol_tercile', observed=True)['proposer_efficiency'].agg(
    ['mean','median']
) * 100
print(eff.to_string())

# By hour of day
hourly_eff = df_clean.groupby('hour')['proposer_efficiency'].median() * 100
print(f"\n  Lowest efficiency hours: {hourly_eff.nsmallest(3).to_string()}")
print(f"  Highest efficiency hours: {hourly_eff.nlargest(3).to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

eff_vals = [df_clean[df_clean['vol_tercile']==t]['proposer_efficiency'].values * 100
            for t in ['Low','Mid','High']]
bp = axes[0].boxplot(eff_vals, labels=['Low','Mid','High'],
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color='red', linewidth=2))
for patch in bp['boxes']:
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.7)
for i, t in enumerate(['Low','Mid','High']):
    m = df_clean[df_clean['vol_tercile']==t]['proposer_efficiency'].median()*100
    axes[0].text(i+1, axes[0].get_ylim()[1]*0.98, f'{m:.2f}%',
                 ha='center', fontsize=9, color='red', fontweight='bold')
axes[0].set_xlabel('Block Value Tercile')
axes[0].set_ylabel('Proposer Revenue Efficiency (%)')
axes[0].set_title('Proposer Revenue Efficiency by Value Tercile')

axes[1].bar(hourly_eff.index, 100 - hourly_eff.values, color='steelblue', alpha=0.8)
axes[1].set_xlabel('Hour of Day (UTC)')
axes[1].set_ylabel('Median Bid Margin (%)')
axes[1].set_title('Builder Surplus Retention by Hour of Day')
axes[1].set_xticks(range(0, 24, 2))

plt.suptitle('Proposer Revenue Efficiency Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'fig_proposer_efficiency.pdf', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig_proposer_efficiency.pdf")

print("\n=== ALL ANALYSES COMPLETE ===")
print(f"Figures saved to: {OUT}")
