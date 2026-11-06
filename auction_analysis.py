import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/Users/kylerothwell/pbs_research/figures/'
import os; os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data...")
df = pd.read_parquet('/Users/kylerothwell/pbs_research/bid_auction_stats.parquet')
df = df.dropna(subset=['n_bids', 'value_eth'])
df['bid_margin_pct'] = df['bid_margin'] * 100
df['log_value'] = np.log(df['value_eth'])

df['vol_tercile'] = pd.qcut(df['value_eth'], q=3, labels=['Low','Mid','High'], duplicates='drop')

print(f"Working dataset: {len(df):,} slots\n")

print("Figure 1: Bid count distribution...")
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].hist(df['n_bids'], bins=80, color='steelblue', edgecolor='none', alpha=0.85)
axes[0].axvline(df['n_bids'].median(), color='red', linestyle='--', label=f"Median = {df['n_bids'].median():.0f}")
axes[0].set_xlabel('Bids per Slot')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Bid Count Distribution')
axes[0].legend()

axes[1].hist(df['n_builders'], bins=30, color='steelblue', edgecolor='none', alpha=0.85)
axes[1].axvline(df['n_builders'].median(), color='red', linestyle='--', label=f"Median = {df['n_builders'].median():.0f}")
axes[1].set_xlabel('Unique Builders per Slot')
axes[1].set_title('Builder Count Distribution')
axes[1].legend()

axes[2].hist(df['bid_margin_pct'].clip(0, 20), bins=80, color='steelblue', edgecolor='none', alpha=0.85)
axes[2].axvline(df['bid_margin_pct'].median(), color='red', linestyle='--', label=f"Median = {df['bid_margin_pct'].median():.3f}%")
axes[2].set_xlabel('Bid Margin % (clipped at 20%)')
axes[2].set_title('Bid Margin Distribution')
axes[2].legend()

plt.suptitle('MEV-Boost Auction Competitiveness', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'fig_auction_competitiveness.pdf', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig_auction_competitiveness.pdf")

print("Figure 2: Bid margin vs block value...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df['value_bin'] = pd.qcut(df['value_eth'], q=50, duplicates='drop')
bin_stats = df.groupby('value_bin', observed=True).agg(
    median_margin=('bid_margin_pct', 'median'),
    median_value=('value_eth', 'median'),
    n=('bid_margin_pct', 'count')
).reset_index()

axes[0].plot(bin_stats['median_value'], bin_stats['median_margin'], 
             color='steelblue', linewidth=2)
axes[0].set_xlabel('Block Value (ETH) — Median per Bin')
axes[0].set_ylabel('Median Bid Margin (%)')
axes[0].set_title('Bid Margin vs Block Value')
axes[0].set_xscale('log')

margin_by_tercile = [df[df['vol_tercile']==t]['bid_margin_pct'].clip(0,10).values 
                     for t in ['Low','Mid','High']]
bp = axes[1].boxplot(margin_by_tercile, labels=['Low\nValue','Mid\nValue','High\nValue'],
                     patch_artist=True, medianprops=dict(color='red', linewidth=2))
for patch in bp['boxes']:
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.7)
axes[1].set_ylabel('Bid Margin % (clipped at 10%)')
axes[1].set_title('Bid Margin by Block Value Tercile')

for i, t in enumerate(['Low','Mid','High']):
    med = df[df['vol_tercile']==t]['bid_margin_pct'].median()
    axes[1].text(i+1, 9.5, f'{med:.3f}%', ha='center', fontsize=9, color='red')

plt.suptitle('Auction Competitiveness Increases with Block Value', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'fig_margin_vs_value.pdf', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig_margin_vs_value.pdf")

print("Figure 3: Bid timing...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

earliest = df['earliest_bid_ms'].clip(-12000, 0)
latest = df['latest_bid_ms'].clip(0, 4000)

axes[0].hist(earliest, bins=80, color='steelblue', edgecolor='none', alpha=0.85)
axes[0].axvline(earliest.median(), color='red', linestyle='--', 
                label=f"Median = {earliest.median():.0f}ms")
axes[0].set_xlabel('Earliest Bid (ms relative to slot time)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('When Do Builders Start Bidding?')
axes[0].legend()

axes[1].hist(latest, bins=80, color='steelblue', edgecolor='none', alpha=0.85)
axes[1].axvline(latest.median(), color='red', linestyle='--',
                label=f"Median = {latest.median():.0f}ms")
axes[1].set_xlabel('Latest Bid (ms relative to slot time)')
axes[1].set_title('When Do Builders Stop Bidding?')
axes[1].legend()

plt.suptitle('Builder Bid Timing Relative to Slot Time', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'fig_bid_timing.pdf', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig_bid_timing.pdf")

print("Figure 4: Timing by value tercile...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for t, color in zip(['Low','Mid','High'], ['#2196F3','#FF9800','#F44336']):
    sub = df[df['vol_tercile']==t]
    axes[0].hist(sub['latest_bid_ms'].clip(0,4000), bins=60, alpha=0.5, 
                 label=t, color=color, density=True)
    axes[1].hist(sub['n_builders'], bins=25, alpha=0.5,
                 label=t, color=color, density=True)

axes[0].set_xlabel('Latest Bid (ms after slot time)')
axes[0].set_ylabel('Density')
axes[0].set_title('Late Bidding by Block Value Tercile')
axes[0].legend(title='Value Tercile')

axes[1].set_xlabel('Unique Builders per Slot')
axes[1].set_title('Builder Participation by Block Value Tercile')
axes[1].legend(title='Value Tercile')

plt.suptitle('High-Value Slots Attract More Builders and Later Bids', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'fig_timing_by_value.pdf', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig_timing_by_value.pdf")

print("Figure 5: Time series of competitiveness...")
import datetime
GENESIS = datetime.datetime(2020, 12, 1, 12, 0, 23)
df['date'] = df['slot'].apply(lambda s: GENESIS + datetime.timedelta(seconds=s*12))
df['week'] = df['date'].dt.to_period('W')

weekly = df.groupby('week').agg(
    median_margin=('bid_margin_pct', 'median'),
    median_builders=('n_builders', 'median'),
    median_bids=('n_bids', 'median'),
).reset_index()
weekly['week_dt'] = weekly['week'].dt.to_timestamp()

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

axes[0].plot(weekly['week_dt'], weekly['median_margin'], color='steelblue', linewidth=1)
axes[0].set_ylabel('Median Bid Margin (%)')
axes[0].set_title('Weekly Median Bid Margin')

axes[1].plot(weekly['week_dt'], weekly['median_builders'], color='darkorange', linewidth=1)
axes[1].set_ylabel('Median Builders per Slot')
axes[1].set_title('Weekly Median Builder Participation')

axes[2].plot(weekly['week_dt'], weekly['median_bids'], color='green', linewidth=1)
axes[2].set_ylabel('Median Bids per Slot')
axes[2].set_title('Weekly Median Bid Count')

plt.suptitle('MEV-Boost Auction Dynamics Over Time', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'fig_competitiveness_timeseries.pdf', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig_competitiveness_timeseries.pdf")

print("\n=== SUMMARY STATISTICS FOR PAPER ===")
print(f"Total slots analyzed: {len(df):,}")
print(f"\nBids per slot:")
print(f"  Mean:   {df['n_bids'].mean():.1f}")
print(f"  Median: {df['n_bids'].median():.0f}")
print(f"  P10:    {df['n_bids'].quantile(0.10):.0f}")
print(f"  P90:    {df['n_bids'].quantile(0.90):.0f}")

print(f"\nBuilders per slot:")
print(f"  Mean:   {df['n_builders'].mean():.1f}")
print(f"  Median: {df['n_builders'].median():.0f}")
print(f"  P10:    {df['n_builders'].quantile(0.10):.0f}")
print(f"  P90:    {df['n_builders'].quantile(0.90):.0f}")

print(f"\nBid margin (%):")
print(f"  Median: {df['bid_margin_pct'].median():.4f}%")
print(f"  Mean:   {df['bid_margin_pct'].mean():.4f}%")
print(f"  P75:    {df['bid_margin_pct'].quantile(0.75):.4f}%")
print(f"  P90:    {df['bid_margin_pct'].quantile(0.90):.4f}%")
print(f"  P99:    {df['bid_margin_pct'].quantile(0.99):.4f}%")

print(f"\nBid margin by value tercile:")
for t in ['Low','Mid','High']:
    med = df[df['vol_tercile']==t]['bid_margin_pct'].median()
    n = (df['vol_tercile']==t).sum()
    print(f"  {t}: {med:.4f}% (n={n:,})")

rho, pval = stats.spearmanr(df['value_eth'], df['bid_margin_pct'])
print(f"\nSpearman rho (value vs margin): {rho:.4f}, p={pval:.2e}")

rho2, pval2 = stats.spearmanr(df['value_eth'], df['n_builders'])
print(f"Spearman rho (value vs n_builders): {rho2:.4f}, p={pval2:.2e}")

rho3, pval3 = stats.spearmanr(df['value_eth'], df['latest_bid_ms'])
print(f"Spearman rho (value vs latest_bid_ms): {rho3:.4f}, p={pval3:.2e}")

print("\nAll figures saved to:", OUTPUT_DIR)
