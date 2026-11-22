"""fig_relay_timing_artifact, drawn from work/relay_timing_q1_2025.csv (written by 01_relay_timing_q1_2025.py).
The original figure was drawn from the same eight medians and counts typed in by hand; this reads them from the output instead.
Relays with fewer than 10,000 bids in the quarter (securerpc, wenmerge) are left out, as in the paper's table."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import paths
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
NAMES = {'relay.edennetwork.io': 'Eden\nNetwork', 'aestus.live': 'Aestus', 'relay.ultrasound.money': 'Ultrasound', 'boost-relay.flashbots.net': 'Flashbots',
         'agnostic-relay.net': 'Agnostic', 'bloxroute.max-profit.blxrbdn.com': 'BloXroute\n(max-profit)', 'bloxroute.regulated.blxrbdn.com': 'BloXroute\n(regulated)', 'titanrelay.xyz': 'Titan'}
t = pd.read_csv(paths.WORK + 'relay_timing_q1_2025.csv')
t = t[t.n >= 10000].sort_values('median_ms'); lab = [NAMES.get(r, r) for r in t.relay]
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
bars = ax[0].barh(lab, t.median_ms, color=['steelblue' if m < 0 else 'darkorange' for m in t.median_ms], alpha=0.85)
ax[0].axvline(0, color='black', lw=0.8); ax[0].set_xlabel('Median Bid Timing (ms relative to slot start)')
ax[0].set_title('Bid Timestamp by Relay\n(Q1 2025, pooled across all builders)')
for b, m in zip(bars, t.median_ms):
    ax[0].text(m + (100 if m >= 0 else -100), b.get_y() + b.get_height() / 2, f'{m:,.0f}ms', va='center', ha='left' if m >= 0 else 'right', fontsize=8)
lo, hi = ax[0].get_xlim(); ax[0].set_xlim(lo - 900, hi + 900)
ax[1].barh(lab, t.n / 1e6, color='steelblue', alpha=0.7); ax[1].set_xscale('log'); ax[1].set_xlabel('Observations (millions, log scale)')
us = 100 * t.loc[t.relay == 'relay.ultrasound.money', 'n'].iloc[0] / pd.read_csv(paths.WORK + 'relay_timing_q1_2025.csv').n.sum()
ax[1].set_title(f'Bid Volume by Relay\n(Ultrasound dominates at {us:.1f}%)')
plt.suptitle('Bid Timing by Relay in the Top-Bid Files (Finding 8)', fontsize=13, fontweight='bold'); plt.tight_layout()
plt.savefig(paths.FIG + 'fig_relay_timing_artifact.pdf', dpi=150, bbox_inches='tight'); plt.close()
print('saved fig_relay_timing_artifact.pdf;', len(t), 'relays; ultrasound share', round(us, 2))
