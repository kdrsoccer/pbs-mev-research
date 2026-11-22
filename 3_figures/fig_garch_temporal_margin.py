"""Figures:
  fig_garch_clustering       AR(5) residuals of log(1+v), their squares, and the Table 4 GARCH(1,1) conditional volatility
  fig_temporal_patterns      median and mean block value by hour (UTC) and by weekday, slot-derived time
  fig_margin_vs_value_fixed  median margin by value bin; margin by value tercile
  fig_margin_regression      partial-effect plots for the margin regression (same sample and model as the regression table)
Inputs: work/slim.parquet, work/matched.parquet (written by 2_analysis/01 and 03), work/garch_level.json (written by 2_analysis/05;
if absent the Table 4 point estimates are used)."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import paths
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import lfilter
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
W, F = paths.WORK, paths.FIG
plt.rcParams.update({'figure.dpi': 150, 'font.size': 10, 'figure.facecolor': 'white'})

df = pd.read_parquet(W + 'slim.parquet').sort_values('slot').reset_index(drop=True)

# ---------------- fig_garch_clustering ----------------
u = np.log1p(df.value_eth.values)
e = np.asarray(AutoReg(u, lags=5, trend='c').fit().resid)            # log units; Table 4 is in percent (x100)
try:
    g = json.load(open(W + 'garch_level.json'))['table4']; om, al, be = g['omega'], g['alpha'], g['beta']; src = 'work/garch_level.json'
except Exception:
    om, al, be = 0.769, 0.044, 0.965; src = 'Table 4 point estimates'
e2p = (100 * e) ** 2                                                  # percent^2, the scale omega is estimated on
s0 = e2p.mean()                                                       # start the recursion at the sample variance
h = lfilter([1.0], [1.0, -be], om + al * e2p[:-1], zi=[be * s0])[0]   # sigma2_t = omega + alpha e2_{t-1} + beta sigma2_{t-1}
cv = np.sqrt(np.r_[s0, h]) / 100                                      # back to log units
slots = df.slot.values[5:]
fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
ax[0].plot(slots, e, color='#4472c4', lw=0.3, rasterized=True); ax[0].set_ylabel('Residual'); ax[0].set_title('AR(5) residuals of log(1+v)')
ax[1].plot(slots, e ** 2, color='#ed7d31', lw=0.3, rasterized=True); ax[1].set_ylabel('Squared residual'); ax[1].set_title('Squared residuals')
ax[2].plot(slots, cv, color='#70ad47', lw=0.3, rasterized=True); ax[2].set_ylabel('Conditional volatility'); ax[2].set_xlabel('Slot')
ax[2].set_title(rf'GARCH(1,1) conditional volatility ($\hat\alpha={al:.3f}$, $\hat\beta={be:.3f}$)')
plt.tight_layout(); plt.savefig(F + 'fig_garch_clustering.pdf', bbox_inches='tight', dpi=150); plt.close()
print(f'fig_garch_clustering: n={len(e):,} params from {src}: omega={om:.3f} alpha={al:.3f} beta={be:.3f}')

# ---------------- fig_temporal_patterns ----------------
hr = df.groupby('hour_c').value_eth.agg(['median', 'mean']); dw = df.groupby('dow_c').value_eth.agg(['median', 'mean'])
Hh = stats.kruskal(*[x.values for _, x in df.groupby('hour_c').value_eth])[0]
Hd = stats.kruskal(*[x.values for _, x in df.groupby('dow_c').value_eth])[0]
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
a = ax[0]; b = a.twinx()
a.plot(hr.index, hr['median'], '-o', color='#1f4e79', ms=4, label='Median'); b.plot(hr.index, hr['mean'], '--s', color='#d98880', ms=4, label='Mean')
a.set_xlabel('Hour of day (UTC)'); a.set_ylabel('Median block value (ETH)', color='#1f4e79'); b.set_ylabel('Mean block value (ETH)', color='#c0504d')
a.set_xticks(range(0, 24, 2)); a.legend(loc='upper left'); b.legend(loc='upper right')
a.set_title(f'Intraday pattern: median and mean block value by hour (KW $H$={Hh:,.0f})')
a = ax[1]; b = a.twinx(); x = np.arange(7); days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
a.bar(x - 0.2, dw['median'], 0.4, color='#1f4e79', label='Median'); b.bar(x + 0.2, dw['mean'], 0.4, color='#d98880', label='Mean')
a.set_xticks(x); a.set_xticklabels(days); a.set_ylabel('Median block value (ETH)', color='#1f4e79'); b.set_ylabel('Mean block value (ETH)', color='#c0504d')
a.legend(loc='upper left'); b.legend(loc='upper right'); a.set_title(f'Day-of-week pattern (KW $H$={Hd:,.0f})')
plt.tight_layout(); plt.savefig(F + 'fig_temporal_patterns.pdf', bbox_inches='tight'); plt.close()
print(f'fig_temporal_patterns: KW hour H={Hh:,.0f}  KW dow H={Hd:,.0f}')

# ---------------- matched sample (same filter as the regression table) ----------------
m = pd.read_parquet(W + 'matched.parquet')
c = m[np.isfinite(m.m) & (m.m >= 0) & (m.m < 100)].copy()

# ---------------- fig_margin_vs_value_fixed ----------------
c['bin'] = pd.qcut(c.value_eth, 50, duplicates='drop')
bm = c.groupby('bin', observed=True).agg(x=('value_eth', 'median'), y=('m', 'median'))
c['ter'] = pd.qcut(c.value_eth, 3, labels=['Low', 'Mid', 'High'])
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot(bm.x, bm.y, color='#4472c4', lw=1.5); ax[0].set_xscale('log')
ax[0].set_xlabel('Block Value (ETH), median per bin'); ax[0].set_ylabel('Median Bid Margin (%)'); ax[0].set_title('Bid Margin vs Block Value')
groups = [c[c.ter == t].m.values for t in ['Low', 'Mid', 'High']]
bp = ax[1].boxplot(groups, tick_labels=['Low\nValue', 'Mid\nValue', 'High\nValue'], showfliers=False, patch_artist=True, medianprops=dict(color='red', lw=2))
for p in bp['boxes']: p.set_facecolor('#7ba7cc')
top = ax[1].get_ylim()[1]
for i, gv in enumerate(groups): ax[1].text(i + 1, top * 1.01, f'{np.median(gv):.3f}%', ha='center', va='bottom', color='red', fontsize=8)
ax[1].set_ylim(0, top * 1.1); ax[1].set_ylabel('Bid Margin (%)'); ax[1].set_title('Bid Margin by Block Value Tercile')
fig.suptitle('Auction Competitiveness Increases with Block Value', fontweight='bold')
plt.tight_layout(); plt.savefig(F + 'fig_margin_vs_value_fixed.pdf', bbox_inches='tight'); plt.close()
print('fig_margin_vs_value_fixed: tercile medians', [round(float(np.median(gv)), 3) for gv in groups], ' n =', len(c))

# ---------------- fig_margin_regression ----------------
X = pd.DataFrame({'log_value': np.log(c.value_eth), 'n_builders': c.n_builders, 'n_relays': c.n_relays, 'hour': c.hour_c, 'dow': c.dow_c})
res = sm.OLS(c.m.values, sm.add_constant(X)).fit(cov_type='HC0')
print('fig_margin_regression: OLS coefficients'); print(res.params.round(4).to_string()); print('R2', round(res.rsquared, 4), 'n', int(res.nobs))
mu = X.mean(); smp = c.sample(40000, random_state=0); Xs = X.loc[smp.index]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for a, (col, lab, colr) in zip(ax, [('log_value', 'log(Block Value ETH)', '#4472c4'), ('n_builders', 'Unique Builders per Slot', '#ed7d31'), ('n_relays', 'Relays per Slot', '#2e8b57')]):
    a.scatter(Xs[col], smp.m, s=1, alpha=0.15, color=colr, rasterized=True)
    xx = np.linspace(X[col].quantile(0.001), X[col].quantile(0.999), 100)
    yy = res.params['const'] + sum(res.params[k] * mu[k] for k in X.columns if k != col) + res.params[col] * xx
    a.plot(xx, yy, color='red', lw=1.8); a.set_ylim(-0.5, 15); a.set_xlabel(lab); a.set_ylabel('Bid Margin (%)')
    star = '***' if res.pvalues[col] < 0.001 else ''
    a.set_title(f"Partial: {lab.split(' (')[0].split(' per')[0]}\n" + rf"$\beta$={res.params[col]:.4f}{star}")
fig.suptitle('Multivariate Bid Margin Regression: Partial Effects', fontweight='bold')
plt.tight_layout(); plt.savefig(F + 'fig_margin_regression.pdf', bbox_inches='tight', dpi=150); plt.close()
print('done')
