"""Auction statistics and figures with every slot cut at the delivered (winning) bid. Source of the top-bid-file statistics in Section 3.2.

The top-bid file keeps recording higher bids after the proposer has chosen, so the slot maximum is usually not the winning bid.
04_winner_cutoff.py (run on the bid archive) keeps, per delivered slot, only rows at or below the delivered value:
  incr_cut  (dv - highest row below dv) / dv
  lead_cut  (dv - highest row below dv from a builder other than the winner) / dv
  n_cut, nb_cut, nr_cut   rows / builders / relays at or below dv;   n_higher rows above dv;  top_ratio = slot maximum / dv
  has_win  the delivered block hash is a row of the file;  wt its slot_t_ms;  t_above slot_t_ms of the first row above dv
Its per-slot output is shipped as data/winner_cutoff_by_slot.parquet, so this script needs no archive.
Needs work/matched.parquet and work/weekly_conc.csv (run_all.sh). Writes figures/fig_cut_*.pdf and work/winner_cutoff_numbers.csv."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import paths
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

c = pd.read_parquet(paths.CUT)
m = pd.read_parquet(paths.WORK + 'matched.parquet', columns=['slot', 'relay', 'value_eth', 'earliest_bid_ms', 'latest_bid_ms', 't', 'hour_c', 'n_bids', 'n_builders'])
d = m.merge(c, on='slot', how='inner').sort_values('slot').reset_index(drop=True)
d['lead'] = 100 * d.lead_cut.astype('float64'); d['incr'] = 100 * d.incr_cut.astype('float64')
d['ter'] = pd.qcut(d.value_eth, 3, labels=['Low', 'Mid', 'High'])
N = {}
def put(k, v): N[k] = float(v); print(f'{k:62s} {v:,.4f}' if abs(v) < 1e5 else f'{k:62s} {v:,.0f}')
def sp(a, b):
    ok = np.isfinite(a) & np.isfinite(b); return stats.spearmanr(a[ok], b[ok])[0]
def qrow(s): return dict(mean=s.mean(), median=s.median(), p10=s.quantile(.1), p90=s.quantile(.9), max=s.max())

print('=== 1. the record runs past the winning bid ===')
put('slots', len(d)); put('slot maximum / delivered value, median', d.top_ratio.median())
put('slot maximum above delivered value (%)', 100 * (d.top_ratio > 1 + 1e-6).mean()); put('slot maximum equals delivered value (%)', 100 * (abs(d.top_ratio - 1) <= 1e-6).mean())
put('rows above delivered value per slot, median', d.n_higher.median()); put('rows above delivered value, share of all rows (%)', 100 * d.n_higher.sum() / (d.n_higher.sum() + d.n_cut.sum()))
put('slots with no row above delivered value (%)', 100 * (d.n_higher == 0).mean())
put('winning block is a row of the file (%)', 100 * d.has_win.mean()); put('slots where winning block is a row', int(d.has_win.sum()))
w = d[d.has_win == 1]
put('winning row time, median (ms)', w.wt.median()); put('winning row time, p10 (ms)', w.wt.quantile(.1)); put('winning row time, p90 (ms)', w.wt.quantile(.9))
put('first row above delivered value, median (ms)', d.t_above.median()); put('old last-improvement time, median (ms)', d.latest_bid_ms.median())
put('first recorded top bid, median (ms)', d.earliest_bid_ms.median())
put('winning row after +4000 ms (% of slots with a winning row)', 100 * (w.wt > 4000).mean())

print('\n=== 2. summary table (rows at or below the delivered value) ===')
tab = pd.DataFrame({'updates': qrow(d.n_cut), 'builders': qrow(d.nb_cut), 'relays': qrow(d.nr_cut), 'incr %': qrow(d.incr), 'lead %': qrow(d.lead), 'first top bid ms': qrow(d.earliest_bid_ms), 'winning row ms': qrow(w.wt)}).T
print(tab.round(3).to_string())
for k in tab.index:
    for cc in tab.columns: N[f'summary {k} {cc}'] = float(tab.loc[k, cc])
put('slots with an increment margin', int(d.incr.notna().sum())); put('slots with a lead margin', int(d.lead.notna().sum()))
put('no recorded row below delivered value (slots)', int(d.incr.isna().sum())); put('no other-builder row below delivered value (slots)', int(d.lead.isna().sum()))
put('lead margin median, winning row in file (%)', d.lead[d.has_win == 1].median()); put('lead margin median, winning row not in file (%)', d.lead[d.has_win == 0].median())
put('old (uncut) median updates per slot', d.n_bids.median()); put('old (uncut) median builders per slot', d.n_builders.median())

print('\n=== 3. block value ===')
g = d.groupby('ter', observed=True)
for t in ['Low', 'Mid', 'High']:
    put(f'lead median, {t} tercile (%)', g.lead.median()[t]); put(f'incr median, {t} tercile (%)', g.incr.median()[t]); put(f'best competing bid / winning bid, {t} (%)', 100 - g.lead.median()[t])
    put(f'builders median, {t} tercile', g.nb_cut.median()[t]); put(f'updates median, {t} tercile', g.n_cut.median()[t])
put('best competing bid / winning bid, median (%)', 100 - d.lead.median()); put('best competing bid / winning bid, p10 slot (%)', 100 - d.lead.quantile(.9))
put('rho value ~ lead', sp(d.value_eth, d.lead)); put('rho value ~ incr', sp(d.value_eth, d.incr)); put('rho value ~ builders', sp(d.value_eth, d.nb_cut)); put('rho value ~ updates', sp(d.value_eth, d.n_cut))
put('rho value ~ winning row time', sp(w.value_eth, w.wt)); put('winning row time median, Low (ms)', w.groupby('ter', observed=True).wt.median()['Low']); put('winning row time median, High (ms)', w.groupby('ter', observed=True).wt.median()['High'])
rng = np.random.default_rng(1)
def boot(a, b, n=8000, B=1000):
    ok = np.isfinite(a) & np.isfinite(b); a = a[ok].to_numpy(); b = b[ok].to_numpy()
    rs = [stats.spearmanr(a[i], b[i])[0] for i in (rng.choice(len(a), n, replace=False) for _ in range(B))]; return np.percentile(rs, 2.5), np.percentile(rs, 97.5)
for nm, (a, b) in {'value ~ lead': (d.value_eth, d.lead), 'value ~ builders': (d.value_eth, d.nb_cut), 'updates ~ lead': (d.n_cut, d.lead)}.items():
    lo, hi = boot(a, b); put(f'bootstrap CI low, {nm}', lo); put(f'bootstrap CI high, {nm}', hi)

print('\n=== 4. record density and relay count ===')
put('rho updates ~ lead', sp(d.n_cut, d.lead)); put('rho updates ~ incr', sp(d.n_cut, d.incr)); put('rho relays ~ lead', sp(d.nr_cut, d.lead)); put('rho relays ~ incr', sp(d.nr_cut, d.incr)); put('rho builders ~ lead', sp(d.nb_cut, d.lead))
rt = d[d.nr_cut.between(1, 7)].groupby('nr_cut').agg(incr=('incr', 'median'), lead=('lead', 'median'), updates=('n_cut', 'median'), slots=('slot', 'size')); print(rt.round(3).to_string())
for r in rt.index:
    put(f'relays={r}: lead median (%)', rt.lead[r]); put(f'relays={r}: incr median (%)', rt.incr[r]); put(f'relays={r}: updates median', rt.updates[r]); put(f'relays={r}: slots', rt.slots[r])
d['nq'] = pd.qcut(d.n_cut, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']); edges = pd.qcut(d.n_cut, 5, retbins=True)[1]; print('density quintile edges', edges)
for i, e in enumerate(edges): N[f'density edge {i}'] = float(e)
pv = d.pivot_table(index='nq', columns='ter', values='lead', aggfunc='median', observed=True); print(pv.round(3).to_string())
for q in pv.index:
    for t in pv.columns: put(f'lead median, density {q}, {t} (%)', pv.loc[q, t])
    put(f'lead median, density {q}, all (%)', d.lead[d.nq == q].median())
for q in ['Q2', 'Q3', 'Q4', 'Q5']: put(f'rho value ~ lead within density {q}', sp(d.value_eth[d.nq == q], d.lead[d.nq == q]))
put('rho value ~ lead within density Q1', sp(d.value_eth[d.nq == 'Q1'], d.lead[d.nq == 'Q1']))

print('\n=== 5. over time ===')
d['week'] = d.t.dt.tz_localize(None).dt.to_period('W').dt.start_time
wm = d.groupby('week').agg(lead=('lead', 'median'), incr=('incr', 'median'), nb=('nb_cut', 'median'), n_cut=('n_cut', 'median'), n=('slot', 'size')); wm.to_csv(paths.WORK + 'weekly_auction_cut.csv')
put('weekly median builders, Sep-Dec 2024: median', wm.loc[:'2024-12-31'].nb.median()); put('weekly median builders, 2026: min', wm.loc['2026-01-01':].nb.min()); put('weekly median builders, 2026: max', wm.loc['2026-01-01':].nb.max()); put('weekly median builders, 2026: median', wm.loc['2026-01-01':].nb.median())
put('weekly median updates: min', wm.n_cut.min()); put('weekly median updates: max', wm.n_cut.max()); put('weekly median lead: min (%)', wm.lead.min()); put('weekly median lead: max (%)', wm.lead.max())
put('weekly median builders, week of 2024-12-30', wm.nb.get(pd.Timestamp('2024-12-30'), np.nan)); put('weekly median builders, week of 2025-01-06', wm.nb.get(pd.Timestamp('2025-01-06'), np.nan))
d['wk2'] = d.t.dt.tz_localize(None).dt.to_period('W-SUN').dt.start_time
periods = [('2024-09..12', '2024-09-01', '2025-01-05'), ('2025-01..04', '2025-01-06', '2025-04-30'), ('2025-05..12', '2025-05-01', '2025-12-31'), ('2026-01..05', '2026-01-01', '2026-06-01')]
for lab, sub in [('pooled', d), ('agnostic', d[d.relay == 'agnostic'])]:
    ww = sub.groupby('wk2').agg(nb=('nb_cut', 'median'), n=('slot', 'size')); ww = ww[ww.n >= 300]
    for nm, a, b in periods: put(f'{lab} builders per slot, {nm}', ww[(ww.index >= a) & (ww.index <= b)].nb.mean())
hh = pd.read_csv(paths.WORK + 'weekly_conc.csv', index_col=0, parse_dates=True)
j = wm.join(hh.drop(columns=['n']), how='inner').sort_index(); j['hhi_lag'] = j.hhi.shift(1); jj = j.dropna(subset=['hhi_lag'])
put('weeks in HHI test', len(jj))
for k in ['lead', 'incr']:
    r, p = stats.spearmanr(jj.hhi_lag, jj[k]); put(f'lagged HHI ~ weekly median {k}: rho', r); put(f'lagged HHI ~ weekly median {k}: p', p); put(f'lagged HHI ~ weekly median {k}: OLS R2', stats.linregress(jj.hhi_lag, jj[k]).rvalue ** 2)
r, p = stats.spearmanr(jj.n_cut, jj.lead); put('weekly median updates ~ weekly median lead: rho', r)
def ljung_box(y, lags):   # direct sums: statsmodels' version is far slower on 1.2M points
    y = y - y.mean(); n = len(y); den = (y * y).sum(); r = np.array([(y[k:] * y[:-k]).sum() / den for k in range(1, max(lags) + 1)])
    return {L: n * (n + 2) * (r[:L] ** 2 / (n - np.arange(1, L + 1))).sum() for L in lags}
x = d.lead.dropna().to_numpy(); lb = ljung_box((x - x.mean()) ** 2, [10, 20, 50])
for L in [10, 20, 50]: put(f'Ljung-Box Q{L} on squared demeaned lead margin', lb[L])
x = np.log(d.n_cut.clip(lower=1)).to_numpy(); put('lag-1 autocorrelation of log updates per slot', np.corrcoef(x[:-1], x[1:])[0, 1]); x = d.lead.fillna(d.lead.median()).rank().to_numpy(); put('lag-1 rank autocorrelation of lead margin', np.corrcoef(x[:-1], x[1:])[0, 1])
hr = d.groupby('hour_c').lead.median(); put('hourly median lead: min (%)', hr.min()); put('hourly median lead: max (%)', hr.max()); print('hours with the widest lead:', hr.nlargest(5).round(3).to_dict())
pd.Series(N).to_csv(paths.WORK + 'winner_cutoff_numbers.csv', header=['value'])

# ------------------------------------------------------------------ figures
F = paths.FIG; plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}); B = 'steelblue'
def med(ax, v, fmt):
    ax.axvline(v, color='crimson', lw=1.2, ls='--'); ax.text(0.97, 0.93, 'median ' + fmt.format(v), transform=ax.transAxes, ha='right', color='crimson')
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
ax[0].hist(d.n_cut.clip(upper=500), bins=100, color=B); med(ax[0], d.n_cut.median(), '{:.0f}'); ax[0].set_xlabel('Top-bid updates at or below the winning bid'); ax[0].set_ylabel('Slots')
ax[1].hist(d.nb_cut, bins=np.arange(-0.5, 37.5), color=B); med(ax[1], d.nb_cut.median(), '{:.0f}'); ax[1].set_xlabel('Builders setting a top bid at or below the winning bid')
ax[2].hist(d.lead.dropna().clip(upper=20), bins=100, color=B); med(ax[2], d.lead.median(), '{:.2f}%'); ax[2].set_xlabel('Lead margin (%), clipped at 20'); ax[2].set_yscale('log')
plt.tight_layout(); plt.savefig(F + 'fig_cut_overview.pdf', bbox_inches='tight'); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
data = [d.lead[d.ter == t].dropna() for t in ['Low', 'Mid', 'High']]
ax[0].boxplot(data, showfliers=False, tick_labels=['Low', 'Mid', 'High'], medianprops=dict(color='crimson'))
for i, s in enumerate(data): ax[0].text(i + 1.08, s.median(), f'{s.median():.2f}%', color='crimson', va='bottom', fontsize=9)
ax[0].set_xlabel('Block value tercile'); ax[0].set_ylabel('Lead margin (%)'); ax[0].set_title('By block value', fontsize=10)
for t, col in zip(['Low', 'Mid', 'High'], ['#9ecae1', '#4292c6', '#084594']): ax[1].plot(range(5), pv[t].to_numpy(), marker='o', color=col, label=f'{t} value')
ax[1].set_yscale('log'); ax[1].set_xticks(range(5)); ax[1].set_xticklabels([f'Q{i + 1}\n{int(edges[i]) + (1 if i else 0)}-{int(edges[i + 1])}' for i in range(5)]); ax[1].set_xlabel('Quintile of top-bid updates at or below the winning bid'); ax[1].set_ylabel('Median lead margin (%)'); ax[1].legend(frameon=False); ax[1].set_title('By record density', fontsize=10)
plt.tight_layout(); plt.savefig(F + 'fig_cut_margin_value.pdf', bbox_inches='tight'); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
ax[0].hist(d.earliest_bid_ms.clip(-12000, 4000), bins=120, color=B); med(ax[0], d.earliest_bid_ms.median(), '{:,.0f} ms'); ax[0].set_xlabel('First recorded top bid (ms from slot start)'); ax[0].set_ylabel('Slots')
ax[1].hist(w.wt.clip(-2000, 6000), bins=120, color=B); med(ax[1], w.wt.median(), '{:,.0f} ms'); ax[1].set_xlabel('Winning bid, where recorded (ms)')
ax[2].hist(d.t_above.dropna().clip(-2000, 6000), bins=120, color=B); med(ax[2], d.t_above.median(), '{:,.0f} ms'); ax[2].set_xlabel('First recorded bid above the winning bid (ms)')
for a in ax: a.axvline(0, color='k', lw=0.6)
plt.tight_layout(); plt.savefig(F + 'fig_cut_timing.pdf', bbox_inches='tight'); plt.close()

fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
ax[0].plot(wm.index, wm.lead, color=B, label='Lead margin'); ax[0].plot(wm.index, wm.incr, color='darkorange', label='Increment margin'); ax[0].set_ylabel('Weekly median (%)'); ax[0].legend(frameon=False)
ax[1].plot(wm.index, wm.nb, color=B); ax[1].set_ylabel('Builders per slot\n(weekly median)')
ax[2].plot(wm.index, wm.n_cut, color=B); ax[2].set_ylabel('Top-bid updates per slot\n(weekly median)')
plt.tight_layout(); plt.savefig(F + 'fig_cut_timeseries.pdf', bbox_inches='tight'); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8)); xs = rt.index.astype(int)
ax[0].bar(xs - 0.2, rt.lead, 0.4, color=B, label='Lead'); ax[0].bar(xs + 0.2, rt.incr, 0.4, color='darkorange', label='Increment'); ax[0].set_xlabel('Relays per slot'); ax[0].set_ylabel('Median margin (%)'); ax[0].legend(frameon=False)
ax[1].bar(xs, rt.updates, color=B); ax[1].set_xlabel('Relays per slot'); ax[1].set_ylabel('Median top-bid updates')
ax[2].bar(xs, rt.slots / 1000, color=B); ax[2].set_xlabel('Relays per slot'); ax[2].set_ylabel('Slots (thousands)')
plt.tight_layout(); plt.savefig(F + 'fig_cut_relay.pdf', bbox_inches='tight'); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4)); bc = [100 - g.lead.median()[t] for t in ['Low', 'Mid', 'High']]
ax[0].bar(['Low', 'Mid', 'High'], bc, color=B); ax[0].set_ylim(99, 100); ax[0].set_xlabel('Block value tercile'); ax[0].set_ylabel('Best competing bid / winning bid, median (%)')
for i, v in enumerate(bc): ax[0].text(i, v + 0.01, f'{v:.2f}', ha='center')
ax[1].plot(hr.index, hr.values, marker='o', color=B); ax[1].set_xlabel('Hour of day (UTC)'); ax[1].set_ylabel('Median lead margin (%)'); ax[1].set_ylim(0, None)
plt.tight_layout(); plt.savefig(F + 'fig_cut_best_competing.pdf', bbox_inches='tight'); plt.close()
print('\nfigures written to', F)
