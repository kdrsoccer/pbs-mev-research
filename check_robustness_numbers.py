"""Recompute the robustness statistics of Findings 1 and 6 and the top-bid-file statistics of Section 3.2 from data/ and compare each to the value printed in the paper.
Needs only data/bid_auction_stats.parquet and data/eth_price_hourly.parquet. About one minute.
The bid-level additions (Findings 8 and 8b) are checked by 5_bid_level/04 and 05, which need the bid archive."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import paths

rows = []
def chk(name, paper, got, dec=None):
    if dec is None:
        s = str(paper); dec = len(s.split('.')[1]) if '.' in s else 0
    ok = abs(round(float(got), dec) - float(paper)) < 1e-9
    rows.append((name, paper, round(float(got), dec + 2), 'PASS' if ok else 'DIFF'))

df = pd.read_parquet(paths.SRC, columns=['slot', 'value_eth', 'relay', 'builder_pubkey', 'n_builders'])
df = df.drop_duplicates('slot', keep='first').sort_values('slot').reset_index(drop=True)
df['t'] = pd.to_datetime(1606824023 + 12 * df['slot'], unit='s', utc=True)
chk('unique delivered slots', 1271079, len(df))
import os
if os.path.exists(paths.CUT):   # builders per slot = builders setting a top bid at or below the winning bid
    df = df.merge(pd.read_parquet(paths.CUT, columns=['slot', 'nb_cut']), on='slot', how='left'); df['n_builders'] = df['nb_cut']

# ---- Finding 1: volatility windows (Table: regime_robust) ----
v = df['value_eth'].to_numpy(); W = 300
def rvol(x): return pd.Series(np.diff(np.log1p(x), prepend=np.nan)).rolling(W).std()
def sp(a, b):
    m = np.isfinite(a) & np.isfinite(b); return spearmanr(a[m], b[m]).statistic
def ratios(val, vol):
    m = np.isfinite(vol); q1, q2 = np.nanquantile(vol, [0.33, 0.67]); lo, hi = val[m & (vol <= q1)], val[m & (vol > q2)]   # same cut points as the regime table
    return hi.mean() / lo.mean(), np.median(hi) / np.median(lo)
rv = rvol(v)
for lag, name, rho, mr, dr in [(0, 'window ending at t', 0.494, 7.0, 2.7), (1, 'window ending at t-1', 0.490, 5.1, 2.7), (W, 'window ending at t-300', 0.414, 3.5, 2.3)]:
    x = rv.shift(lag).to_numpy(); a, b = ratios(v, x)
    chk(f'{name}: Spearman', rho, sp(v, x), 3); chk(f'{name}: ratio of means', mr, a, 1); chk(f'{name}: ratio of medians', dr, b, 1)
rng = np.random.default_rng(0); P = []
for _ in range(5):
    vs = rng.permutation(v); rs = rvol(vs).to_numpy(); P.append((sp(vs, rs),) + ratios(vs, rs))
P = np.mean(P, axis=0)
chk('placebo (values permuted): Spearman', 0.005, P[0], 3); chk('placebo: ratio of means', 2.5, P[1], 1); chk('placebo: ratio of medians', 1.0, P[2], 1)

# ---- Findings 1 and 6: ETH volatility ----
e = pd.read_parquet(paths.ETH)[['datetime', 'eth_close']].copy(); e.columns = ['h', 'p']
e['h'] = pd.to_datetime(e['h'], utc=True) if not str(e['h'].dtype).startswith('datetime64') else e['h']
if e['h'].dt.tz is None: e['h'] = e['h'].dt.tz_localize('UTC')
e = e.sort_values('h').drop_duplicates('h'); e['ret'] = np.log(e['p']).diff(); e['absret'] = e['ret'].abs(); e['rv24'] = e['ret'].rolling(24).std()
df['h'] = df['t'].dt.floor('h')
hv = df.groupby('h')['value_eth'].agg(['mean', 'median', 'count']).reset_index()
m = hv.merge(e, on='h', how='inner'); m = m[m['count'] >= 20].sort_values('h')
chk('hours with >=20 blocks and an ETH price', 14621, len(m))
chk('Spearman hourly median value vs 24h ETH realized vol', 0.245, sp(m['median'].to_numpy(), m['rv24'].to_numpy()), 3)
chk('Spearman hourly median value vs |hourly ETH return|', 0.234, sp(m['median'].to_numpy(), m['absret'].to_numpy()), 3)
q1, q2 = m['rv24'].quantile([1 / 3, 2 / 3]); lo, mid, hi = m[m.rv24 <= q1], m[(m.rv24 > q1) & (m.rv24 <= q2)], m[m.rv24 > q2]
for nm, g, mean, med in [('low', lo, 0.035, 0.0137), ('mid', mid, 0.049, 0.0172), ('high', hi, 0.085, 0.0210)]:
    chk(f'ETH-vol tercile {nm}: mean block value', mean, g['mean'].mean(), 3); chk(f'ETH-vol tercile {nm}: median of hourly medians', med, g['median'].median(), 4)
chk('ETH-vol high/low ratio of means', 2.4, hi['mean'].mean() / lo['mean'].mean(), 1)
chk('ETH-vol high/low ratio of medians', 1.5, hi['median'].median() / lo['median'].median(), 1)
m['dlv'] = np.log(m['median']).diff(); ok = np.isfinite(m['dlv']) & np.isfinite(m['ret']) & (m['h'].diff() == pd.Timedelta('1h'))
chk('R2: hourly change in log median value on hourly ETH return', 0.005, np.corrcoef(m.loc[ok, 'ret'], m.loc[ok, 'dlv'])[0, 1] ** 2, 3)

# ---- Findings 3 and 5c: within-relay concentration and participation (Table: hhi_relay) ----
df['week'] = df['t'].dt.tz_localize(None).dt.to_period('W-SUN').dt.start_time
def hhi(s): p = s.value_counts(normalize=True); return float((p ** 2).sum() * 10000)
def weekly(sub):
    g = sub.groupby('week'); w = pd.DataFrame({'hhi': g['builder_pubkey'].apply(hhi), 'nb': g['n_builders'].median(), 'n': g.size()}); return w[w['n'] >= 300]
PER = [('Sep-Dec 2024', '2024-09-01', '2025-01-05'), ('Jan-Apr 2025', '2025-01-06', '2025-04-30'), ('May-Dec 2025', '2025-05-01', '2025-12-31'), ('Jan-May 2026', '2026-01-01', '2026-06-01')]
paper = {'pooled': [1573, 991, 886, 897], 'agnostic': [1573, 1163, 1675, 2357], 'bloxroute': [None, 1258, 1124, 1555]}
nbp = [14.8, 12.3, 9.8, 8.6]
for key, sub in [('pooled', df), ('agnostic', df[df.relay == 'agnostic']), ('bloxroute', df[df.relay == 'bloxroute'])]:
    w = weekly(sub)
    for (nm, a, b), pv in zip(PER, paper[key]):
        x = w[(w.index >= a) & (w.index <= b)]
        if pv is not None: chk(f'avg weekly HHI, {key}, {nm}', pv, x.hhi.mean(), 0)
    if key == 'agnostic':
        for (nm, a, b), pv in zip(PER, nbp):
            chk(f'avg weekly median builders/slot, agnostic, {nm}', pv, w[(w.index >= a) & (w.index <= b)].nb.mean(), 1)
wk = df.groupby('week')
chk('pooled HHI, week of 2024-12-30', 1767, hhi(wk.get_group(pd.Timestamp('2024-12-30'))['builder_pubkey']), 0)
chk('pooled HHI, week of 2025-01-06', 1074, hhi(wk.get_group(pd.Timestamp('2025-01-06'))['builder_pubkey']), 0)
chk('share of sample from Agnostic + BloXroute (%)', 87, 100 * df.relay.isin(['agnostic', 'bloxroute']).mean(), 0)

# ---- top-bid file statistics: every slot cut at the delivered (winning) bid ----
import os
if os.path.exists(paths.LEAD):
    chk('same builder holds the two highest recorded rows (%)', 41.6, 100 * pd.read_parquet(paths.LEAD, columns=['same']).same.dropna().mean(), 1)
if os.path.exists(paths.CUT):
    c = pd.read_parquet(paths.CUT)
    full = pd.read_parquet(paths.SRC, columns=['slot', 'value_eth']).drop_duplicates('slot', keep='first')
    mm = full.merge(c, on='slot', how='inner'); mm['lead'] = 100 * mm.lead_cut.astype('float64'); mm['incr'] = 100 * mm.incr_cut.astype('float64')
    chk('slots matched to the bid archive', 1219842, len(mm))
    chk('slot maximum above the delivered value (% of slots)', 96.2, 100 * (mm.top_ratio > 1 + 1e-6).mean(), 1); chk('slot maximum / delivered value, median', 1.083, mm.top_ratio.median(), 3)
    chk('rows above the delivered value (% of all rows)', 15.4, 100 * mm.n_higher.sum() / (mm.n_higher.sum() + mm.n_cut.sum()), 1)
    chk('winning block is a recorded row (% of slots)', 28.8, 100 * mm.has_win.mean(), 1); chk('slots where the winning block is a recorded row', 351545, int(mm.has_win.sum()))
    w = mm[mm.has_win == 1]
    chk('winning bid time, median (ms)', 957, w.wt.median(), 0); chk('winning bid time, p10 (ms)', 419, w.wt.quantile(.1), 0); chk('winning bid time, p90 (ms)', 2370, w.wt.quantile(.9), 0); chk('winning bid time, mean (ms)', 1104, w.wt.mean(), 0)
    chk('first row above the winning bid, median (ms)', 1067, mm.t_above.median(), 0)
    chk('updates at or below the winning bid, median', 112, mm.n_cut.median(), 0); chk('updates, mean', 119.7, mm.n_cut.mean(), 1); chk('updates, p10', 13, mm.n_cut.quantile(.1), 0); chk('updates, p90', 213, mm.n_cut.quantile(.9), 0); chk('updates, max', 5665, mm.n_cut.max(), 0)
    chk('builders setting a top bid, median', 9, mm.nb_cut.median(), 0); chk('builders, mean', 9.8, mm.nb_cut.mean(), 1); chk('builders, p10', 5, mm.nb_cut.quantile(.1), 0); chk('builders, p90', 15, mm.nb_cut.quantile(.9), 0)
    chk('relays, mean', 1.7, mm.nr_cut.mean(), 1); chk('relays, p90', 3, mm.nr_cut.quantile(.9), 0)
    chk('increment margin: slots', 1217330, int(mm.incr.notna().sum())); chk('increment margin mean (%)', 2.652, mm.incr.mean(), 3); chk('increment margin median (%)', 0.187, mm.incr.median(), 3); chk('increment margin p90 (%)', 5.279, mm.incr.quantile(.9), 3)
    chk('lead margin: slots', 1216743, int(mm.lead.notna().sum())); chk('lead margin mean (%)', 4.726, mm.lead.mean(), 3); chk('lead margin median (%)', 0.437, mm.lead.median(), 3); chk('lead margin p10 (%)', 0.003, mm.lead.quantile(.1), 3); chk('lead margin p90 (%)', 12.809, mm.lead.quantile(.9), 3)
    chk('no recorded row below the winning bid (slots)', 2512, int(mm.incr.isna().sum())); chk('no other-builder row below the winning bid (slots)', 3099, int(mm.lead.isna().sum()))
    chk('lead margin median, winning block recorded (%)', 0.437, mm.lead[mm.has_win == 1].median(), 3); chk('lead margin median, winning block not recorded (%)', 0.436, mm.lead[mm.has_win == 0].median(), 3)
    mm['ter'] = pd.qcut(mm.value_eth, 3, labels=['low', 'mid', 'high']); g = mm.groupby('ter', observed=True)
    for nm, pl, pb, pi, pu, pm in [('low', 0.53, 8, 0.264, 93, 99.74), ('mid', 0.45, 10, 0.182, 116, 99.82), ('high', 0.35, 11, 0.141, 133, 99.86)]:
        chk(f'lead margin median, {nm}-value tercile (%)', pl, g.lead.median()[nm], 2); chk(f'best competing bid / winning bid, {nm} tercile (%)', round(100 - pl, 2), 100 - g.lead.median()[nm], 2)
        chk(f'builders median, {nm} tercile', pb, g.nb_cut.median()[nm], 0); chk(f'increment margin median, {nm} tercile (%)', pi, g.incr.median()[nm], 3); chk(f'updates median, {nm} tercile', pu, g.n_cut.median()[nm], 0)
        chk(f'previous top bid / winning bid, {nm} tercile (%)', pm, 100 - g.incr.median()[nm], 2)
    chk('best competing bid / winning bid, median (%)', 99.56, 100 - mm.lead.median(), 2); chk('best competing bid / winning bid, one slot in ten below (%)', 87.2, 100 - mm.lead.quantile(.9), 1); chk('previous top bid / winning bid, median (%)', 99.81, 100 - mm.incr.median(), 2)
    spn = lambda a, b: sp(mm[a].to_numpy(dtype='float64'), mm[b].to_numpy(dtype='float64'))
    chk('Spearman block value vs lead margin', -0.013, spn('value_eth', 'lead'), 3); chk('Spearman block value vs increment margin', -0.066, spn('value_eth', 'incr'), 3)
    chk('Spearman block value vs builders', 0.283, spn('value_eth', 'nb_cut'), 3); chk('Spearman block value vs updates', 0.211, spn('value_eth', 'n_cut'), 3)
    chk('Spearman updates vs lead margin', -0.400, spn('n_cut', 'lead'), 3); chk('Spearman relay count vs lead margin', 0.182, spn('nr_cut', 'lead'), 3); chk('Spearman relay count vs increment margin', 0.134, spn('nr_cut', 'incr'), 3)
    chk('Spearman block value vs winning bid time', 0.085, sp(w.value_eth.to_numpy(), w.wt.to_numpy(dtype='float64')), 3)
    wt_t = w.groupby(pd.qcut(mm.value_eth, 3, labels=['low', 'mid', 'high']).loc[w.index], observed=True).wt.median(); chk('winning bid time median, low tercile (ms)', 935, wt_t['low'], 0); chk('winning bid time median, high tercile (ms)', 970, wt_t['high'], 0)
    rt = mm.groupby('nr_cut').agg(lead=('lead', 'median'), incr=('incr', 'median'), upd=('n_cut', 'median'), n=('slot', 'size'))
    for r_, pl, pi, pu, pn in [(1, 0.262, 0.122, 125, 645737), (2, 0.559, 0.273, 103, 364997), (3, 1.288, 0.352, 77, 165229), (4, 3.352, 0.413, 22, 37299), (5, 2.742, 0.353, 60, 3764), (6, 2.078, 0.316, 80, 375), (7, 1.729, 0.194, 40, 33)]:
        chk(f'{r_} relay(s): lead margin median (%)', pl, rt.lead[r_], 3); chk(f'{r_} relay(s): increment margin median (%)', pi, rt.incr[r_], 3); chk(f'{r_} relay(s): updates median', pu, rt.upd[r_], 0); chk(f'{r_} relay(s): slots', pn, rt.n[r_], 0)
    mm['nq'] = pd.qcut(mm.n_cut, 5, labels=[1, 2, 3, 4, 5]); pv = mm.pivot_table(index='nq', columns='ter', values='lead', aggfunc='median', observed=True)
    for q, row, allv in [(1, (3.78, 8.06, 10.53), 7.16), (2, (0.53, 0.38, 0.39), 0.46), (3, (0.28, 0.23, 0.20), 0.24), (4, (0.18, 0.17, 0.15), 0.16), (5, (0.18, 0.13, 0.12), 0.13)]:
        for t_, v_ in zip(['low', 'mid', 'high'], row): chk(f'lead margin median, density quintile {q}, {t_} value (%)', v_, pv.loc[q, t_], 2)
        chk(f'lead margin median, density quintile {q}, all (%)', allv, mm.lead[mm.nq == q].median(), 2)
else:
    print('data/winner_cutoff_by_slot.parquet not found: cutoff checks skipped')

out = pd.DataFrame(rows, columns=['statistic', 'paper', 'recomputed', 'result'])
import os as _os
REPORTED = set(open(_os.path.join(paths.ROOT, 'reference', 'reported_statistics.txt')).read().splitlines())   # the pipeline computes more than the paper reports
out = out[out.statistic.isin(REPORTED)].reset_index(drop=True)
print(out.to_string(index=False))
print(f"\n{(out.result == 'PASS').sum()} of {len(out)} match the paper.")
