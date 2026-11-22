"""Operator-level concentration and auction statistics from the archive's all-bids files. Source of the auction numbers in Sections 5.5 and 5.6.

Inputs (all shipped in data/):
  bid_auction_stats.parquet   delivered payloads (slot, relay, builder key, value)
  all_bids_by_slot.parquet    per delivered slot, from every recorded bid (7_all_bids/pbs_all_bids.ipynb, then 7_all_bids/relink.py)
  all_bids_keys.parquet       per builder key per day: bids, slots, wins, timing
  operator_map.csv            builder key -> operator under each linking rule
Operators (7_all_bids/label_operators.py):
  B  shared blocks only: keys that repeatedly submit the identical block (>= 0.2% of the smaller key's bids on the linked days)
  D  B plus pairs with >= 200 exact value ties and a tie rate >= 2%
  L  public builder labels plus shared blocks                                  (baseline)
  M  L with beaverbuild, BuilderNet and Flashbots as one operator
Per slot (winner = operator of the delivered block):
  stand_X  1 - (highest STANDING bid of another operator when the winning bid arrived) / winning value. A key's standing bid is its most recent bid.
  any_X    1 - (highest bid at or below the winning value from another operator, at any time) / winning value
  riv1_X   other operators whose standing bid is within 1% of the winning value
Writes figures/fig_ab_*.pdf and work/all_bids_numbers.csv."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import paths
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)

N = {}
def put(k, v, nd=3):
    N[k] = float(v); print(f'{k:74s} {float(v):,.{nd}f}')
hhi = lambda s: 10000 * ((s.value_counts(normalize=True)) ** 2).sum()
hhv = lambda v: 10000 * ((v / v.sum()) ** 2).sum()

dl = pd.read_parquet(paths.SRC, columns=['slot', 'relay', 'builder_pubkey', 'value_eth']).drop_duplicates('slot', keep='first').sort_values('slot').reset_index(drop=True)
dl['t'] = pd.to_datetime(1606824023 + 12 * dl.slot, unit='s'); dl['week'] = dl.t.dt.to_period('W-SUN').dt.start_time; dl['q'] = dl.t.dt.to_period('Q').astype(str)
om = pd.read_csv(paths.OPS).set_index('pubkey')
for X in ['B', 'D', 'L', 'M']: dl['op' + X] = dl.builder_pubkey.map(om[X]).fillna(-1).astype(int); dl.loc[dl['op' + X] < 0, 'op' + X] = -1 - dl.builder_pubkey[dl['op' + X] < 0].factorize()[0]
ab = pd.read_parquet(paths.AB); ab['day'] = ab.day.astype(str); K = pd.read_parquet(paths.ABK)
d = dl.merge(ab, on='slot', how='inner'); d['ter'] = pd.qcut(d.value_eth, 3, labels=['Low', 'Mid', 'High'])
w = d[d.has_win == 1]
PER = [('Sep-Dec 2024', '2024-09-01', '2025-01-05'), ('Jan-Apr 2025', '2025-01-06', '2025-04-30'), ('May-Dec 2025', '2025-05-01', '2025-12-31'), ('Jan-May 2026', '2026-01-01', '2026-06-01')]

print('=== 1. concentration of delivered blocks: by key and by operator ===')
put('delivered slots', len(dl), 0); put('builder keys with a delivered block', dl.builder_pubkey.nunique(), 0)
put('whole-sample HHI by key', hhi(dl.builder_pubkey), 0)
for X in ['B', 'D', 'L', 'M']:
    put(f'operators with a delivered block ({X})', dl['op' + X].nunique(), 0); put(f'whole-sample HHI by operator ({X})', hhi(dl['op' + X]), 0)
sh = dl.opL.value_counts(normalize=True); put('largest operator share (L, %)', 100 * sh.iloc[0], 1); put('top-3 operator share (L, %)', 100 * sh.head(3).sum(), 1); put('top-5 operator share (L, %)', 100 * sh.head(5).sum(), 1)
put('top-5 KEY share (%)', 100 * dl.builder_pubkey.value_counts(normalize=True).head(5).sum(), 1)
nk = om.L.value_counts(); top5 = sh.head(5).index.tolist(); put('keys of the largest operator (L)', nk.get(top5[0], 1), 0)
for i, o in enumerate(top5): put(f'operator {i + 1}: share of delivered blocks (%)', 100 * sh[o], 1); put(f'operator {i + 1}: keys', nk.get(o, 1), 0)
tab = {}
for lab, sub in [('pooled', dl), ('agnostic', dl[dl.relay == 'agnostic']), ('bloxroute', dl[dl.relay == 'bloxroute'])]:
    g = sub.groupby('week'); wk = pd.DataFrame({'key': g.builder_pubkey.apply(hhi), 'opB': g.opB.apply(hhi), 'opL': g.opL.apply(hhi), 'opM': g.opM.apply(hhi), 'top1': g.opL.apply(lambda s: s.value_counts(normalize=True).iloc[0]), 'n': g.size()}); wk = wk[wk.n >= 300]
    tab[lab] = wk
    for nm, a, b in PER:
        x = wk[(wk.index >= a) & (wk.index <= b)]
        if len(x):
            put(f'{lab} avg weekly HHI by key, {nm}', x.key.mean(), 0); put(f'{lab} avg weekly HHI by operator, {nm}', x.opL.mean(), 0); put(f'{lab} avg weekly HHI by operator (B), {nm}', x.opB.mean(), 0); put(f'{lab} avg weekly HHI by operator (M), {nm}', x.opM.mean(), 0)
            put(f'{lab} avg weekly share of the largest operator, {nm} (%)', 100 * x.top1.mean(), 1)
put('pooled weekly operator HHI: min', tab['pooled'].opL.min(), 0); put('pooled weekly operator HHI: max', tab['pooled'].opL.max(), 0); put('pooled weekly operator HHI: median', tab['pooled'].opL.median(), 0)
put('share of weeks with pooled operator HHI above 2,500 (%)', 100 * (tab['pooled'].opL > 2500).mean(), 1)
put('pooled operator HHI, week of 2024-12-30', tab['pooled'].opL.get(pd.Timestamp('2024-12-30'), np.nan), 0); put('pooled operator HHI, week of 2025-01-06', tab['pooled'].opL.get(pd.Timestamp('2025-01-06'), np.nan), 0)
qs = dl.groupby('q'); print(pd.DataFrame({'hhi_key': qs.builder_pubkey.apply(hhi), 'hhi_op': qs.opL.apply(hhi), 'top1': qs.opL.apply(lambda s: s.value_counts(normalize=True).iloc[0]), 'top3': qs.opL.apply(lambda s: s.value_counts(normalize=True).head(3).sum()), 'ops': qs.opL.nunique(), 'keys': qs.builder_pubkey.nunique()}).round(3).to_string())

print('\n=== 2. the auction from all bids ===')
put('matched slots', len(d), 0); put('archive days read', d.day.nunique(), 0); put('winning block found among the recorded bids (%)', 100 * d.has_win.mean(), 1); put('slots with the winning block found', int(d.has_win.sum()), 0)
q = lambda s: dict(mean=s.mean(), median=s.median(), p10=s.quantile(.1), p90=s.quantile(.9))
S = pd.DataFrame({'bids': q(d.n_all), 'keys': q(d.nb_all), 'operators': q(d.nop_all), 'stand %': q(100 * w.stand_L.astype('float64')), 'any %': q(100 * w.any_L.astype('float64')), 'winning bid ms': q(w.wt), 'next bid gap ms': q(w.wnext_gap)}).T; print(S.round(3).to_string())
for k in S.index:
    for c in S.columns: N[f'summary {k} {c}'] = float(S.loc[k, c])
for X in ['B', 'D', 'L', 'M', 'KEY']:
    s_ = w['stand_' + X].astype('float64'); a_ = w['any_' + X].astype('float64')
    put(f'median standing margin ({X}, %)', 100 * s_.median()); put(f'no rival within 1% ({X}, % of slots)', 100 * (w['riv1_' + X] == 0).mean(), 1) if X != 'KEY' else None; put(f'median margin over any lower rival bid ({X}, %)', 100 * a_.median(), 4)
    put(f'a rival operator had a higher standing bid ({X}, % of slots)', 100 * (s_.dropna() < 0).mean(), 1); put(f'... higher by more than 1% ({X}, % of slots)', 100 * (s_.dropna() < -.01).mean(), 1)
s_ = w.stand_L.astype('float64').dropna()
put('winner more than 1% ahead of every rival operator (% of slots)', 100 * (s_ > .01).mean(), 1); put('winner more than 5% ahead (% of slots)', 100 * (s_ > .05).mean(), 1)
put('standing margin p25 (%)', 100 * s_.quantile(.25)); put('standing margin p75 (%)', 100 * s_.quantile(.75)); put('standing margin p5 (%)', 100 * s_.quantile(.05)); put('standing margin p95 (%)', 100 * s_.quantile(.95))
put('no rival operator within 1% (% of slots)', 100 * (w.riv1_L == 0).mean(), 1); put('exactly one rival within 1% (% of slots)', 100 * (w.riv1_L == 1).mean(), 1); put('two or more rivals within 1% (% of slots)', 100 * (w.riv1_L >= 2).mean(), 1); put('rival operators within 1%, mean', w.riv1_L.mean(), 2)
put('no rival operator among the candidates (% of slots)', 100 * w.stand_L.isna().mean(), 2)
d['q'] = d.q.astype(str); g = d.groupby('q'); gw = w.groupby(w.q.astype(str))
Q = pd.DataFrame({'slots': g.size(), 'bids': g.n_all.median(), 'keys': g.nb_all.median(), 'ops': g.nop_all.median(), 'stand_L': 100 * gw.stand_L.median(), 'stand_B': 100 * gw.stand_B.median(), 'stand_M': 100 * gw.stand_M.median(), 'any_L': 100 * gw.any_L.median(),
                  'rival_higher': 100 * gw.stand_L.apply(lambda v: (v.dropna() < 0).mean()), 'riv1': gw.riv1_L.mean()}); print('\nby quarter:'); print(Q.round(3).to_string())
for qq in Q.index:
    for c in Q.columns: N[f'quarter {qq} {c}'] = float(Q.loc[qq, c])
g = d.groupby('ter', observed=True); gw = w.groupby('ter', observed=True)
V = pd.DataFrame({'ops': g.nop_all.median(), 'keys': g.nb_all.median(), 'bids': g.n_all.median(), 'stand_L': 100 * gw.stand_L.median(), 'any_L': 100 * gw.any_L.median(), 'rival_higher': 100 * gw.stand_L.apply(lambda v: (v.dropna() < 0).mean()), 'riv1': gw.riv1_L.mean(), 'wt': gw.wt.median()}); print('\nby value tercile:'); print(V.round(3).to_string())
for t in V.index:
    for c in V.columns: N[f'tercile {t} {c}'] = float(V.loc[t, c])
def sp(a, b):
    ok = np.isfinite(a) & np.isfinite(b); return stats.spearmanr(a[ok], b[ok])[0]
rng = np.random.default_rng(1)
def boot(a, b, n=8000, B=1000):
    ok = np.isfinite(a) & np.isfinite(b); a = np.asarray(a[ok]); b = np.asarray(b[ok]); r = [stats.spearmanr(a[i], b[i])[0] for i in (rng.choice(len(a), n, replace=False) for _ in range(B))]; return np.percentile(r, 2.5), np.percentile(r, 97.5)
for nm, (a, b) in {'value ~ operators bidding': (d.value_eth, d.nop_all.astype('float64')), 'value ~ keys bidding': (d.value_eth, d.nb_all.astype('float64')), 'value ~ standing margin': (w.value_eth, w.stand_L.astype('float64')), 'value ~ bids': (d.value_eth, d.n_all.astype('float64')),
               'operators bidding ~ standing margin': (w.nop_all.astype('float64'), w.stand_L.astype('float64')), 'value ~ winning bid time': (w.value_eth, w.wt.astype('float64'))}.items():
    put('rho ' + nm, sp(a, b)); lo, hi = boot(a, b); put('rho ' + nm + ' CI low', lo); put('rho ' + nm + ' CI high', hi)

print('\n=== 3. over time ===')
d['wk'] = d.t.dt.to_period('W-SUN').dt.start_time; wm = d.groupby('wk').agg(ops=('nop_all', 'median'), keys=('nb_all', 'median'), bids=('n_all', 'median'), n=('slot', 'size')); wm['stand'] = 100 * d[d.has_win == 1].groupby('wk').stand_L.median(); wm = wm[wm.n >= 300]
wm.to_csv(paths.WORK + 'weekly_all_bids.csv')
for lab, sub in [('pooled', d), ('agnostic', d[d.relay == 'agnostic'])]:
    ww = sub.groupby('wk').agg(ops=('nop_all', 'median'), keys=('nb_all', 'median'), n=('slot', 'size')); ww = ww[ww.n >= 300]
    for nm, a, b in PER: put(f'{lab} operators bidding per slot, {nm}', ww[(ww.index >= a) & (ww.index <= b)].ops.mean(), 1); put(f'{lab} keys bidding per slot, {nm}', ww[(ww.index >= a) & (ww.index <= b)].ops.mean() * 0 + ww[(ww.index >= a) & (ww.index <= b)]['keys'].mean(), 1)
put('weekly median operators per slot: first 8 weeks, mean', wm.ops.head(8).mean(), 1); put('weekly median operators per slot: last 8 weeks, mean', wm.ops.tail(8).mean(), 1)
put('weekly median standing margin: min (%)', wm.stand.min()); put('weekly median standing margin: max (%)', wm.stand.max()); put('weekly median bids: min', wm.bids.min(), 0); put('weekly median bids: max', wm.bids.max(), 0)
ko = K.assign(o=K.pubkey.map(om.L), q=pd.PeriodIndex(pd.to_datetime(K.day), freq='Q').astype(str))
od = ko.groupby(['q', 'day']).o.nunique().groupby('q').median(); kd = ko.groupby(['q', 'day']).pubkey.nunique().groupby('q').median()
for qq in od.index: put(f'operators bidding per day, {qq}', od[qq], 0); put(f'keys bidding per day, {qq}', kd[qq], 0)
j = wm.join(tab['pooled'][['opL']].rename(columns={'opL': 'hhi'}), how='inner').sort_index(); j['hhi_lag'] = j.hhi.shift(1); jj = j.dropna(subset=['hhi_lag', 'stand'])
r, p = stats.spearmanr(jj.hhi_lag, jj.stand); put('weeks in the HHI-margin test', len(jj), 0); put('lagged operator HHI ~ weekly median standing margin: rho', r); put('lagged operator HHI ~ weekly median standing margin: p', p, 4)
r, p = stats.spearmanr(jj.hhi, jj.stand); put('same-week operator HHI ~ weekly median standing margin: rho', r); put('same-week: p', p, 4)

print('\n=== 4. operators ===')
put('builder keys bidding', K.pubkey.nunique(), 0); put('operators bidding (L)', ko.o.nunique(), 0); put('operators bidding (B)', K.pubkey.map(om.B).nunique(), 0); put('operators bidding (D)', K.pubkey.map(om.D).nunique(), 0); put('operators bidding (M)', K.pubkey.map(om.M).nunique(), 0)
nkL = om.L.value_counts(); put('operators with more than one key (L)', int((nkL > 1).sum()), 0); put('keys held by multi-key operators (L)', int(nkL[nkL > 1].sum()), 0)
o = ko.groupby('o').agg(keys=('pubkey', 'nunique'), bids=('bids', 'sum'), wins=('wins', 'sum'), days=('day', 'nunique'))
for c in ['t_median', 'after_start', 'above_winner']: o[c] = ko.assign(z=ko[c].fillna(0) * ko.bids).groupby('o').z.sum() / o.bids
o['bid_share'] = 100 * o.bids / o.bids.sum(); o['win_share'] = 100 * o.wins / o.wins.sum(); o['wins_per_1000'] = 1000 * o.wins / o.bids; o = o.sort_values('wins', ascending=False)
put('operators with at least one win (matched slots)', int((o.wins > 0).sum()), 0); put('bid share of the five largest winners (%)', o.head(5).bid_share.sum(), 1); put('win share of the five largest winners (%)', o.head(5).win_share.sum(), 1)
nm_ = om.dropna(subset=['label']).groupby('L').label.first(); o['name'] = o.index.map(nm_); extra = [i for i in o.index if o.name[i] in ('bobthebuilder', 'Eureka Builder', 'BTCS')]
OT = pd.concat([o.head(5), o.loc[extra]]); OT = OT[~OT.index.duplicated()]; print(OT.round(3).to_string())
for i in OT.index:
    for c in ['keys', 'bid_share', 'win_share', 'wins_per_1000', 't_median', 'after_start', 'above_winner', 'days']: N[f'operator {OT.name[i]} {c}'] = float(OT.loc[i, c])
kd_ = ko[ko.o.isin(OT.index)].groupby(['o', 'pubkey']).day.nunique().groupby('o').median()
for i in OT.index: put(f'operator {OT.name[i]} median days a key is active', kd_[i], 0)
first = ko.groupby('o').day.min()
for i in OT.index: print(OT.name[i], 'first bid', first[i])
qs_ = dl.assign(nm=dl.opL.map(nm_)).groupby(['q', 'nm']).size().unstack(1).reindex(columns=['Titan', 'buildernet', 'beaverbuild.org', 'Quasar', 'rsync']).fillna(0); qs_ = 100 * qs_.div(dl.groupby('q').size(), axis=0); print(qs_.round(1).to_string())
for q_ in qs_.index:
    for c in qs_.columns: N[f'share {c} {q_}'] = float(qs_.loc[q_, c])
top3q = dl.groupby('q').opL.apply(lambda v: v.value_counts(normalize=True).head(3).sum()); top2m = dl.groupby('q').opM.apply(lambda v: v.value_counts(normalize=True).head(2).sum()); hm = dl.groupby('q').opM.apply(hhi)
for q_ in top3q.index: put(f'top-3 share {q_} (L, %)', 100 * top3q[q_], 1); put(f'top-2 share {q_} (M, %)', 100 * top2m[q_], 1); put(f'HHI {q_} (M)', hm[q_], 0); put(f'HHI {q_} (L)', dl[dl.q == q_].opL.pipe(hhi), 0)
shm = dl.opM.value_counts(normalize=True); put('top-2 share (M, %)', 100 * shm.head(2).sum(), 1); put('top-3 share (M, %)', 100 * shm.head(3).sum(), 1); put('largest operator share (M, %)', 100 * shm.iloc[0], 1)
V_ = pd.read_csv(paths.WORK + 'label_validation.csv', index_col=0) if os.path.exists(paths.WORK + 'label_validation.csv') else None
if V_ is not None:
    for X in V_.index:
        for c in V_.columns: N[f'validation {X} {c}'] = float(V_.loc[X, c])
pd.Series(N).to_csv(paths.WORK + 'all_bids_numbers.csv', header=['value'])

# ------------------------------------------------------------------ figures
F = paths.FIG; plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}); C1 = 'steelblue'
def med(ax, v, fmt): ax.axvline(v, color='crimson', lw=1.2, ls='--'); ax.text(0.97, 0.93, 'median ' + fmt.format(v), transform=ax.transAxes, ha='right', color='crimson')
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
ax[0].hist(d.n_all.clip(upper=10000), bins=100, color=C1); med(ax[0], d.n_all.median(), '{:,.0f}'); ax[0].set_xlabel('Bids per slot'); ax[0].set_ylabel('Slots')
ax[1].hist(d.nb_all, bins=np.arange(-0.5, 96.5), color=C1); med(ax[1], d.nb_all.median(), '{:.0f}'); ax[1].set_xlabel('Builder keys bidding per slot')
ax[2].hist(d.nop_all, bins=np.arange(-0.5, 47.5), color=C1); med(ax[2], d.nop_all.median(), '{:.0f}'); ax[2].set_xlabel('Operators bidding per slot')
plt.tight_layout(); plt.savefig(F + 'fig_ab_overview.pdf', bbox_inches='tight'); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4)); x = 100 * w.stand_L.astype('float64').dropna()
ax[0].hist(x.clip(-10, 15), bins=125, color=C1); ax[0].axvline(0, color='k', lw=0.7); med(ax[0], x.median(), '{:.2f}%'); ax[0].set_xlabel('Winning bid minus best standing bid of another operator (% of winning bid), clipped'); ax[0].set_ylabel('Slots')
for X, st, lab in [('L', '-', 'Labels + shared blocks (baseline)'), ('B', '--', 'Shared blocks only'), ('M', ':', 'BuilderNet partners merged'), ('KEY', '-.', 'By key')]: ax[1].plot(range(len(Q)), 100 * w.groupby(w.q.astype(str))['stand_' + X].median().reindex(Q.index), ls=st, marker='o', ms=3, label=lab)
ax[1].set_xticks(range(len(Q))); ax[1].set_xticklabels([s[2:4] + 'Q' + s[-1] for s in Q.index]); ax[1].set_ylabel('Median standing margin (%)'); ax[1].axhline(0, color='k', lw=0.6); ax[1].legend(frameon=False)
plt.tight_layout(); plt.savefig(F + 'fig_ab_margin.pdf', bbox_inches='tight'); plt.close()

fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
ax[0].plot(wm.index, wm.ops, color=C1, label='Operators'); ax[0].plot(wm.index, wm['keys'], color='darkorange', label='Builder keys'); ax[0].set_ylabel('Bidding per slot\n(weekly median)'); ax[0].legend(frameon=False)
ax[1].plot(wm.index, wm.stand, color=C1); ax[1].set_ylabel('Standing margin\n(weekly median, %)'); ax[1].axhline(0, color='k', lw=0.6)
ax[2].plot(wm.index, wm.bids, color=C1); ax[2].set_ylabel('Bids per slot\n(weekly median)')
plt.tight_layout(); plt.savefig(F + 'fig_ab_timeseries.pdf', bbox_inches='tight'); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4)); dl['m'] = dl.t.dt.to_period('M').dt.start_time
PRETTY = {'buildernet': 'BuilderNet', 'beaverbuild.org': 'beaverbuild'}; names = {o_: PRETTY.get(nm_[o_], nm_[o_]) for o_ in top5}; sm = pd.crosstab(dl.m, dl.opL.map(names).fillna('All others'), normalize='index')[[names[o_] for o_ in top5] + ['All others']]
ax[0].stackplot(sm.index, (100 * sm).T.to_numpy(), labels=sm.columns, colors=['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#d9d9d9']); ax[0].set_ylabel('Share of delivered blocks (%)'); ax[0].set_ylim(0, 100); ax[0].legend(frameon=False, fontsize=8, loc='lower left', ncol=2)
mq = dl.groupby('m'); ax[1].plot(mq.builder_pubkey.apply(hhi), color='darkorange', label='By builder key'); ax[1].plot(mq.opL.apply(hhi), color=C1, label='By operator'); ax[1].plot(mq.opM.apply(hhi), color=C1, ls=':', label='By operator, BuilderNet partners merged'); ax[1].axhline(2500, color='k', lw=0.6, ls=':'); ax[1].set_ylabel('Monthly HHI of delivered blocks'); ax[1].legend(frameon=False)
plt.tight_layout(); plt.savefig(F + 'fig_ab_operators.pdf', bbox_inches='tight'); plt.close()

fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for lab, col in [('pooled', 'k'), ('agnostic', C1), ('bloxroute', 'darkorange')]:
    ax[0].plot(tab[lab].index, tab[lab].opL, color=col, label={'pooled': 'All relays pooled', 'agnostic': 'Agnostic only', 'bloxroute': 'BloXroute only'}[lab])
ax[0].plot(tab['pooled'].index, tab['pooled'].key, color='grey', ls=':', label='Pooled, by builder key'); ax[0].set_ylabel('Weekly HHI by operator'); ax[0].legend(frameon=False, ncol=2, fontsize=8)
for lab, sub, col in [('pooled', d, 'k'), ('agnostic', d[d.relay == 'agnostic'], C1), ('bloxroute', d[d.relay == 'bloxroute'], 'darkorange')]:
    ww = sub.groupby('wk').agg(ops=('nop_all', 'median'), n=('slot', 'size')); ww = ww[ww.n >= 300]; ax[1].plot(ww.index, ww.ops, color=col)
ax[1].set_ylabel('Operators bidding per slot\n(weekly median)')
for a in ax: a.axvline(pd.Timestamp('2025-01-06'), color='grey', ls='--', lw=0.8)
plt.tight_layout(); plt.savefig(F + 'fig_hhi_by_relay.pdf', bbox_inches='tight'); plt.close()
print('\nfigures written to', F)
