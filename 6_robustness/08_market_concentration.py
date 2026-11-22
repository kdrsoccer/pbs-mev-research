"""Concentration on ALL MEV-Boost blocks (every relay), and how the paper's five-relay sample differs from it.

data/market_delivered.parquet: one row per slot with an MEV-Boost delivered payload on any relay, from Data Always mevboost-data
(slot, builder_pubkey, builder_label, n_relays), 10 Sep 2024 - 2 Jun 2026.  The paper's own delivered-payload sample (five relays, 31% of
these slots) over-represents blocks delivered through BloXroute and Agnostic, and so mis-states builder shares; this script measures by how much,
and re-weights the auction margins to market-wide winner shares.
Writes work/market_numbers.csv, figures/fig_market_operators.pdf."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import paths
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
pd.set_option('display.width', 230); pd.set_option('display.max_rows', 200)
N = {}
def put(k, v, nd=1): N[k] = float(v); print(f'{k:84s} {float(v):,.{nd}f}')
hhi = lambda s: 10000 * (s.value_counts(normalize=True) ** 2).sum()
D = paths.DATA; dl = pd.read_parquet(paths.SRC, columns=['slot', 'relay', 'builder_pubkey']).drop_duplicates('slot'); om = pd.read_csv(paths.OPS).set_index('pubkey')
m = pd.read_parquet(D + 'market_delivered.parquet'); m = m[(m.slot >= dl.slot.min()) & (m.slot <= dl.slot.max())].copy(); m['builder_pubkey'] = m.builder_pubkey.astype(str)
nm = om.dropna(subset=['label']).groupby('L').label.first(); PRETTY = {'buildernet': 'BuilderNet', 'beaverbuild.org': 'beaverbuild'}; BIG = ['Titan', 'BuilderNet', 'beaverbuild', 'Quasar', 'rsync']
for df in (m, dl):
    for X in ['B', 'D', 'L', 'M']: df['op' + X] = df.builder_pubkey.map(om[X])
    df['name'] = df.opL.map(nm).replace(PRETTY); df['g'] = df.name.where(df.name.isin(BIG), 'Other'); df['t'] = pd.to_datetime(1606824023 + 12 * df.slot, unit='s'); df['q'] = df.t.dt.to_period('Q').astype(str); df['week'] = df.t.dt.to_period('W-SUN').dt.start_time; df['mo'] = df.t.dt.to_period('M').dt.start_time
nslots = int(dl.slot.max() - dl.slot.min() + 1)
print('=== market: every MEV-Boost block ===')
put('MEV-Boost slots, all relays', len(m), 0); put('share of all slots in the period (%)', 100 * len(m) / nslots); put('sample share of MEV-Boost slots (%)', 100 * len(dl) / len(m)); put('sample slots found in the market set (%)', 100 * dl.slot.isin(m.slot).mean(), 2)
put('builder keys with a delivered block, market', m.builder_pubkey.nunique(), 0); put('blocks whose key is not in the operator map', (~m.builder_pubkey.isin(om.index)).sum(), 0); put('operators with a delivered block, market (L)', m.opL.nunique(), 0); put('blocks whose key has no label (%)', 100 * m.name.isna().mean(), 2)
put('market HHI by key', hhi(m.builder_pubkey), 0)
for X, lab in [('B', 'shared blocks only'), ('D', 'shared blocks and value ties'), ('L', 'labels and shared blocks'), ('M', 'BuilderNet partners merged')]: put(f'market HHI by operator ({lab})', hhi(m['op' + X]), 0)
put('market top-5 key share (%)', 100 * m.builder_pubkey.value_counts(normalize=True).head(5).sum())
sh = m.opL.value_counts(normalize=True); put('market top-1 share (L, %)', 100 * sh.iloc[0]); put('market top-2 share (L, %)', 100 * sh.head(2).sum()); put('market top-3 share (L, %)', 100 * sh.head(3).sum()); put('market top-5 share (L, %)', 100 * sh.head(5).sum())
shm = m.opM.value_counts(normalize=True); put('market top-1 share (M, %)', 100 * shm.iloc[0]); put('market top-2 share (M, %)', 100 * shm.head(2).sum()); put('market top-3 share (M, %)', 100 * shm.head(3).sum())
for b in BIG: put(f'market share {b} (%)', 100 * (m.name == b).mean()); put(f'sample share {b} (%)', 100 * (dl.name == b).mean())
put('sample HHI by operator (L)', hhi(dl.opL), 0); put('sample HHI by key', hhi(dl.builder_pubkey), 0)
w = m.groupby('week'); wk = pd.DataFrame({'key': w.builder_pubkey.apply(hhi), 'L': w.opL.apply(hhi), 'M': w.opM.apply(hhi), 'B': w.opB.apply(hhi), 'n': w.size()}); wk = wk[wk.n >= 20000]
put('weeks', len(wk), 0); put('weekly market HHI by operator: min', wk.L.min(), 0); put('weekly market HHI by operator: median', wk.L.median(), 0); put('weekly market HHI by operator: max', wk.L.max(), 0); put('weeks with market operator HHI above 2,500 (%)', 100 * (wk.L > 2500).mean())
put('weekly market HHI by key: min', wk.key.min(), 0); put('weekly market HHI by key: max', wk.key.max(), 0); put('weekly market HHI by key: median', wk.key.median(), 0); put('weeks with key-level HHI above 2,500 (%)', 100 * (wk.key > 2500).mean()); put('weeks with key-level HHI above 1,500 (%)', 100 * (wk.key > 1500).mean())
g = m.groupby('q'); Q = pd.DataFrame({'slots': g.size(), 'key': g.builder_pubkey.apply(hhi), 'B': g.opB.apply(hhi), 'L': g.opL.apply(hhi), 'M': g.opM.apply(hhi), 'top3': 100 * g.opL.apply(lambda v: v.value_counts(normalize=True).head(3).sum()), 'top2M': 100 * g.opM.apply(lambda v: v.value_counts(normalize=True).head(2).sum())})
S = (100 * pd.crosstab(m.q, m.g, normalize='index'))[BIG + ['Other']]; S2 = (100 * pd.crosstab(dl.q, dl.g, normalize='index')).reindex(columns=BIG + ['Other']).fillna(0); Q['sample_slots'] = dl.groupby('q').size(); Q['sample_L'] = dl.groupby('q').opL.apply(hhi)
print(Q.round(1).to_string()); print('market shares by quarter'); print(S.round(1).to_string()); print('sample shares by quarter'); print(S2.round(1).to_string())
for q in Q.index:
    for c in Q.columns: N[f'quarter {q} {c}'] = float(Q.loc[q, c])
    for b in BIG + ['Other']: N[f'market share {b} {q}'] = float(S.loc[q, b]); N[f'sample share {b} {q}'] = float(S2.loc[q, b])
first = m.groupby('name').t.min(); put('Quasar: first delivered block, days after sample start', (first['Quasar'] - m.t.min()).days, 0); print('Quasar first delivered block', first['Quasar'])

print('\n=== the auction sample re-weighted to market winner shares (quarter x winner) ===')
ab = pd.read_parquet(paths.AB, columns=['slot', 'has_win', 'stand_L', 'any_L', 'riv1_L', 'stand_M']); ab = ab.merge(dl[['slot', 'g', 'q']], on='slot'); a = ab[ab.has_win == 1].copy()
cs = a.groupby(['q', 'g']).size(); cm = m.groupby(['q', 'g']).size(); wt = (cm / cm.groupby('q').transform('sum')) / (cs / cs.groupby('q').transform('sum')); wq_ = cm.groupby('q').sum() / cm.sum() / (cs.groupby('q').sum() / cs.sum())
a['wgt'] = [wt.get((q, g), 0.0) * wq_[q] for q, g in zip(a.q, a.g)]
def wquant(v, wg, p=0.5):
    ok = v.notna().to_numpy(); x = v.to_numpy()[ok].astype('float64'); ww = wg.to_numpy()[ok]; o = np.argsort(x); c = np.cumsum(ww[o]); return x[o][np.searchsorted(c, p * c[-1])]
G = a.groupby('g').agg(slots=('slot', 'size'), median=('stand_L', lambda v: 100 * v.median()), higher=('stand_L', lambda v: 100 * (v.dropna() < 0).mean()), none=('riv1_L', lambda v: 100 * (v == 0).mean())).reindex(BIG + ['Other']); G['sample_share'] = 100 * G.slots / G.slots.sum(); G['market_share'] = [100 * (m.g == b).mean() for b in G.index]; print(G.round(2).to_string())
for b in G.index:
    for c in G.columns: N[f'winner {b} {c}'] = float(G.loc[b, c])
P = 100 * a.pivot_table(index='q', columns='g', values='stand_L', aggfunc='median'); print(P.round(2).to_string())
for q in P.index:
    for b in ['Titan', 'BuilderNet', 'Quasar', 'beaverbuild']: N[f'winner {b} {q} median'] = float(P.loc[q, b])
put('median standing margin, sample (%)', 100 * a.stand_L.median(), 3); put('median standing margin, re-weighted (%)', 100 * wquant(a.stand_L, a.wgt), 3)
ok = a.stand_L.notna(); put('rival live bid higher, re-weighted (%)', 100 * np.average(a.stand_L[ok] < 0, weights=a.wgt[ok])); put('no rival within 1%, re-weighted (%)', 100 * np.average(a.riv1_L == 0, weights=a.wgt)); put('median margin over any lower rival bid, re-weighted (%)', 100 * wquant(a.any_L, a.wgt), 3)
put('median standing margin, re-weighted, partners merged (%)', 100 * wquant(a.stand_M, a.wgt), 3)
rq = pd.DataFrame({'sample': a.groupby('q').stand_L.median() * 100, 'reweighted': [100 * wquant(a.stand_L[a.q == q], a.wgt[a.q == q]) for q in sorted(a.q.unique())]}); print(rq.round(3).to_string())
for q in rq.index: N[f'quarter {q} margin reweighted'] = float(rq.loc[q, 'reweighted'])
a['week'] = pd.to_datetime(1606824023 + 12 * a.slot, unit='s').dt.to_period('W-SUN').dt.start_time; wmar = a.groupby('week').agg(st=('stand_L', 'median'), n=('slot', 'size')); wmar = wmar[wmar.n >= 300]
j = wmar.join(wk[['L']], how='inner').sort_index(); j['lag'] = j.L.shift(1); jj = j.dropna(); r, p = stats.spearmanr(jj.lag, jj.st); put('weeks in the HHI-margin test', len(jj), 0); put('lagged market operator HHI ~ weekly median margin: rho', r, 3); put('lagged market operator HHI ~ weekly median margin: p', p, 3)
r, p = stats.spearmanr(j.L, j.st); put('same-week market operator HHI ~ weekly median margin: rho', r, 3); put('same-week: p', p, 3)
pd.Series(N).to_csv(paths.WORK + 'market_numbers.csv', header=['value'])
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}); fig, ax = plt.subplots(1, 3, figsize=(15, 4)); COL = ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#d9d9d9']
sm = pd.crosstab(m.mo, m.g, normalize='index')[BIG + ['Other']]; ax[0].stackplot(sm.index, (100 * sm).T.to_numpy(), labels=sm.columns, colors=COL); ax[0].set_ylim(0, 100); ax[0].set_ylabel('Share of MEV-Boost blocks (%)'); ax[0].set_title('All relays', fontsize=10); ax[0].legend(frameon=False, fontsize=8, loc='lower left', ncol=2)
s2 = pd.crosstab(dl.mo, dl.g, normalize='index').reindex(columns=BIG + ['Other']).fillna(0); ax[1].stackplot(s2.index, (100 * s2).T.to_numpy(), colors=COL); ax[1].set_ylim(0, 100); ax[1].set_title('Five-relay sample of this paper', fontsize=10); ax[1].set_ylabel('Share of sample blocks (%)')
mq = m.groupby('mo'); ax[2].plot(mq.opL.apply(hhi), color='#08519c', label='By operator'); ax[2].plot(mq.opM.apply(hhi), color='#08519c', ls=':', label='By operator, BuilderNet partners merged'); ax[2].plot(mq.builder_pubkey.apply(hhi), color='darkorange', label='By builder key'); ax[2].axhline(2500, color='k', lw=0.6, ls=':'); ax[2].set_ylim(0); ax[2].set_ylabel('Monthly HHI, all relays'); ax[2].legend(frameon=False, fontsize=8)
for x in ax: x.tick_params(axis='x', labelrotation=30)
plt.tight_layout(); plt.savefig(paths.FIG + 'fig_market_operators.pdf', bbox_inches='tight'); plt.close()
