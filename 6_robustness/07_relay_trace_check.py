"""Why is a rival's live bid above the winning bid in a fifth of slots?  A same-relay check on 24 sample days.

The bid archive stores each bid once, whatever the number of relays it was sent to, so it cannot say on which relay a rival bid was live.
ethPandaOps Xatu (mev_relay_bid_trace) records bids per relay with the relay's own timestamp.  7_all_bids/pbs_relay_traces.ipynb reduced 24 daily
files (one per month, four in May 2026, two in November 2025) to, per delivered slot, relay and builder key: the key's latest bid on that relay when
the winning block reached that relay (live_w, as a ratio to the winning value) and when that bid arrived (tlive_w).
  data/relay_trace_keys.parquet, data/relay_trace_relays.parquet   the reduced traces
  data/relay_trace_delivered.parquet                               which relays delivered each slot (Data Always, mevboost-data)
Writes work/relay_trace_numbers.csv and figures/fig_relay_trace.pdf."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import paths
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
pd.set_option('display.width', 230); pd.set_option('display.max_rows', 200)
N = {}
def put(k, v, nd=2): N[k] = float(v); print(f'{k:86s} {float(v):,.{nd}f}')
D = paths.DATA; k = pd.read_parquet(D + 'relay_trace_keys.parquet'); r = pd.read_parquet(D + 'relay_trace_relays.parquet'); mb = pd.read_parquet(D + 'relay_trace_delivered.parquet')
om = pd.read_csv(paths.OPS).set_index('pubkey'); ab = pd.read_parquet(paths.AB, columns=['slot', 'has_win', 'stand_L'])
k['op'] = k.pk.map(om.L); w = k[k.isw].groupby('slot').op.first().rename('wop'); k = k.join(w, on='slot'); k = k[k.wop.notna()]
r = r.join(w, on='slot'); put('sample days', r.day.nunique(), 0); put('sample slots with bids in the traces', r.slot.nunique(), 0)
cov = r.assign(h=r.tw.notna()).groupby('relay').agg(slots=('slot', 'size'), winner_found=('h', 'mean'), bids=('n', 'median')); print(cov.round(3).to_string())
r = r[r.tw.notna() & r.wop.notna()]
riv = k[(k.op != k.wop) & k.live_w.notna()]; top = riv.loc[riv.groupby(['slot', 'relay']).live_w.idxmax(), ['slot', 'relay', 'live_w', 'tlive_w', 'pk']]; top['rlab'] = top.pk.map(om.label).fillna('unlabelled'); top = top.drop(columns='pk')
x = r.merge(top, on=['slot', 'relay'], how='left'); x['neg'] = x.live_w > 1; x['age'] = x.tw - x.tlive_w
MAP = {'BloXroute Max Profit': 'bloxroute_max_profit', 'BloXroute Regulated': 'bloxroute_regulated', 'Agnostic Gnosis': 'agnostic', 'Flashbots': 'flashbots', 'Aestus': 'aestus', 'Titan Relay': 'titan', 'Ultra Sound': 'ultrasound', 'EthGas': 'ethgas'}
x['mrel'] = x.relay.map(MAP); x = x.merge(mb.assign(deliv=True).rename(columns={'relay': 'mrel'}), on=['slot', 'mrel'], how='left'); d = x[x.deliv == True].copy()
d = d.merge(ab, on='slot', how='left'); d['aneg'] = d.stand_L < 0; d['q'] = pd.to_datetime(1606824023 + 12 * d.slot, unit='s').dt.to_period('Q').astype(str)
put('slot-relay pairs: winning block received and delivered by that relay', len(d), 0); put('... slots', d.slot.nunique(), 0)
print('\n=== on the relay that delivered the block, on that relay\'s clock ===')
put('a rival operator had a higher live bid on the delivering relay (% of slot-relay pairs)', 100 * d.neg.mean())
AG = [0, 5, 10, 25, 50, 100, 250, 500, 1000]
for a in AG: put(f'... and that bid was at least {a} ms old (% of slot-relay pairs)', 100 * (d.neg & (d.age >= a)).mean())
n = d[d.neg]; put('age of the higher rival bid: median (ms)', n.age.median(), 0); put('age of the higher rival bid: p25 (ms)', n.age.quantile(.25), 0); put('age of the higher rival bid: p75 (ms)', n.age.quantile(.75), 0); put('age of the higher rival bid: p90 (ms)', n.age.quantile(.9), 0)
put('share of higher rival bids less than 50 ms old (%)', 100 * (n.age < 50).mean(), 1); put('share of higher rival bids less than 100 ms old (%)', 100 * (n.age < 100).mean(), 1)
print(d.groupby('relay').agg(pairs=('neg', 'size'), higher=('neg', 'mean'), higher_100ms=('neg', lambda v: (v & (d.loc[v.index, 'age'] >= 100)).mean()), archive=('aneg', 'mean'), margin=('live_w', lambda v: 1 - v.median())).round(4).to_string())
d['m'] = 1 - d.live_w; d['neg100'] = d.neg & (d.age >= 100)
s = d.groupby('slot').agg(any0=('neg', 'max'), any100=('neg100', 'max'), m=('m', 'min'), a=('stand_L', 'first'), aneg=('aneg', 'first'), q=('q', 'first')); s = s[s.a.notna()]
print('\n=== slot level ===')
put('slots in the same-relay check', len(s), 0); put('archive: rival live bid higher (% of these slots)', 100 * s.aneg.mean()); put('same relay: rival live bid higher on a delivering relay (% of slots)', 100 * s.any0.mean()); put('same relay: higher and at least 100 ms old (% of slots)', 100 * s.any100.mean())
put('median margin on the delivering relay (%)', 100 * s.m.median(), 3); put('median archive standing margin, same slots (%)', 100 * s.a.median(), 3); put('rank correlation of the two margins', s.m.corr(s.a, method='spearman'), 3)
Q = s.groupby('q').agg(slots=('a', 'size'), archive=('aneg', lambda v: 100 * v.mean()), same=('any0', lambda v: 100 * v.mean()), same100=('any100', lambda v: 100 * v.mean()), m_same=('m', lambda v: 100 * v.median()), m_arch=('a', lambda v: 100 * v.median())); print(Q.round(2).to_string())
for q_ in Q.index:
    for c in Q.columns: N[f'quarter {q_} {c}'] = float(Q.loc[q_, c])
k2 = k[k.op != k.wop]; g_any = k2.groupby('slot').live_g.max(); a = s[s.aneg].join(g_any.rename('g'))
wh = np.where(a.any0, 'delivering', np.where(a.g > 1, 'other', 'none')); put('archive-negative slots in the check', len(a), 0)
put('archive-negative slots: higher rival bid found on a delivering relay (%)', 100 * (wh == 'delivering').mean(), 1); put('archive-negative slots: found on another relay only (%)', 100 * (wh == 'other').mean(), 1); put('archive-negative slots: not found on any covered relay (%)', 100 * (wh == 'none').mean(), 1)
print('\n=== who holds the higher bids that are at least 100 ms old ===')
o = d[d.neg100]; BIG5 = ['Titan', 'buildernet', 'beaverbuild.org', 'Quasar', 'rsync']; vc = o.rlab.value_counts(normalize=True); print((100 * vc.head(8)).round(1).to_dict())
put('old higher bids: slot-relay pairs', len(o), 0); put('old higher bids from one of the five largest operators (%)', 100 * o.rlab.isin(BIG5).mean(), 1); put('old higher bids from bobthebuilder (%)', 100 * (o.rlab == 'bobthebuilder').mean(), 1); put('old higher bids from bloXroute (%)', 100 * (o.rlab == 'bloXroute').mean(), 1)
put('old higher bids: median excess over the winning bid (%)', 100 * (o.live_w.median() - 1), 1); put('old higher bids from the five largest: median excess (%)', 100 * (o.live_w[o.rlab.isin(BIG5)].median() - 1), 2); put('old higher bids: median age (ms)', o.age.median(), 0)
d['neg100big'] = d.neg100 & d.rlab.isin(BIG5); put('slots with an old higher bid from one of the five largest operators (%)', 100 * d.groupby('slot').neg100big.max().reindex(s.index).mean(), 2)
dd = d[d.day != '2024-12-15']; put('slots with an old higher bid, excluding 15 December 2024 (%)', 100 * dd.groupby('slot').neg100.max().mean(), 2); put('15 December 2024, BloXroute: old higher bids (% of slot-relay pairs)', 100 * d[(d.day == '2024-12-15') & d.relay.str.startswith('BloXroute')].neg100.mean(), 1)
pd.Series(N).to_csv(paths.WORK + 'relay_trace_numbers.csv', header=['value'])
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}); fig, ax = plt.subplots(1, 2, figsize=(12.5, 3.9))
ax[0].plot(range(len(AG)), [100 * (d.neg & (d.age >= a)).mean() for a in AG], marker='o', color='steelblue'); ax[0].set_xticks(range(len(AG))); ax[0].set_xticklabels(AG); ax[0].set_ylim(0); ax[0].set_xlabel('Minimum age of the rival bid when the winning block reached the relay (ms)'); ax[0].set_ylabel('Rival live bid above the winning bid\n(% of slot-relay pairs)')
xx = np.arange(len(Q)); ax[1].bar(xx - 0.27, Q.archive, 0.27, label='Bid archive, across relays', color='#9ecae1'); ax[1].bar(xx, Q.same, 0.27, label='Delivering relay, any age', color='#3182bd'); ax[1].bar(xx + 0.27, Q.same100, 0.27, label='Delivering relay, at least 100 ms old', color='#08519c')
ax[1].set_xticks(xx); ax[1].set_xticklabels([q_[2:4] + 'Q' + q_[-1] for q_ in Q.index]); ax[1].set_ylabel('Slots with a higher rival live bid (%)'); ax[1].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(paths.FIG + 'fig_relay_trace.pdf', bbox_inches='tight'); plt.close()
