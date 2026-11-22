"""Finding 8b and the relay-artifact part of Finding 8: bid timing by relay on the complete Q1 2025 bid archive.

Reads the 90 daily files 2025-01-01 .. 2025-03-31 from the bid archive (paths.BIDARCHIVE), about 86.8M bids.
Reproduces: the bid-timing-by-relay table, the per-builder bloxroute-vs-other comparison, and the corrected
win rates of the builders in the apparent "post-slot" cluster (median_timing > 1000 ms in reference/builder_clusters.csv).

This is the analysis originally run inline on 2026-06-16; the logic is unchanged. It is restructured to stream:
each day is reduced to a small cache file in work/q1_cache/, so it needs about 1.5 GB of memory and can be resumed.
Needs: the bid archive, work/dedup.parquet (2_analysis/01), reference/builder_clusters.csv.
Optional: set MAX_SECONDS to stop caching after that many seconds (rerun to continue)."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import paths
import time, zipfile, datetime
from datetime import date, timedelta
from pathlib import Path
import numpy as np, pandas as pd

BIDDIR = Path(paths.BIDARCHIVE); CACHE = Path(paths.WORK) / 'q1_cache'; CACHE.mkdir(parents=True, exist_ok=True)
MAXS = float(os.environ.get('MAX_SECONDS', '1e18')); t0 = time.time()
days = [date(2025, 1, 1) + timedelta(days=i) for i in range(90)]          # 2025-01-01 .. 2025-03-31

# ---- pass 1: one compact cache file per day -------------------------------------------------
missing = []
for d in days:
    out = CACHE / f'{d}.parquet'
    if out.exists(): continue
    src = BIDDIR / f'{d}_top.csv.zip'
    if not src.exists(): missing.append(str(d)); continue
    if time.time() - t0 > MAXS: print('time budget reached; rerun to continue'); sys.exit(2)
    with zipfile.ZipFile(src) as zf:
        with zf.open(zf.namelist()[0]) as f:
            tmp = pd.read_csv(f, usecols=['slot', 'value', 'builder_pubkey', 'relay', 'slot_t_ms'])
    for c in ('value', 'slot', 'slot_t_ms'): tmp[c] = pd.to_numeric(tmp[c], errors='coerce')
    tmp = tmp.dropna(subset=['value', 'slot', 'slot_t_ms'])               # same filter as the original run
    tmp = tmp[['slot', 'slot_t_ms', 'relay', 'builder_pubkey']]
    tmp['slot'] = tmp['slot'].astype('int64'); tmp['relay'] = tmp['relay'].astype('category'); tmp['builder_pubkey'] = tmp['builder_pubkey'].astype('category')
    tmp.to_parquet(out.with_suffix('.tmp'), index=False); os.replace(out.with_suffix('.tmp'), out)
if missing: print('MISSING archive days:', missing)

# ---- pass 2: aggregate ----------------------------------------------------------------------
clus = pd.read_csv(paths.REF + 'builder_clusters.csv')
c2 = set(clus[clus['median_timing'] > 1000]['builder_pubkey'])
per_relay, c2_rows, n = {}, [], 0
for d in days:
    p = CACHE / f'{d}.parquet'
    if not p.exists(): continue
    x = pd.read_parquet(p); n += len(x)
    for r, g in x.groupby('relay', observed=True)['slot_t_ms']:
        per_relay.setdefault(str(r), []).append(g.to_numpy(dtype='float64'))
    s = x[x['builder_pubkey'].isin(c2)]
    c2_rows.append(pd.DataFrame({'slot': s['slot'].to_numpy(), 't': s['slot_t_ms'].to_numpy(), 'relay': s['relay'].astype(str).to_numpy(), 'builder': s['builder_pubkey'].astype(str).to_numpy()}))
print(f'days cached: {len(list(CACHE.glob("*.parquet")))}   total bid observations: {n:,}   (paper: 86,805,657)')

rows = []
for r, parts in per_relay.items():
    a = np.concatenate(parts); rows.append((r, np.median(a), a.mean(), len(a))); del a
tab = pd.DataFrame(rows, columns=['relay', 'median_ms', 'mean_ms', 'n']).sort_values('median_ms', ascending=False)
tab['share_pct'] = 100 * tab.n / n
print('\n=== Bid timing by relay, all builders pooled ===')
print(tab.to_string(index=False, formatters={'median_ms': '{:,.0f}'.format, 'mean_ms': '{:,.0f}'.format, 'n': '{:,}'.format, 'share_pct': '{:.2f}'.format}))
print(f'spread of relay medians: {tab.median_ms.max() - tab.median_ms.min():,.0f} ms')
tab.to_csv(paths.WORK + 'relay_timing_q1_2025.csv', index=False)

c2b = pd.concat(c2_rows, ignore_index=True); del c2_rows
print(f'\n=== Builders in the apparent post-slot cluster: {len(c2)} builders, {len(c2b):,} bids ===')
print(c2b.groupby('relay')['t'].agg(['median', 'mean', 'count']).sort_values('count', ascending=False).round(0).to_string())
print('\n=== Same builder, bloxroute vs every other relay (builders with more than 10 bids on each side) ===')
isb = c2b.relay.str.contains('bloxroute', case=False)
cmp_ = []
for pk, g in c2b.groupby('builder'):
    b, o = g[isb.loc[g.index]], g[~isb.loc[g.index]]
    if len(b) > 10 and len(o) > 10: cmp_.append((pk[:10], b.t.median(), len(b), o.t.median(), len(o)))
print(pd.DataFrame(cmp_, columns=['builder', 'bloxroute_median', 'n_blox', 'other_median', 'n_other']).sort_values('n_blox', ascending=False).to_string(index=False))
cd_ = pd.DataFrame(cmp_, columns=['builder', 'bm', 'nb', 'om', 'no']); diff = cd_.bm - cd_.om
print(f'bloxroute median minus other-relay median, per builder: {len(cd_)} builders, later via bloxroute for {(diff > 0).sum()}, range {diff.min():+,.0f} to {diff.max():+,.0f} ms, median {diff.median():+,.0f} ms')
us_all = float(tab.loc[tab.relay == 'relay.ultrasound.money', 'median_ms'].iloc[0]); us_c2 = c2b[c2b.relay == 'relay.ultrasound.money']
print(f'within ultrasound alone: these builders median {us_c2.t.median():+,.0f} ms ({len(us_c2):,} bids, {100 * len(us_c2) / len(c2b):.1f}% of their bids); all builders median {us_all:+,.0f} ms')

# corrected win rate: wins in delivered payloads / slots in which the builder bid, both over the same quarter
GEN = datetime.datetime(2020, 12, 1, 12, 0, 23)
dl = pd.read_parquet(paths.WORK + 'dedup.parquet', columns=['slot', 'builder_pubkey', 'value_eth'])
dl['date'] = pd.to_datetime(GEN) + pd.to_timedelta(dl.slot * 12, unit='s')
win = dl[(dl.date >= '2025-01-01') & (dl.date <= '2025-03-31')]           # as in the original run (ends 2025-03-31 00:00)
bidslots = c2b.groupby('builder').slot.nunique()
res = pd.DataFrame({'n_bid_slots': bidslots.reindex(sorted(c2)).fillna(0).astype(int)})
res['n_wins'] = [int((win.builder_pubkey == pk).sum()) for pk in res.index]
res['win_rate_pct'] = np.where(res.n_bid_slots > 0, 100 * res.n_wins / res.n_bid_slots.replace(0, np.nan), np.nan)
res['avg_win_value_eth'] = [win[win.builder_pubkey == pk].value_eth.mean() for pk in res.index]
print('\n=== Corrected win rates ===')
print(res.assign(builder=[k[:10] for k in res.index]).set_index('builder').sort_values('n_bid_slots', ascending=False).round(4).to_string())
valid = res[(res.win_rate_pct <= 100)]
print(f'\nmedian win rate (valid): {valid.win_rate_pct.median():.2f}%   (paper: 0.59%)')
print(f'pooled win rate: {100 * res.n_wins.sum() / res.n_bid_slots.sum():.2f}%   (paper: 8.62%)')
print(f'mean winning value, these builders: {win[win.builder_pubkey.isin(c2)].value_eth.mean():.4f} ETH   all builders: {win.value_eth.mean():.4f} ETH')
print(f'done in {time.time() - t0:.0f}s')
