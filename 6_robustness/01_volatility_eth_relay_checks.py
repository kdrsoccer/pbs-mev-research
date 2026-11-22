"""Independent checks on the referee-level points. Reads the two parquet files only."""
import sys
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

D = (sys.argv[1] if len(sys.argv) > 1 else paths.DATA.rstrip(os.sep))
cols = ['slot', 'value_eth', 'relay', 'builder_pubkey', 'n_builders', 'bid_margin']
df = pd.read_parquet(D + '/bid_auction_stats.parquet', columns=cols)
df = df.drop_duplicates('slot', keep='first').sort_values('slot').reset_index(drop=True)
df['t'] = pd.to_datetime(1606824023 + 12 * df['slot'], unit='s', utc=True)
print('rows', len(df), df['t'].min(), df['t'].max())

# ---------- A. Finding 1: is the volatility-value link mechanical?
v = df['value_eth'].to_numpy()
u = np.log1p(v)
r = pd.Series(np.diff(u, prepend=np.nan))
W = 300
rv = r.rolling(W).std()


def sp(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return spearmanr(a[m], b[m]).statistic, int(m.sum())


def ratio(val, vol):
    m = np.isfinite(vol)
    q1, q2 = np.nanquantile(vol, [0.33, 0.67])   # same cut points as the regime table
    lo = val[m & (vol <= q1)].mean()
    hi = val[m & (vol > q2)].mean()
    return hi / lo, np.median(val[m & (vol > q2)]) / np.median(val[m & (vol <= q1)])


print('\nA. Finding 1')
print(' contemporaneous window (paper):      rho=%.3f n=%d' % sp(v, rv.to_numpy()), ' mean ratio hi/lo=%.2f median ratio=%.2f' % ratio(v, rv.to_numpy()))
for lag, name in [(1, 'window ending one block earlier'), (W, 'window ending 300 blocks earlier (no overlap)')]:
    x = rv.shift(lag).to_numpy()
    print(' %-36s rho=%.3f n=%d' % ((name + ':',) + sp(v, x)), ' mean ratio hi/lo=%.2f median ratio=%.2f' % ratio(v, x))
rng = np.random.default_rng(0)
ps = []
pr = []
for _ in range(5):
    vs = rng.permutation(v)
    rs = pd.Series(np.diff(np.log1p(vs), prepend=np.nan)).rolling(W).std().to_numpy()
    ps.append(sp(vs, rs)[0])
    pr.append(ratio(vs, rs)[0])
print(' placebo, values shuffled in time:    rho=%.3f (5 shuffles, range %.3f..%.3f); mean ratio hi/lo=%.2f' % (np.mean(ps), min(ps), max(ps), np.mean(pr)))

# ETH volatility as an external regime variable
eth = pd.read_parquet(D + '/eth_price_hourly.parquet')
print(' eth columns', list(eth.columns)[:8], len(eth))
tcol = [c for c in eth.columns if 'time' in c.lower() or 'date' in c.lower()]
pcol = [c for c in eth.columns if 'close' in c.lower() or 'price' in c.lower()]
if not tcol:
    eth = eth.reset_index()
    tcol = [eth.columns[0]]
e = eth[[tcol[0], pcol[0]]].copy()
e.columns = ['h', 'p']
if not np.issubdtype(e['h'].dtype, np.datetime64) and 'datetime64' not in str(e['h'].dtype):
    e['h'] = pd.to_datetime(e['h'], unit='s', utc=True) if np.issubdtype(e['h'].dtype, np.number) else pd.to_datetime(e['h'], utc=True)
if e['h'].dt.tz is None:
    e['h'] = e['h'].dt.tz_localize('UTC')
e = e.sort_values('h').drop_duplicates('h')
e['ret'] = np.log(e['p']).diff()
e['absret'] = e['ret'].abs()
e['rv24'] = e['ret'].rolling(24).std()
df['h'] = df['t'].dt.floor('h')
hv = df.groupby('h')['value_eth'].agg(['mean', 'median', 'count']).reset_index()
m = hv.merge(e, on='h', how='inner')
m = m[m['count'] >= 20]
print(' hours with >=20 blocks and ETH price:', len(m))
for col, name in [('absret', 'hourly |ETH return|'), ('rv24', '24h ETH realized vol')]:
    a = sp(m['median'].to_numpy(), m[col].to_numpy())
    b = sp(m['mean'].to_numpy(), m[col].to_numpy())
    print(' %-22s vs hourly median value rho=%.3f (n=%d); vs hourly mean value rho=%.3f' % (name, a[0], a[1], b[0]))
q1, q2 = m['rv24'].quantile([1 / 3, 2 / 3])
print(' hourly mean block value by ETH 24h-vol tercile: low %.4f mid %.4f high %.4f; medians %.4f %.4f %.4f' % (
    m.loc[m.rv24 <= q1, 'mean'].mean(), m.loc[(m.rv24 > q1) & (m.rv24 <= q2), 'mean'].mean(), m.loc[m.rv24 > q2, 'mean'].mean(),
    m.loc[m.rv24 <= q1, 'median'].median(), m.loc[(m.rv24 > q1) & (m.rv24 <= q2), 'median'].median(), m.loc[m.rv24 > q2, 'median'].median()))

# ---------- C. ETH independence at an hourly frequency (instead of slot-level returns on hourly bars)
m2 = hv.merge(e, on='h', how='inner')
m2 = m2[m2['count'] >= 20].sort_values('h')
m2['dlv'] = np.log(m2['median']).diff()
ok = np.isfinite(m2['dlv']) & np.isfinite(m2['ret']) & (m2['h'].diff() == pd.Timedelta('1h'))
x = m2.loc[ok, 'ret'].to_numpy()
y = m2.loc[ok, 'dlv'].to_numpy()
print('\nC. hourly change in log median block value on hourly ETH return: R2=%.4f, corr=%.3f, n=%d' % (np.corrcoef(x, y)[0, 1] ** 2, np.corrcoef(x, y)[0, 1], ok.sum()))
print('   same, on |ETH return|: corr(log median value, |ret|)=%.3f' % np.corrcoef(np.log(m2.loc[ok, 'median']), np.abs(x))[0, 1])

# ---------- B. composition break: within-relay concentration and participation
print('\nB. relay composition')
df['week'] = df['t'].dt.tz_localize(None).dt.to_period('W-SUN').dt.start_time


def hhi(s):
    p = s.value_counts(normalize=True)
    return float((p ** 2).sum() * 10000)


def summarize(sub, label):
    g = sub.groupby('week')
    w = pd.DataFrame({'hhi': g['builder_pubkey'].apply(hhi), 'nb': g['n_builders'].median(), 'n': g.size()})
    w = w[w['n'] >= 300]
    periods = [('2024-09..12', '2024-09-01', '2025-01-05'), ('2025-01..04', '2025-01-06', '2025-04-30'),
               ('2025-05..12', '2025-05-01', '2025-12-31'), ('2026-01..05', '2026-01-01', '2026-06-01')]
    out = []
    for name, a, b in periods:
        x = w[(w.index >= a) & (w.index <= b)]
        out.append('%s: HHI %s, builders/slot %s (%d wks)' % (name, '%.0f' % x.hhi.mean() if len(x) else 'n/a', '%.1f' % x.nb.mean() if len(x) else 'n/a', len(x)))
    print(' ' + label + '\n   ' + '\n   '.join(out))


summarize(df, 'all relays pooled (paper)')
for rl in df['relay'].value_counts().index[:2]:
    summarize(df[df['relay'] == rl], 'only slots won via ' + str(rl))
wk = df[(df['week'] >= '2024-12-09') & (df['week'] <= '2025-01-27')].groupby('week')
print(' weeks around the break (pooled): ' + '; '.join('%s HHI %.0f nb %.0f n %d' % (str(k.date()), hhi(g['builder_pubkey']), g['n_builders'].median(), len(g)) for k, g in wk))
ag = df[(df['relay'] == df['relay'].value_counts().index[1]) & (df['week'] >= '2024-12-09') & (df['week'] <= '2025-01-27')].groupby('week')
print(' same weeks, second-largest relay only: ' + '; '.join('%s HHI %.0f nb %.0f n %d' % (str(k.date()), hhi(g['builder_pubkey']), g['n_builders'].median(), len(g)) for k, g in ag))
print(' relay shares:', df['relay'].value_counts(normalize=True).round(3).to_dict())
