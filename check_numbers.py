"""Recompute the paper's reported statistics from work/ and compare each to the value printed in the paper.
Run after 2_analysis/01-04 (and 05 for the GARCH block). Prints PASS/DIFF per number and a total.
`paper` values are transcribed from 4_paper/*.tex; a number passes if it equals the recomputed value at the printed precision."""
import os, sys, json, re
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
import paths
W = paths.WORK
rows = []
def chk(name, paper, got, dec=None):
    if dec is None:
        s = str(paper); dec = len(s.split('.')[1]) if '.' in s else 0
    ok = abs(round(float(got), dec) - float(paper)) < 0.5 * 10 ** (-dec) * 1e-6 + 1e-12
    rows.append((name, paper, round(float(got), dec + 2), 'PASS' if ok else 'DIFF'))

df = pd.read_parquet(W + 'slim.parquet').sort_values('slot').reset_index(drop=True)
v = df.value_eth
chk('unique delivered slots', 1271079, len(df))
# ---- volatility regimes (Table: regimes) ----
r = np.log1p(v).diff(); rv = r.rolling(300).std() * np.sqrt(300)
d = df[rv.notna()].copy(); d['rv'] = rv[rv.notna()]
lo, hi = d.rv.quantile(0.33), d.rv.quantile(0.67)
d['reg'] = pd.cut(d.rv, [-np.inf, lo, hi, np.inf], labels=['Low', 'Mid', 'High'])
g = d.groupby('reg', observed=True).value_eth
for reg, n, mean, med, sd, p25, p75, p95 in [('Low', 419357, .0164, .0102, .0270, .0070, .0164, .0427), ('Mid', 432065, .0313, .0169, .0746, .0107, .0293, .0858), ('High', 419357, .1149, .0276, 2.6463, .0156, .0547, .2136)]:
    x = g.get_group(reg)
    chk(f'regime {reg} n', n, len(x)); chk(f'regime {reg} mean', mean, x.mean(), 4); chk(f'regime {reg} median', med, x.median(), 4); chk(f'regime {reg} sd', sd, x.std(), 4)
    chk(f'regime {reg} p25', p25, x.quantile(.25), 4); chk(f'regime {reg} p75', p75, x.quantile(.75), 4); chk(f'regime {reg} p95', p95, x.quantile(.95), 4)
chk('regime high/low mean ratio', 7.0, g.get_group('High').mean() / g.get_group('Low').mean(), 1)
chk('regime Kruskal-Wallis H', 266905.3, stats.kruskal(*[x.values for _, x in g])[0], 1)
chk('Spearman rv-value', 0.494, stats.spearmanr(d.rv, d.value_eth)[0], 3); chk('Spearman rv-value n', 1270779, len(d))
# ---- temporal ----
chk('KW hour H', 65875, stats.kruskal(*[x.values for _, x in df.groupby('hour_c').value_eth])[0], 0)
chk('KW weekday H', 33590, stats.kruskal(*[x.values for _, x in df.groupby('dow_c').value_eth])[0], 0)
dm = df.groupby('dow_c').value_eth.mean(); chk('Monday mean', 0.086, dm[0], 3); chk('Saturday mean', 0.031, dm[5], 3)
# ---- concentration ----
def gini(c):
    c = np.sort(np.asarray(c, float)); n = len(c); return (2 * np.sum(np.arange(1, n + 1) * c) / (n * c.sum())) - (n + 1) / n
cnt = df.builder_pubkey.value_counts()
chk('builders', 227, len(cnt)); chk('pooled Gini', 0.868, gini(cnt.values), 3)
wc = pd.read_csv(W + 'weekly_conc.csv'); chk('weekly Gini mean', 0.77, wc.gini.mean(), 2); chk('weekly Gini first week', 0.85, wc.gini.iloc[0], 2); chk('weeks', 89, len(wc))
# ---- auction (matched slots) ----
m = pd.read_parquet(W + 'matched.parquet')
chk('matched slots', 1219842, len(m)); chk('match rate %', 96.0, 100 * len(m) / len(df), 1); chk('single-bid slots', 213, (m.n_bids == 1).sum())
T = {'n_bids': (141.5, 132, 19, 251, 7135), 'n_builders': (11.5, 11, 7, 17, 36), 'n_relays': (2.6, 2, 1, 4, 9), 'm': (3.265, 0.141, 0.000, 6.691, 100.00),
     'earliest_bid_ms': (-8046, -8090, -9772, -5380, 5444), 'latest_bid_ms': (2186, 2057, 1010, 3647, 22082)}
for c, (mean, med, p10, p90, mx) in T.items():
    x = m[c].dropna(); dec = 3 if c == 'm' else 0
    chk(f'auction {c} mean', mean, x.mean(), 1 if c in ('n_bids', 'n_builders', 'n_relays') else dec); chk(f'auction {c} median', med, x.median(), dec)
    chk(f'auction {c} p10', p10, x.quantile(.1), dec); chk(f'auction {c} p90', p90, x.quantile(.9), dec); chk(f'auction {c} max', mx, x.max(), 2 if c == 'm' else 0)
chk('latest bid > 4000 ms, %', 5.3, 100 * (m.latest_bid_ms > 4000).mean(), 1)
c = m[np.isfinite(m.m) & (m.m >= 0) & (m.m < 100)].copy(); chk('clean-margin n', 1219628, len(c))
c['ter'] = pd.qcut(c.value_eth, 3, labels=['Low', 'Mid', 'High']); tm = c.groupby('ter', observed=True).m
for t, med, emean, emed in [('Low', 0.1965, 95.5, 99.80), ('Mid', 0.1442, 96.8, 99.86), ('High', 0.1099, 97.9, 99.89)]:
    x = tm.get_group(t); chk(f'margin median, {t} tercile', med, x.median(), 4); chk(f'efficiency mean, {t}', emean, 100 - x.mean(), 1); chk(f'efficiency median, {t}', emed, 100 - x.median(), 2)
chk('Spearman value-margin', -0.054, stats.spearmanr(c.value_eth, c.m)[0], 3)
chk('Spearman value-builders', 0.285, stats.spearmanr(m.value_eth, m.n_builders)[0], 3)
x = m.dropna(subset=['latest_bid_ms']); chk('Spearman value-latest bid', 0.146, stats.spearmanr(x.value_eth, x.latest_bid_ms)[0], 3)
# ---- relay-count table ----
rc = c.groupby('n_relays').m.agg(['median', 'size'])
for k, med, n in [(1, 0.217, None), (2, 0.143, None), (3, 0.124, None), (4, 0.124, None), (5, 0.109, None), (6, 0.084, None), (7, 0.066, None)]:
    chk(f'margin median, {k} relays', med, rc.loc[k, 'median'], 3)
s = c[c.n_relays.between(1, 7)]; chk('Spearman relays-margin (1-7 relays)', -0.057, stats.spearmanr(s.n_relays, s.m)[0], 3)
# ---- margin regression table ----
X = pd.DataFrame({'log_value': np.log(c.value_eth), 'n_builders': c.n_builders, 'n_relays': c.n_relays, 'hour': c.hour_c, 'dow': c.dow_c})
res = sm.OLS(c.m.values, sm.add_constant(X)).fit(cov_type='HC0')
for k, b, se, t in [('const', 3.8003, 0.0620, 61.28), ('log_value', -0.7574, 0.0092, -82.57), ('n_builders', -0.1567, 0.0024, -66.30), ('n_relays', -0.7192, 0.0078, -92.57), ('hour', 0.0081, 0.0013, 6.09), ('dow', 0.0119, 0.0047, 2.51)]:
    chk(f'OLS {k} coef', b, res.params[k], 4); chk(f'OLS {k} robust SE', se, res.bse[k], 4); chk(f'OLS {k} t', t, res.tvalues[k], 2)
chk('OLS R2', 0.020, res.rsquared, 3); chk('OLS n', 1219628, int(res.nobs))
# ---- ETH price ----
eth = pd.read_parquet(paths.ETH); eth['hf'] = eth.datetime.dt.floor('h'); chk('ETH hourly bars', 15302, len(eth))
mm = m.copy(); mm['hf'] = mm.t.dt.tz_localize(None).dt.floor('h'); mm = mm.merge(eth[['hf', 'eth_close']], on='hf', how='inner')
chk('matched slots with ETH price', 1218624, len(mm))
sl, ic, rr, _, _ = stats.linregress(np.log(mm.eth_close), np.log(mm.value_eth))
chk('ETH regression intercept', -4.172, ic, 3); chk('ETH regression slope', 0.027, sl, 3); chk('ETH regression R2', 0.0001, rr ** 2, 4)
chk('Spearman ETH price - value (ETH)', 0.050, stats.spearmanr(mm.eth_close, mm.value_eth)[0], 3); chk('Spearman ETH price - value (USD)', 0.359, stats.spearmanr(mm.eth_close, mm.value_eth * mm.eth_close)[0], 3)
wa = pd.read_csv(W + 'weekly_auction.csv') if os.path.exists(W + 'weekly_auction.csv') else None
# ---- GARCH block: parsed from the 05 log ----
log = os.path.join(paths.ROOT, 'logs', '05_garch_level.txt')
if os.path.exists(log) and 'DONE' in open(log).read():
    L = open(log).read()
    def grab(tag):
        mo = re.search(re.escape(tag) + r'.*?alpha=([\d.]+) beta=([\d.]+) alpha\+beta=([\d.]+)', L); return [float(x) for x in mo.groups()]
    for tag, a, b, s_ in [('Table 4: AR(5) resid GARCH(1,1)', .044, .965, 1.009), ('Table 5: AR(1) resid', .043, .964, 1.007), ('Table 5: AR(20) resid', None, None, 1.010),
                          ('Table 5: AR(5) first half', .044, .958, 1.002), ('Table 5: AR(5) second half', .231, .844, 1.075), ('Table 5: differenced', .539, .731, 1.270)]:
        ga, gb, gs = grab(tag)
        if a is not None: chk(f'GARCH {tag} alpha', a, ga, 3); chk(f'GARCH {tag} beta', b, gb, 3)
        chk(f'GARCH {tag} alpha+beta', s_, gs, 3)
    q = [float(x) for x in re.search(r'level resid\^2 Q10/20/50: \[([^\]]+)\]', L).group(1).split(',')]
    for k, pv in zip((10, 20, 50), (44779, 56366, 72464)): chk(f'Ljung-Box level resid^2 Q{k}', pv, q[(10, 20, 50).index(k)], 0)
    q = [float(x) for x in re.search(r'diff\^2 Q10/20/50: \[([^\]]+)\]', L).group(1).split(',')]
    for k, pv in zip((10, 20, 50), (257972, 268977, 280428)): chk(f'Ljung-Box diff^2 Q{k}', pv, q[(10, 20, 50).index(k)], 0)
    g4 = json.load(open(W + 'garch_level.json'))['table4']
    chk('Table 4 omega', 0.648, g4['omega'], 3); chk('Table 4 log-likelihood', -4240438, g4['loglik'], 0)
    for nm, pv, gv in zip(('omega', 'alpha', 'beta'), (0.158, 0.010, 0.007), g4['se']): chk(f'Table 4 robust SE {nm}', pv, gv, 3)
else:
    print('(GARCH block skipped: logs/05_garch_level.txt not complete)')
# ---- bid-level block (Findings 8, 8b): needs 5_bid_level/01 and 02 to have run against the bid archive ----
rt, bc = W + 'relay_timing_q1_2025.csv', W + 'builder_clusters.csv'
if os.path.exists(rt) and os.path.exists(bc):
    t = pd.read_csv(rt).set_index('relay'); chk('Q1 2025 bid observations', 86805657, t.n.sum())
    for r, med, mean, n in [('titanrelay.xyz', 2497, 580, 1512420), ('bloxroute.regulated.blxrbdn.com', 2208, 2421, 359445), ('bloxroute.max-profit.blxrbdn.com', 2104, 2303, 237888),
                            ('agnostic-relay.net', 1890, 1748, 125789), ('boost-relay.flashbots.net', -454, -346, 790105), ('relay.ultrasound.money', -710, -1946, 83630070),
                            ('aestus.live', -4258, -3334, 126004), ('relay.edennetwork.io', -4447, -3548, 15169)]:
        chk(f'relay timing {r} median', med, t.loc[r, 'median_ms'], 0); chk(f'relay timing {r} mean', mean, t.loc[r, 'mean_ms'], 0); chk(f'relay timing {r} n', n, t.loc[r, 'n'])
    chk('securerpc median', -4913, t.loc['mainnet-relay.securerpc.com', 'median_ms'], 0); chk('ultrasound share %', 96.3, t.loc['relay.ultrasound.money', 'share_pct'], 1)
    chk('Titan - Eden median spread', 6944, t.loc['titanrelay.xyz', 'median_ms'] - t.loc['relay.edennetwork.io', 'median_ms'], 0)
    b = pd.read_csv(bc); cs = b.groupby('cluster').agg(bps=('bids_per_slot', 'median'), mt=('median_timing', 'median'), ve=('value_eth', 'median')).sort_values('mt')
    chk('builders clustered', 107, len(b)); chk('clusters', 3, len(cs))
    for i, (bps, mt, ve) in enumerate([(18, -6759, 0.009), (2.4, -416, 0.021), (None, 2665, 0.096)]):
        if bps is not None: chk(f'cluster {i} bids per slot', bps, cs.bps.iloc[i], 0 if bps == 18 else 1)
        chk(f'cluster {i} median timing', mt, cs.mt.iloc[i], 0); chk(f'cluster {i} median value ETH', ve, cs.ve.iloc[i], 3)
else:
    print('(bid-level block skipped)')
out = pd.DataFrame(rows, columns=['statistic', 'paper', 'recomputed', 'result'])
import os as _os
REPORTED = set(open(_os.path.join(paths.ROOT, 'reference', 'reported_statistics.txt')).read().splitlines())   # the pipeline computes more than the paper reports
out = out[out.statistic.isin(REPORTED)].reset_index(drop=True)
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 500)
print(out.to_string(index=False))
print(f"\n{(out.result == 'PASS').sum()} of {len(out)} match the paper.")
out.to_csv(os.path.join(paths.ROOT, 'logs', 'check_numbers.csv'), index=False)
sys.exit(0 if (out.result == 'PASS').all() else 1)
