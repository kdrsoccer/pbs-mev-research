import os, glob, zipfile, io, collections, numpy as np, pandas as pd
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/pbs_all_bids'; RES = OUT + '/results'
cat = lambda sub: pd.concat([pd.read_parquet(f).assign(day=os.path.basename(f)[:10]) for f in sorted(glob.glob(f'{RES}/{sub}/*.parquet'))], ignore_index=True)
K = cat('keys'); L = cat('links')
dl = pd.read_parquet(f'{OUT}/delivered_slim.parquet', columns=['slot', 'value_wei', 'builder_pubkey']); dl['builder_pubkey'] = dl.builder_pubkey.astype(str)
keys = sorted(set(K.pubkey) | set(dl.builder_pubkey) | set(L.a) | set(L.b)); code = {k: i for i, k in enumerate(keys)}; NK = len(keys)
# ---- 1. hash links: keys that submitted the same block; strength = shared blocks / bids of the smaller key on the linked days
L['p'] = [tuple(sorted(x)) for x in zip(L.a, L.b)]; db = K.set_index(['day', 'pubkey']).bids
L['mn'] = np.minimum(db.reindex(pd.MultiIndex.from_frame(L[['day', 'a']])).to_numpy(), db.reindex(pd.MultiIndex.from_frame(L[['day', 'b']])).to_numpy())
H = L.groupby('p').agg(n=('n', 'sum'), days=('day', 'nunique'), mn=('mn', 'sum')).reset_index(); H['share'] = H.n / H.mn
# ---- 2. value ties: keys whose best bid at or below the winning value is exactly equal in the same slot (pass 1 over the top tables)
tops = sorted(glob.glob(f'{RES}/top/*.parquet')); ties = collections.Counter(); pres = np.zeros(NK, 'int64')
for i, f in enumerate(tops):
    t = pd.read_parquet(f, columns=['slot', 'pubkey', 'best_le']); t['k'] = t.pubkey.astype(str).map(code).astype('int32')
    pres += np.bincount(t.k.to_numpy(), minlength=NK)
    m = t[t.duplicated(['slot', 'best_le'], keep=False)]
    if len(m):
        a = m.groupby(['slot', 'best_le']).k.transform('min').to_numpy(); b = m.k.to_numpy(); s = a != b
        ties.update(zip(a[s].tolist(), b[s].tolist()))
    if i % 100 == 0: print('ties pass', i, len(tops), flush=True)
T = pd.DataFrame([(a, b, n) for (a, b), n in ties.items()], columns=['a', 'b', 'ties']).astype('int64'); T['rate'] = T.ties / np.minimum(pres[T.a.to_numpy()], pres[T.b.to_numpy()])
# ---- 3. operator definitions
def comps(pairs):
    par = list(range(NK))
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in pairs: par[find(a)] = find(b)
    return np.array([find(x) for x in range(NK)])
hp = lambda sel: [(code[p[0]], code[p[1]]) for p in sel.p]
tp = lambda sel: list(zip(sel.a.tolist(), sel.b.tolist()))
strict = T[(T.ties >= 200) & (T.rate >= 0.05)]; loose = T[(T.ties >= 50) & (T.rate >= 0.005)]
OPS = {'B': comps(hp(H[H.share >= 0.002])), 'C': comps(hp(H[H.share >= 0.002]) + tp(strict)), 'D': comps(hp(H[H.share >= 0.002]) + tp(loose)), 'KEY': np.arange(NK)}
# ---- 4. recompute each slot's margins under every definition (pass 2)
dv = dl.set_index('slot').value_wei; wk = dl.set_index('slot').builder_pubkey.map(code)
parts, cands = [], []
for i, f in enumerate(tops):
    t = pd.read_parquet(f); t['k'] = t.pubkey.astype(str).map(code).astype('int32'); t['dv'] = t.slot.map(dv).to_numpy(); t['wk'] = t.slot.map(wk).to_numpy()
    r = pd.DataFrame({'slot': np.sort(t.slot.unique())}).set_index('slot')
    for nm, op in OPS.items():
        x = t[op[t.k.to_numpy()] != op[t.wk.to_numpy()]]; g = x.groupby('slot'); d1 = g.dv.first()
        r['any_' + nm] = (1 - g.best_le.max() / d1).astype('float32'); r['pre_' + nm] = (1 - g.best_pre_le.max() / d1).astype('float32'); r['stand_' + nm] = (1 - g.standing.max() / d1).astype('float32')
        c = x[x.standing >= 0.99 * x.dv]; r['riv1_' + nm] = c.assign(o=op[c.k.to_numpy()]).drop_duplicates(['slot', 'o']).groupby('slot').size(); r['riv1_' + nm] = r['riv1_' + nm].fillna(0).astype('int16')
    r['wopB'] = t.groupby('slot').wk.first().map(lambda v: OPS['B'][int(v)]).astype('int16')
    # rival candidates under the hash-only definition B (8 best rival operators per slot), so other groupings can be applied later without the raw files
    oB = OPS['B'][t.k.to_numpy()]; xb = t.assign(o=oB)[oB != OPS['B'][t.wk.to_numpy()]]
    cb = xb.groupby(['slot', 'o']).agg(a=('best_le', 'max'), st=('standing', 'max'), dv=('dv', 'first')).reset_index(); cb['m'] = cb[['a', 'st']].max(axis=1)
    cb = cb.sort_values(['slot', 'm'], ascending=[True, False]); cb = cb[cb.groupby('slot').cumcount() < 8]
    cands.append(pd.DataFrame({'slot': cb.slot.astype('int32'), 'o': cb.o.astype('int16'), 'any_r': (cb.a / cb.dv).astype('float32'), 'st_r': (cb.st / cb.dv).astype('float32'), 'day': os.path.basename(f)[:10]}))
    parts.append(r.reset_index().assign(day=os.path.basename(f)[:10]))
    if i % 100 == 0: print('margins pass', i, len(tops), flush=True)
s = pd.concat(parts, ignore_index=True); s['n'] = s.groupby('slot').slot.transform('size')
hw = cat('days')[['slot', 'has_win', 'n_all', 'day']].sort_values(['slot', 'n_all'], ascending=[True, False]).drop_duplicates('slot')
s = s.merge(hw[['slot', 'day', 'has_win']], on=['slot', 'day'], how='inner').drop(columns='n')       # keep each slot from the day file that holds most of its bids
s['q'] = pd.PeriodIndex(pd.to_datetime(s.day), freq='Q').astype(str)
# ---- 5. summary
B = io.StringIO(); P = lambda *a: print(*a, file=B)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 300)
P('slots', len(s), 'days', s.day.nunique(), '| keys', NK, '| hash pairs', len(H), 'robust', int((H.share >= 0.002).sum()), '| tie pairs', len(T), 'strict', len(strict), 'loose', len(loose))
x = T.sort_values('ties', ascending=False).head(60).copy(); x['a'] = [keys[i][:10] for i in x.a]; x['b'] = [keys[i][:10] for i in x.b]; P('\n== top value-tie pairs =='); P(x.round(4).to_string(index=False))
P('\n== tie-rate distribution =='); P(T.rate.describe(percentiles=[.5, .75, .9, .95, .99]).round(4).to_string()); P(pd.crosstab(pd.cut(T.ties, [0, 5, 50, 200, 1000, 1e9]), pd.cut(T.rate, [0, .005, .05, .2, 1.01])).to_string())
kw = K.groupby('pubkey').wins.sum(); kwq = K.assign(q=pd.PeriodIndex(pd.to_datetime(K.day), freq='Q').astype(str)).groupby(['q', 'pubkey']).wins.sum().reset_index(); kwq['k'] = kwq.pubkey.map(code)
hh = lambda v: 10000 * ((v / v.sum()) ** 2).sum()
P('\n== operators and concentration of wins ==')
for nm, op in OPS.items():
    o = pd.Series(kw.reindex(keys).fillna(0).to_numpy()).groupby(op).sum().sort_values(ascending=False); nk = pd.Series(1, index=range(NK)).groupby(op).sum()
    P(f'[{nm}] operators {len(o)} | multi-key {int((nk > 1).sum())} | whole-sample HHI {hh(o):.0f} | top6 share {(o.head(6) / o.sum()).round(3).tolist()} keys {nk.reindex(o.head(6).index).tolist()}')
    qq = kwq.assign(o=op[kwq.k.to_numpy()]).groupby(['q', 'o']).wins.sum()
    P(pd.DataFrame({'hhi': qq.groupby('q').apply(hh), 'top1': qq.groupby('q').apply(lambda v: v.max() / v.sum()), 'top3': qq.groupby('q').apply(lambda v: v.nlargest(3).sum() / v.sum()), 'ops_winning': qq[qq > 0].groupby('q').size()}).round(3).T.to_string())
w = s[s.has_win == 1]
for nm in OPS:
    g = w.groupby('q'); P(f'\n== margins by quarter, definition {nm} (%, slots with the winning block found) ==')
    P(pd.DataFrame({'slots': g.size(), 'stand_med': 100 * g['stand_' + nm].median(), 'stand_p10': 100 * g['stand_' + nm].quantile(.1), 'stand_p25': 100 * g['stand_' + nm].quantile(.25), 'stand_p75': 100 * g['stand_' + nm].quantile(.75), 'stand_p90': 100 * g['stand_' + nm].quantile(.9),
        'pre_med': 100 * g['pre_' + nm].median(), 'any_med': 100 * g['any_' + nm].median(), 'rival_higher': g['stand_' + nm].apply(lambda v: (v.dropna() < 0).mean()), 'exact_tie': g['stand_' + nm].apply(lambda v: (v.dropna() == 0).mean()), 'any_tie': g['any_' + nm].apply(lambda v: (v.dropna() == 0).mean()),
        'no_rival': g['stand_' + nm].apply(lambda v: v.isna().mean()), 'riv1_mean': g['riv1_' + nm].mean()}).round(3).to_string())
    a = w; P('all:', 'stand median', round(100 * a['stand_' + nm].median(), 3), 'p10', round(100 * a['stand_' + nm].quantile(.1), 3), 'p90', round(100 * a['stand_' + nm].quantile(.9), 3), '| pre', round(100 * a['pre_' + nm].median(), 3), '| any', round(100 * a['any_' + nm].median(), 3), '| rival higher', round((a['stand_' + nm].dropna() < 0).mean(), 3))
txt = B.getvalue(); open(f'{RES}/summary2_for_claude.txt', 'w').write(txt); print(txt[:2500])
s.drop(columns=['q']).to_parquet('/content/by_slot2.parquet', index=False, compression='zstd')
cd = pd.concat(cands, ignore_index=True).merge(s[['slot', 'day']], on=['slot', 'day'], how='inner').drop(columns='day'); cd.to_parquet('/content/cand.parquet', index=False, compression='zstd')
pd.DataFrame({'pubkey': keys, 'code': range(NK), **{'op' + nm: op for nm, op in OPS.items() if nm != 'KEY'}}).to_csv('/content/opmap.csv', index=False)
T.assign(a=[keys[i] for i in T.a], b=[keys[i] for i in T.b]).to_csv('/content/tie_pairs.csv', index=False); H.assign(a=[p[0] for p in H.p], b=[p[1] for p in H.p]).drop(columns='p').to_csv('/content/hash_pairs.csv', index=False)
out = f'{RES}/results2_for_claude.zip'
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in ['by_slot2.parquet', 'opmap.csv', 'tie_pairs.csv', 'hash_pairs.csv']: z.write('/content/' + f, f)
    z.write(f'{RES}/summary2_for_claude.txt', 'summary2.txt')
out2 = f'{RES}/results3_candidates.zip'
with zipfile.ZipFile(out2, 'w') as z: z.write('/content/cand.parquet', 'cand.parquet')
print('DONE:', len(s), 'slots;', round(os.path.getsize(out) / 1e6), 'MB ->', out, '| and', round(os.path.getsize(out2) / 1e6), 'MB ->', out2)
