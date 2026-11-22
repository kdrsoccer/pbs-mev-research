import sys, numpy as np, pandas as pd
R1, R2, OUTD = sys.argv[1], sys.argv[2], sys.argv[3]; pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300); pd.set_option('display.max_columns', 40)
c = pd.read_parquet(R2 + '/cand.parquet'); T = pd.read_csv(R2 + '/tie_pairs.csv'); O = pd.read_csv(R2 + '/opmap.csv'); D = pd.read_csv(R2 + '/days.csv')
K = pd.read_parquet(R1 + '/keys.parquet'); L = pd.read_parquet(R1 + '/links.parquet')
bs = pd.read_parquet(R1 + '/by_slot.parquet'); bs['day'] = bs.day.astype(str)
c = c.merge(D, on='d').merge(bs[['slot', 'day']], on=['slot', 'day']).drop(columns=['d', 'day'])
dl = pd.read_parquet(sys.argv[4], columns=['slot', 'value_wei', 'builder_pubkey']); dl['builder_pubkey'] = dl.builder_pubkey.astype(str)
code = O.set_index('pubkey').code; NK = len(O); opB = O.sort_values('code').opB.to_numpy()
T['rate'] = T.ties / np.minimum(T.pres_a, T.pres_b); T['ca'] = T.a.map(code); T['cb'] = T.b.map(code)
def comps(base, pairs):
    par = base.copy().tolist()
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in pairs: par[find(a)] = find(b)
    return np.array([find(x) for x in range(NK)])
tp = lambda sel: list(zip(sel.ca.tolist(), sel.cb.tolist()))
anyhash = [(code[a], code[b]) for a, b in L[['a', 'b']].drop_duplicates().itertuples(index=False)]
DEF = {'KEY': np.arange(NK), 'B hash': opB, 'A hash-any': comps(opB, anyhash), 'C hash+ties5%': comps(opB, tp(T[(T.ties >= 200) & (T.rate >= .05)])), 'D hash+ties2%': comps(opB, tp(T[(T.ties >= 200) & (T.rate >= .02)])),
       'E hash+ties1%': comps(opB, tp(T[(T.ties >= 200) & (T.rate >= .01)])), 'F hash-any+ties2%': comps(comps(opB, anyhash), tp(T[(T.ties >= 200) & (T.rate >= .02)]))}
kq = K.assign(q=pd.PeriodIndex(pd.to_datetime(K.day), freq='Q').astype(str), k=K.pubkey.map(code)); hh = lambda v: 10000 * ((v / v.sum()) ** 2).sum()
bs = bs.merge(dl, on='slot'); bs['wk'] = bs.builder_pubkey.map(code); bs['q'] = pd.PeriodIndex(pd.to_datetime(bs.day), freq='Q').astype(str); bs['ter'] = pd.qcut(bs.value_wei, 3, labels=['low', 'mid', 'high'])
res = bs[['slot', 'day', 'q', 'ter', 'has_win', 'wt', 'n_all', 'nb_all', 'value_wei', 'wnext_gap']].copy()
cw = c.merge(bs[['slot', 'wk']], on='slot')
print('=== concentration of delivered blocks (HHI) and operator counts ===')
rows = {}
for nm, op in DEF.items():
    tot = kq.assign(o=op[kq.k.to_numpy()]).groupby('o').wins.sum().sort_values(ascending=False); qq = kq.assign(o=op[kq.k.to_numpy()]).groupby(['q', 'o']).wins.sum()
    nk = pd.Series(1, index=range(NK)).groupby(op).sum()
    rows[nm] = {'operators': len(np.unique(op[np.unique(kq.k)])), 'multi-key': int((nk > 1).sum()), 'HHI all': round(hh(tot)), 'top1': round(tot.iloc[0] / tot.sum(), 3), 'top3': round(tot.head(3).sum() / tot.sum(), 3), 'top5': round(tot.head(5).sum() / tot.sum(), 3), **{q: round(v) for q, v in qq.groupby('q').apply(hh).items()}}
    rows[nm]['top6 shares'] = (tot.head(6) / tot.sum()).round(3).tolist(); rows[nm]['top6 keys'] = nk.reindex(tot.head(6).index).tolist()
print(pd.DataFrame(rows).T.to_string())
print('\n=== margins: winner vs best rival operator, by definition (slots with the winning block found) ===')
out = {}
for nm, op in DEF.items():
    if nm == 'KEY': continue
    x = cw[op[cw.o.to_numpy()] != op[cw.wk.to_numpy()]].copy(); x['om'] = op[x.o.to_numpy()]; g = x.groupby('slot')
    r = pd.DataFrame({'stand': 1 - g.st_r.max(), 'any': 1 - g.any_r.max()}); y = x[x.st_r >= 0.99]; r['riv1'] = y.drop_duplicates(['slot', 'om']).groupby('slot').size(); r['nops8'] = x.drop_duplicates(['slot', 'om']).groupby('slot').size()
    r = res[['slot', 'q', 'has_win']].merge(r.reset_index(), on='slot', how='left'); r['riv1'] = r.riv1.fillna(0); w = r[r.has_win == 1]
    tag = nm.split()[0]; res['stand_' + tag] = r.stand.to_numpy(); res['any_' + tag] = r['any'].to_numpy(); res['riv1_' + tag] = r.riv1.to_numpy()
    g = w.groupby('q'); t = pd.DataFrame({'stand_med%': 100 * g.stand.median(), 'p10': 100 * g.stand.quantile(.1), 'p25': 100 * g.stand.quantile(.25), 'p75': 100 * g.stand.quantile(.75), 'p90': 100 * g.stand.quantile(.9), 'any_med%': 100 * g['any'].median(),
        'rival_higher': g.stand.apply(lambda v: (v.dropna() < 0).mean()), 'rival_higher_1pct': g.stand.apply(lambda v: (v.dropna() < -0.01).mean()), 'exact_tie': g.stand.apply(lambda v: (v.dropna() == 0).mean()), 'no_rival_in_8': g.stand.apply(lambda v: v.isna().mean()), 'rivals_within_1%': g.riv1.mean()}).round(3)
    a = pd.DataFrame({'stand_med%': 100 * w.stand.median(), 'p10': 100 * w.stand.quantile(.1), 'p25': 100 * w.stand.quantile(.25), 'p75': 100 * w.stand.quantile(.75), 'p90': 100 * w.stand.quantile(.9), 'any_med%': 100 * w['any'].median(), 'rival_higher': (w.stand.dropna() < 0).mean(),
        'rival_higher_1pct': (w.stand.dropna() < -0.01).mean(), 'exact_tie': (w.stand.dropna() == 0).mean(), 'no_rival_in_8': w.stand.isna().mean(), 'rivals_within_1%': w.riv1.mean()}, index=['ALL']).round(3)
    print(f'\n[{nm}]'); print(pd.concat([t, a]).to_string())
res.to_parquet(OUTD + '/allbids_final.parquet', index=False, compression='zstd')
pd.DataFrame({'pubkey': O.sort_values('code').pubkey.to_numpy(), **{nm.split()[0]: op for nm, op in DEF.items() if nm != 'KEY'}}).to_csv(OUTD + '/operator_map.csv', index=False)
m = DEF['D hash+ties2%']; newm = T[(T.ties >= 200) & (T.rate >= .02)]; print('\ntie links used in D:', len(newm), '| of which join different hash-operators:', int((opB[newm.ca] != opB[newm.cb]).sum()))
