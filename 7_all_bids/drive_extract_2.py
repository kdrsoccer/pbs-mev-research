import os, glob, zipfile, collections, numpy as np, pandas as pd
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/pbs_all_bids'; RES = OUT + '/results'
cat = lambda sub: pd.concat([pd.read_parquet(f).assign(day=os.path.basename(f)[:10]) for f in sorted(glob.glob(f'{RES}/{sub}/*.parquet'))], ignore_index=True)
K = cat('keys'); L = cat('links')
dl = pd.read_parquet(f'{OUT}/delivered_slim.parquet', columns=['slot', 'value_wei', 'builder_pubkey']); dl['builder_pubkey'] = dl.builder_pubkey.astype(str)
keys = sorted(set(K.pubkey) | set(dl.builder_pubkey) | set(L.a) | set(L.b)); code = {k: i for i, k in enumerate(keys)}; NK = len(keys)
L['p'] = [tuple(sorted(x)) for x in zip(L.a, L.b)]; db = K.set_index(['day', 'pubkey']).bids
L['mn'] = np.minimum(db.reindex(pd.MultiIndex.from_frame(L[['day', 'a']])).to_numpy(), db.reindex(pd.MultiIndex.from_frame(L[['day', 'b']])).to_numpy())
H = L.groupby('p').agg(n=('n', 'sum'), mn=('mn', 'sum')).reset_index(); H = H[H.n / H.mn >= 0.002]
par = list(range(NK))
def find(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for p in H.p: par[find(code[p[0]])] = find(code[p[1]])
op = np.array([find(x) for x in range(NK)])
dv = dl.set_index('slot').value_wei; wk = dl.set_index('slot').builder_pubkey.map(code)
ties = collections.Counter(); pres = np.zeros(NK, 'int64'); cands = []
tops = sorted(glob.glob(f'{RES}/top/*.parquet'))
for i, f in enumerate(tops):
    t = pd.read_parquet(f); t['k'] = t.pubkey.astype(str).map(code).astype('int32'); t['dv'] = t.slot.map(dv).to_numpy(); t['wk'] = t.slot.map(wk).to_numpy()
    pres += np.bincount(t.k.to_numpy(), minlength=NK)
    m = t[t.duplicated(['slot', 'best_le'], keep=False)]
    a = m.groupby(['slot', 'best_le']).k.transform('min').to_numpy(); b = m.k.to_numpy(); ties.update(zip(a[a != b].tolist(), b[a != b].tolist()))
    o = op[t.k.to_numpy()]; x = t.assign(o=o)[o != op[t.wk.to_numpy()]]
    c = x.groupby(['slot', 'o']).agg(a=('best_le', 'max'), st=('standing', 'max'), dv=('dv', 'first')).reset_index(); c['m'] = c[['a', 'st']].max(axis=1)
    c = c.sort_values(['slot', 'm'], ascending=[True, False]); c = c[c.groupby('slot').cumcount() < 8]
    cands.append(pd.DataFrame({'slot': c.slot.astype('int32'), 'o': c.o.astype('int16'), 'any_r': (c.a / c.dv).astype('float32'), 'st_r': (c.st / c.dv).astype('float32'), 'd': np.int16(i)}))
    if i % 50 == 0: print(i, len(tops), flush=True)
pd.concat(cands, ignore_index=True).to_parquet('/content/cand.parquet', index=False, compression='zstd')
pd.DataFrame([(keys[a], keys[b], n, pres[a], pres[b]) for (a, b), n in ties.items()], columns=['a', 'b', 'ties', 'pres_a', 'pres_b']).to_csv('/content/tie_pairs.csv', index=False)
pd.DataFrame({'pubkey': keys, 'code': range(NK), 'opB': op}).to_csv('/content/opmap.csv', index=False)
pd.DataFrame({'d': range(len(tops)), 'day': [os.path.basename(f)[:10] for f in tops]}).to_csv('/content/days.csv', index=False)
out = f'{RES}/results2_for_claude.zip'
with zipfile.ZipFile(out, 'w') as z:
    for f in ['cand.parquet', 'tie_pairs.csv', 'opmap.csv', 'days.csv']: z.write('/content/' + f, f)
print('DONE:', round(os.path.getsize(out) / 1e6), 'MB ->', out)
