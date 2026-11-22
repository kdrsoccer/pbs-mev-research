"""Operators from public builder labels plus shared blocks; validation of the behavioural linking rules against the labels.

Labels: Data Always, mevboost-data (https://github.com/dataalways/mevboost-data): mapping_pubkey_builder_name.csv and, for keys not in it, the modal
extra_data label (builder_label) of the key's delivered blocks.  data/builder_labels.csv holds both for the 803 keys of this sample
(build_labels() below rebuilds it from a clone of that repository).

Rules written to data/operator_map.csv:
  B  shared blocks only (no labels)                                 D  B plus exact value ties (no labels)
  L  labels plus shared blocks: keys with the same label are one operator; an unlabelled key joins the operator it shares blocks with   (baseline)
  M  L with beaverbuild, BuilderNet and Flashbots treated as one operator (beaverbuild and Flashbots operate BuilderNet)
and stand_X / any_X / riv1_X for X in L, M added to data/all_bids_by_slot.parquet.
Usage: python3 7_all_bids/label_operators.py            (from the repository root)"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import paths
pd.set_option('display.width', 230); pd.set_option('display.max_rows', 200)
D = paths.DATA; R = D + 'all_bids_intermediate/'
NORM = {'beaverbuild': 'beaverbuild.org', 'Titan Relay': 'Titan'}; PARTNERS = {'beaverbuild.org': 'buildernet', 'flashbots': 'buildernet'}

def build_labels(repo, keys):
    """repo = path of a clone of dataalways/mevboost-data"""
    import glob
    mp = pd.read_csv(repo + '/mapping_pubkey_builder_name.csv').drop_duplicates('pubkey').set_index('pubkey').builder_name
    fs = [f for f in sorted(glob.glob(repo + '/data/*.parquet')) if '2024-09-10' <= f.split('__')[-1][:10] <= '2026-06-02']
    df = pd.concat([pd.read_parquet(f, columns=['slot', 'builder_label', 'builder_pubkey']) for f in fs], ignore_index=True).dropna(subset=['builder_pubkey']).drop_duplicates(['slot', 'builder_pubkey'])
    g = df.groupby('builder_pubkey').builder_label
    t = pd.DataFrame({'pubkey': keys}).set_index('pubkey').join(mp.rename('map_name')).join(g.agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None).rename('xd_label'))
    t = t.join(g.agg(lambda s: s.value_counts(normalize=True).iloc[0] if s.notna().any() else np.nan).rename('xd_purity')).join(df.groupby('builder_pubkey').size().rename('blocks'))
    t.reset_index().to_csv(D + 'builder_labels.csv', index=False)

O = pd.read_csv(R + 'opmap.csv'); code = O.set_index('pubkey').code; NK = len(O); opB = O.sort_values('code').opB.to_numpy(); keys = O.sort_values('code').pubkey.to_numpy()
if len(sys.argv) > 1: build_labels(sys.argv[1], keys)
t = pd.read_csv(D + 'builder_labels.csv').set_index('pubkey'); t['label'] = t.map_name.fillna(t.xd_label).replace(NORM)
K = pd.read_parquet(D + 'all_bids_keys.parquet'); kb = K.groupby('pubkey').agg(bids=('bids', 'sum'), wins=('wins', 'sum')); t = t.join(kb); t[['bids', 'wins']] = t[['bids', 'wins']].fillna(0)
old = pd.read_csv(D + 'operator_map.csv').set_index('pubkey')
print(f'keys {len(t)}; labelled {t.label.notna().sum()} ({t.map_name.notna().sum()} in the mapping file, {(t.map_name.isna() & t.xd_label.notna()).sum()} by extra_data only); labels {t.label.nunique()}')
print(f'labelled keys: {100 * t.bids[t.label.notna()].sum() / t.bids.sum():.1f}% of bids, {100 * t.wins[t.label.notna()].sum() / t.wins.sum():.2f}% of wins')
both = t.dropna(subset=['map_name', 'xd_label']); print(f'keys with both sources: {len(both)}; they agree for {(both.map_name.replace(NORM).str.lower().str[:5] == both.xd_label.replace(NORM).str.lower().str[:5]).sum()}')
print('\n=== behavioural rules against the labels (labelled keys only) ===')
lab = t[t.label.notna()]; V = {}
for X in ['B', 'D', 'F']:
    c = old[X].reindex(lab.index); row = {}
    for wgt in ['bids', 'wins']:
        g = pd.DataFrame({'o': c, 'l': lab.label, 'w': lab[wgt]}).groupby(['o', 'l']).w.sum().reset_index()
        row[f'purity_{wgt}'] = g.groupby('o').w.max().sum() / g.w.sum(); row[f'completeness_{wgt}'] = g.groupby('l').w.max().sum() / g.w.sum()
    mixed = pd.DataFrame({'o': c, 'l': lab.label}).groupby('o').l.nunique(); row['operators with two or more labels'] = int((mixed > 1).sum()); row['operators'] = c.nunique(); V[X] = row
    for o in mixed[mixed > 1].index: print(f'  rule {X}: operator {o} mixes', lab[c == o].groupby('label').bids.sum().sort_values(ascending=False).astype(int).to_dict())
V = pd.DataFrame(V).T; print(V.round(4).to_string()); V.to_csv(paths.WORK + 'label_validation.csv')

def comps(base, pairs):
    par = base.copy().tolist()
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in pairs: par[find(a)] = find(b)
    return np.array([find(x) for x in range(NK)])
def by_label(lbl):
    lc = pd.Series(lbl.values, index=lbl.index.map(code)).dropna(); pairs = []
    for _, idx in lc.groupby(lc).groups.items(): idx = [int(i) for i in idx]; pairs += [(idx[0], j) for j in idx[1:]]
    return comps(opB, pairs)
L = by_label(t.label.dropna()); M = by_label(t.label.dropna().replace(PARTNERS))
out = old[[c for c in old.columns if c not in ('L', 'M', 'label')]].copy(); out['L'] = pd.Series(L, index=keys).reindex(out.index); out['M'] = pd.Series(M, index=keys).reindex(out.index); out['label'] = t.label.reindex(out.index)
out.reset_index().to_csv(D + 'operator_map.csv', index=False); print('\noperators: L', len(np.unique(L[np.unique(K.pubkey.map(code))])), ' M', len(np.unique(M[np.unique(K.pubkey.map(code))])))

c = pd.read_parquet(R + 'cand.parquet'); Dd = pd.read_csv(R + 'days.csv'); bs = pd.read_parquet(R + 'by_slot.parquet', columns=['slot', 'day', 'has_win']); bs['day'] = bs.day.astype(str)
c = c.merge(Dd, on='d').merge(bs[['slot', 'day']], on=['slot', 'day'])[['slot', 'o', 'any_r', 'st_r']]
dl = pd.read_parquet(D + 'delivered_slim.parquet', columns=['slot', 'builder_pubkey']); bs['wk'] = bs.slot.map(dl.set_index('slot').builder_pubkey.astype(str).map(code)); c = c.merge(bs[['slot', 'wk']], on='slot')
ab = pd.read_parquet(D + 'all_bids_by_slot.parquet'); ab = ab[[x for x in ab.columns if not x.endswith(('_L', '_M'))]]
for nm, op in [('L', L), ('M', M)]:
    x = c[op[c.o.to_numpy()] != op[c.wk.to_numpy()]].copy(); x['om'] = op[x.o.to_numpy()]; g = x.groupby('slot')
    r = pd.DataFrame({'stand_' + nm: 1 - g.st_r.max(), 'any_' + nm: 1 - g.any_r.max()}); r['riv1_' + nm] = x[x.st_r >= 0.99].drop_duplicates(['slot', 'om']).groupby('slot').size()
    ab = ab.merge(r.reset_index(), on='slot', how='left'); ab['riv1_' + nm] = ab['riv1_' + nm].fillna(0); w = ab[ab.has_win == 1]
    print(f'[{nm}] median standing margin {100 * w["stand_" + nm].median():.3f}%   rival higher {100 * (w["stand_" + nm].dropna() < 0).mean():.1f}%   no rival within 1% {100 * (w["riv1_" + nm] == 0).mean():.1f}%')
ab.to_parquet(D + 'all_bids_by_slot.parquet', index=False, compression='zstd', compression_level=19)
