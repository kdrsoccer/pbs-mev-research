import os, time, zipfile, urllib.request, shutil
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.csv as pacsv, pyarrow.compute as pc

URL = 'https://bidarchive.relayscan.io/ethereum/mainnet/{m}/{d}_all.csv.zip'
COLS = ['slot', 'slot_t_ms', 'value', 'block_hash', 'builder_pubkey']
_LUT = np.zeros(256, 'uint64')
for _i, _c in enumerate('0123456789abcdef'): _LUT[ord(_c)] = _i
for _i, _c in enumerate('ABCDEF'): _LUT[ord(_c)] = 10 + _i
_SH = np.arange(60, -1, -4).astype('uint64')


def hash64(col):
    """First 16 hex digits of each 0x-prefixed block hash as uint64 (0 when missing)."""
    n = len(col)
    if n and col.null_count == 0 and col.type == pa.string():
        offs = np.frombuffer(col.buffers()[1], dtype='int32', count=n + 1, offset=col.offset * 4)
        if (np.diff(offs) == 66).all():
            data = np.frombuffer(col.buffers()[2], dtype='uint8', count=66 * n, offset=int(offs[0])).reshape(n, 66)[:, 2:18]
            return (_LUT[data] << _SH).sum(axis=1, dtype='uint64')
    return np.array([int(x[2:18], 16) if x and len(x) >= 18 else 0 for x in col.to_pylist()], dtype='uint64')


def load_delivered(path):
    """Delivered (winning) block of every matched slot: slot, value in wei, first 18 chars of the block hash, builder."""
    d = pd.read_parquet(path).sort_values('slot').reset_index(drop=True)
    pubs = list(d.builder_pubkey.astype(str).unique())
    code = {p: i for i, p in enumerate(pubs)}                      # global builder dictionary, seeded with the winners
    return dict(slot=d.slot.to_numpy('int64'), dv=d.value_wei.to_numpy('float64'), h=np.array([int(x[2:18], 16) for x in d.hash18], dtype='uint64'),
                b=d.builder_pubkey.astype(str).map(code).to_numpy('int32'), code=code)


def days_needed(D):
    t = pd.to_datetime(1606824023 + 12 * D['slot'], unit='s')
    return sorted({f'{x:%Y-%m-%d}' for x in pd.Series(t).dt.normalize().unique()})


def download(day, dest):
    url = URL.format(m=day[:7], d=day)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest + '.part', 'wb') as f:
        shutil.copyfileobj(r, f, 16 << 20)
    os.replace(dest + '.part', dest)
    return os.path.getsize(dest)


def read_day(zip_path, D):
    """One pass over the day's CSV. Keeps only rows of delivered slots, in compact arrays (about 21 bytes per row)."""
    S, T, V, B, H = [], [], [], [], []
    code = D['code']; rows = 0
    with zipfile.ZipFile(zip_path) as z:
        f = z.open(z.namelist()[0])
        rd = pacsv.open_csv(f, read_options=pacsv.ReadOptions(block_size=64 << 20),
                            convert_options=pacsv.ConvertOptions(include_columns=COLS, column_types={'slot': pa.float64(), 'slot_t_ms': pa.float64(), 'value': pa.float64(), 'block_hash': pa.string(), 'builder_pubkey': pa.string()}, strings_can_be_null=True))
        for batch in rd:
            rows += batch.num_rows
            slot = batch.column('slot').to_numpy(zero_copy_only=False); val = batch.column('value').to_numpy(zero_copy_only=False)
            ok = np.isfinite(slot) & np.isfinite(val) & (val < 1e21)
            si = np.searchsorted(D['slot'], np.where(ok, slot, -1)); si[si >= len(D['slot'])] = 0
            ok &= D['slot'][si] == slot
            if not ok.any(): continue
            keep = np.flatnonzero(ok); si = si[keep].astype('int32'); val = val[keep]
            t = batch.column('slot_t_ms').to_numpy(zero_copy_only=False)[keep].astype('float32')
            enc = pc.dictionary_encode(batch.column('builder_pubkey').take(pa.array(keep)))
            dic = enc.dictionary.to_pylist()
            for p in dic:
                if p is not None and p not in code: code[p] = len(code)
            lut = np.array([code[p] if p is not None else -1 for p in dic] + [-1], dtype='int32')
            ind = enc.indices.to_numpy(zero_copy_only=False); ind = np.where(np.isnan(ind.astype('float64')), len(dic), ind).astype('int64') if enc.null_count else ind.astype('int64')
            b = lut[ind]
            S.append(si); T.append(t); V.append(val); B.append(b); H.append(hash64(batch.column('block_hash').take(pa.array(keep))))
    if not S: return None, rows
    x = pd.DataFrame({'si': np.concatenate(S), 't': np.concatenate(T), 'v': np.concatenate(V), 'b': np.concatenate(B), 'h': np.concatenate(H)})
    x['win'] = x.h.to_numpy() == D['h'][x.si.to_numpy()]                                   # the delivered block itself
    return x, rows


def slot_stats(x, D):
    """Per delivered slot, from every recorded bid. 'Operator' = builder keys that submitted the same block that day (one firm, several keys).
       lead_any_op    winner's margin over the best bid at or below the winning value from another operator, whenever it arrived
       lead_pre_op    the same, counting only bids received no later than the winning bid
       stand_op       winner's margin over the best STANDING bid of another operator when the winning bid arrived: each key's most recent
                      bid at that moment (a newer bid replaces a key's older one). Negative = another operator's standing bid was higher.
       lead_any / lead_pre   the first two by key instead of by operator (what a key-level analysis would report)
       nb_all, nop_all       keys / operators with any bid;  nop_pre operators with a bid no later than the winning bid
       nop_stand_1pct        other operators whose standing bid is within 1% of the winning value"""
    si = x.si.to_numpy(); dv = D['dv'][si]; x = x.assign(dv=dv)
    kn0 = x[x.b >= 0]
    # ---- operators for the day: union-find over keys sharing a block hash, plus delivered-record key vs archive key of the winning block
    u = kn0[['h', 'b']].drop_duplicates(); m = u[u.duplicated('h', keep=False)]
    pairs = pd.DataFrame({'a': m.groupby('h').b.transform('min').to_numpy(), 'b': m.b.to_numpy()}); pairs = pairs[pairs.a != pairs.b]
    ww = kn0[kn0.win & (kn0.b.to_numpy() != D['b'][kn0.si.to_numpy()])]; pairs = pd.concat([pairs, pd.DataFrame({'a': D['b'][ww.si.to_numpy()], 'b': ww.b.to_numpy()})])
    links = pairs.value_counts().reset_index(name='n')
    par = np.arange(len(D['code']))
    def find(a):
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    for a, b in links[['a', 'b']].itertuples(index=False): par[find(a)] = find(b)
    op_of = np.array([find(k) for k in range(len(par))] + [-1])                               # last entry: unknown key (-1)
    x['op'] = op_of[x.b.to_numpy()]; wop = op_of[D['b'][si]]
    x['isw'] = x.b.to_numpy() == D['b'][si]; x['isop'] = x.op.to_numpy() == wop
    x['low'] = x.v.to_numpy() <= dv * (1 + 1e-12)
    g = x.groupby('si', sort=True)
    r = pd.DataFrame({'n_all': g.size(), 'n_le': g['low'].sum(), 'n_winner_op': g.isop.sum(), 'has_win': g.win.max().astype('int8'), 'top_any': g.v.max()})
    r['wt'] = x[x.win].groupby('si').t.min()
    x['pre'] = x.t.to_numpy() <= r.wt.reindex(si).to_numpy()                                  # False when either time is missing
    r['n_pre'] = x.groupby('si').pre.sum()
    kn = x[x.b >= 0]
    r['nb_all'] = kn.drop_duplicates(['si', 'b']).groupby('si').size(); r['nop_all'] = kn.drop_duplicates(['si', 'op']).groupby('si').size()
    r['nop_pre'] = kn[kn.pre].drop_duplicates(['si', 'op']).groupby('si').size()
    for nm, sub in [('', kn[~kn.isw]), ('_op', kn[~kn.isop])]:
        lo = sub[sub['low']]
        r['best_any' + nm] = lo.groupby('si').v.max(); r['best_pre' + nm] = lo[lo.pre].groupby('si').v.max()
        r['n_pre_gt' + nm] = sub[sub.pre & ~sub['low']].groupby('si').size()
    # ---- standing bids when the winning bid arrived: each key's most recent bid received no later than that
    kp = kn[kn.pre]
    st = kp.loc[kp.groupby(['si', 'b']).t.idxmax(), ['si', 'b', 'op', 'v', 'dv', 'isop', 'isw']]
    so = st[~st.isop]
    r['stand_best_op'] = so.groupby('si').v.max(); r['stand_best_key'] = st[~st.isw].groupby('si').v.max()
    r['nop_stand_1pct'] = so[so.v >= 0.99 * so.dv].drop_duplicates(['si', 'op']).groupby('si').size()
    # ---- the proposer chose between the winning bid's arrival and the next bid from the same key (which replaces it):
    #      standing bids of other operators just before that next bid give the other end of the window
    wk = kn[kn.win].sort_values('t').drop_duplicates('si').set_index('si').b                  # key on the winning row
    nx = kn[(kn.b.to_numpy() == wk.reindex(kn.si.to_numpy()).to_numpy()) & (kn.t.to_numpy() > r.wt.reindex(kn.si.to_numpy()).to_numpy())]
    r['wnext'] = nx.groupby('si').t.min()
    k2 = kn[kn.t.to_numpy() < r.wnext.reindex(kn.si.to_numpy()).to_numpy()]
    s2 = k2.loc[k2.groupby(['si', 'b']).t.idxmax(), ['si', 'v', 'isop']]
    r['stand_best_op_next'] = s2[~s2.isop].groupby('si').v.max()
    r = r.reset_index(); r['slot'] = D['slot'][r.si.to_numpy()]; r['dv'] = D['dv'][r.si.to_numpy()]
    r['stand_op_next'] = (r.dv - r.stand_best_op_next) / r.dv; r['wnext_gap'] = r.wnext - r.wt
    for nm in ['', '_op']:
        r['lead_any' + nm] = (r.dv - r['best_any' + nm]) / r.dv; r['lead_pre' + nm] = (r.dv - r['best_pre' + nm]) / r.dv
    r['stand_op'] = (r.dv - r.stand_best_op) / r.dv; r['stand_key'] = (r.dv - r.stand_best_key) / r.dv; r['top_ratio'] = r.top_any / r.dv
    for c in ['nb_all', 'nop_all', 'nop_pre', 'n_pre_gt', 'n_pre_gt_op', 'nop_stand_1pct']: r[c] = r[c].fillna(0).astype('int32')
    # ---- kept for relinking keys across days later: links, and each slot's 40 best keys below the winning value
    inv = np.empty(len(D['code']), object)
    for k, v in D['code'].items(): inv[v] = k
    links['a'] = inv[links.a.to_numpy()]; links['b'] = inv[links.b.to_numpy()]
    low = kn[kn['low']]
    tb = low.groupby(['si', 'b']).v.max().rename('best_le').reset_index().merge(low[low.pre].groupby(['si', 'b']).v.max().rename('best_pre_le').reset_index(), on=['si', 'b'], how='left')
    tb = tb.merge(st[['si', 'b', 'v']].rename(columns={'v': 'standing'}), on=['si', 'b'], how='left').sort_values(['si', 'best_le'], ascending=[True, False]); tb = tb[tb.groupby('si').cumcount() < 40]
    top = pd.DataFrame({'slot': D['slot'][tb.si.to_numpy()].astype('int32'), 'pubkey': pd.Categorical(inv[tb.b.to_numpy()]), 'best_le': tb.best_le.to_numpy(), 'best_pre_le': tb.best_pre_le.to_numpy(), 'standing': tb.standing.to_numpy()})
    # ---- per key for the day: how much and when it bids (for operator-level bidding styles)
    kg = kn.groupby('b'); ks = pd.DataFrame({'bids': kg.size(), 'slots': kg.si.nunique(), 't_median': kg.t.median(), 'after_start': kn.assign(a=kn.t > 0).groupby('b').a.mean(),
                                             'above_winner': kn.assign(a=~kn['low']).groupby('b').a.mean(), 'v_median': kg.v.median()}).reset_index()
    sl = np.unique(kn.si.to_numpy()); ks['wins'] = ks.b.map(pd.Series(D['b'][sl]).value_counts()).fillna(0).astype('int32'); ks['day_slots'] = len(sl)
    ks['operator'] = inv[op_of[ks.b.to_numpy()]]; ks['pubkey'] = inv[ks.b.to_numpy()]; ks = ks.drop(columns='b')
    cols = ['slot', 'n_all', 'n_le', 'n_pre', 'n_winner_op', 'nb_all', 'nop_all', 'nop_pre', 'nop_stand_1pct', 'has_win', 'wt', 'lead_any', 'lead_pre', 'lead_any_op', 'lead_pre_op', 'stand_key', 'stand_op', 'stand_op_next', 'wnext_gap', 'n_pre_gt', 'n_pre_gt_op', 'top_ratio']
    return r[cols], top, links, ks


def process_day(day, D, out_dir, tmp='/tmp', local_zip=None, log=print):
    out = f'{out_dir}/days/{day}.parquet'
    if os.path.exists(out): return 'done already'
    t0 = time.time(); zp = local_zip or f'{tmp}/{day}_all.csv.zip'
    size = os.path.getsize(zp) if local_zip else download(day, zp); t1 = time.time()
    try:
        x, rows = read_day(zp, D); t2 = time.time()
        if x is None: log(f'{day}: no delivered slots in file ({rows:,} rows)'); return 'empty'
        r, top, links, ks = slot_stats(x, D)
        top.to_parquet(f'{out_dir}/top/{day}.parquet', index=False); links.to_parquet(f'{out_dir}/links/{day}.parquet', index=False); ks.to_parquet(f'{out_dir}/keys/{day}.parquet', index=False)
        r.to_parquet(out + '.tmp', index=False); os.replace(out + '.tmp', out)                # written last: its presence marks the day as done
    finally:
        if not local_zip and os.path.exists(zp): os.remove(zp)
    log(f'{day}: {size / 1e9:.2f} GB zip, {rows:,} rows, {len(x):,} kept, {len(r):,} slots | download {t1 - t0:.0f}s, read {t2 - t1:.0f}s, stats {time.time() - t2:.0f}s')
    return 'ok'


def summarize(out_dir, delivered_path=None, log=print):
    import glob
    fs = sorted(glob.glob(f'{out_dir}/days/*.parquet'))
    if not fs: log('no days done yet'); return None
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).sort_values(['slot', 'n_all'], ascending=[True, False]).drop_duplicates('slot')
    d.to_parquet(f'{out_dir}/all_bids_by_slot.parquet', index=False)
    w = d[d.has_win == 1]; q = lambda x: f'median {100 * x.median():.3f}%  p10 {100 * x.quantile(.1):.3f}%  p90 {100 * x.quantile(.9):.3f}%  (n={x.notna().sum():,})'
    log(f'days {len(fs)}; slots {len(d):,}; winning block found in the file: {d.has_win.mean():.1%}; winning bid time: median {w.wt.median():.0f} ms')
    log(f'bids per slot: median {d.n_all.median():.0f}; keys bidding: median {d.nb_all.median():.0f}; operators bidding: median {d.nop_all.median():.0f}; operators with a bid before the winning bid: median {w.nop_pre.median():.0f}')
    log('BY OPERATOR')
    log('  margin over best lower bid, any arrival time   : ' + q(d.lead_any_op))
    log('  margin over best lower bid, arrived earlier    : ' + q(w.lead_pre_op))
    log('  margin over best STANDING bid at winning time  : ' + q(w.stand_op))
    log('  same, just before the winning key bid again    : ' + q(w.stand_op_next))
    log(f'  time from the winning bid to the same key\'s next bid: median {w.wnext_gap.median():.0f} ms, p90 {w.wnext_gap.quantile(.9):.0f} ms; the key never bid again in {w.wnext_gap.isna().mean():.1%} of slots')
    log(f'  another operator\'s standing bid was higher just before that next bid: {(w.stand_op_next < 0).mean():.1%} of slots')
    log(f'  another operator had a higher standing bid when the winning bid arrived: {(w.stand_op < 0).mean():.1%} of slots; no other operator had bid yet: {w.stand_op.isna().mean():.1%}')
    log(f'  other operators with a standing bid within 1% of the winner: median {w.nop_stand_1pct.median():.0f}, mean {w.nop_stand_1pct.mean():.2f}')
    log(f'  another operator had bid above the winning value earlier (standing or not): {(w.n_pre_gt_op > 0).mean():.1%} of slots')
    log('BY KEY (for comparison)')
    log('  margin over best lower bid, any arrival time   : ' + q(d.lead_any))
    log('  margin over best lower bid, arrived earlier    : ' + q(w.lead_pre))
    log('  margin over best STANDING bid at winning time  : ' + q(w.stand_key))
    return d
