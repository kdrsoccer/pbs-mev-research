"""Margins with each slot cut at the delivered (winning) bid.
The top-bid file keeps rising after the proposer has signed, so the slot maximum is usually not the winning bid.
Per delivered slot (dv = delivered value, dpk = delivered builder):
  incr_cut = (dv - highest recorded row below dv) / dv
  lead_cut = (dv - highest recorded row below dv from a builder other than dpk) / dv
  n_cut, nb_cut, nr_cut = rows / builders / relays at or below dv;  n_higher = rows above dv
  has_win = the delivered block hash is a row of the file;  wt = its slot_t_ms;  t_above = first slot_t_ms of a row above dv
Usage: python3 04_winner_cutoff.py <shard> <nshards>   (writes days/<day>.parquet, skips days already done)"""
import os, sys, glob, zipfile, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, '..', '..'))
sh, ns = int(sys.argv[1]), int(sys.argv[2])
dl = pd.read_parquet(H + '/delivered_slim.parquet', dtype_backend='pyarrow')   # slot, dv, dh, dpk: built once from bid_auction_stats.parquet (deduplicated on slot)
dl['dv'] = dl.dv.astype('float64'); dl['slot'] = dl.slot.astype('int64')
files = sorted(glob.glob(R + '/bidarchive/top/*_top.csv.zip'))[sh::ns]
for fp in files:
    day = os.path.basename(fp)[:10]; out = f'{H}/days/{day}.parquet'
    if os.path.exists(out): continue
    try:
        with zipfile.ZipFile(fp) as z:
            x = pd.read_csv(z.open(z.namelist()[0]), usecols=['slot', 'slot_t_ms', 'value', 'block_hash', 'builder_pubkey', 'relay'], engine='pyarrow', dtype_backend='pyarrow')   # pyarrow reader: about a third of the memory of the default
    except Exception as e:
        print('skipped', day, e, flush=True); continue
    x['value'] = pd.to_numeric(x.value, errors='coerce'); x['slot'] = pd.to_numeric(x.slot, errors='coerce'); x['slot_t_ms'] = pd.to_numeric(x.slot_t_ms, errors='coerce')
    x = x.dropna(subset=['value', 'slot']); x = x[x.value < 1e21]; x['slot'] = x.slot.astype('int64'); x['value'] = x.value.astype('float64'); x['slot_t_ms'] = x.slot_t_ms.astype('float64')
    d = dl[dl.slot.isin(x.slot.unique())].set_index('slot')
    x = x[x.slot.isin(d.index)]
    if not len(x): continue
    x['dv'] = x.slot.map(d.dv).astype('float64')
    x['is_win'] = (x.block_hash == x.slot.map(d.dh).astype('string[pyarrow]')).fillna(False).astype(bool); x = x.drop(columns='block_hash')
    x['is_dpk'] = (x.builder_pubkey == x.slot.map(d.dpk).astype('string[pyarrow]')).fillna(False).astype(bool)
    x['b'] = pd.factorize(x.builder_pubkey)[0]; x['r'] = pd.factorize(x.relay)[0]; x = x.drop(columns=['builder_pubkey', 'relay'])   # b = -1 when the builder is not identified
    res = x.groupby('slot').agg(dv=('dv', 'first'), top=('value', 'max'), n=('value', 'size')).reset_index().set_index('slot')
    lo = x[x.value < x.dv]; le = x[x.value <= x.dv]; hi = x[x.value > x.dv]
    res['prev'] = lo.groupby('slot').value.max()
    oth = lo[(lo.b >= 0) & ~lo.is_dpk]
    res['prev_other'] = oth.groupby('slot').value.max()
    res['n_cut'] = le.groupby('slot').size(); res['nb_cut'] = le[le.b >= 0].groupby('slot').b.nunique(); res['nr_cut'] = le.groupby('slot').r.nunique()
    res['n_higher'] = hi.groupby('slot').size(); res['t_above'] = hi.groupby('slot').slot_t_ms.min()
    w = x[x.is_win]; res['wt'] = w.groupby('slot').slot_t_ms.min(); res['has_win'] = res.index.isin(w.slot).astype('int8')
    res[['n_cut', 'nb_cut', 'nr_cut', 'n_higher']] = res[['n_cut', 'nb_cut', 'nr_cut', 'n_higher']].fillna(0).astype('int32')
    res['incr_cut'] = (res.dv - res.prev) / res.dv; res['lead_cut'] = (res.dv - res.prev_other) / res.dv
    res.reset_index()[['slot', 'dv', 'top', 'n', 'n_cut', 'nb_cut', 'nr_cut', 'n_higher', 'has_win', 'wt', 't_above', 'incr_cut', 'lead_cut']].to_parquet(out + '.tmp', index=False)
    os.replace(out + '.tmp', out); print('done', day, len(res), flush=True)
