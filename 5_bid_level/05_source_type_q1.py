"""Bid timing by relay and collection channel (source_type), full Q1 2025 bid archive.
Reads the 90 raw daily files BIDARCHIVE/2025-MM-DD_top.csv.zip. Usage: python source_type_q1.py /path/to/bidarchive/top reference/builder_clusters.csv
Same row filter as 5_bid_level/01_relay_timing_q1_2025.py (drop rows with missing value, slot or slot_t_ms). Medians are exact to 1 ms (histogram)."""
import sys, zipfile, warnings
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths
from datetime import date, timedelta
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
TOP = (sys.argv[1] if len(sys.argv) > 1 else paths.BIDARCHIVE)
CLUS = (sys.argv[2] if len(sys.argv) > 2 else paths.REF + 'builder_clusters.csv')
clus = pd.read_csv(CLUS); late = set(clus[clus['median_timing'] > 1000]['builder_pubkey'])
LO, HI = -20000, 40000
A, L, n = {}, {}, 0
def add(dst, x):
    for k, g in x.groupby(['relay', 'source_type'])['slot_t_ms']:
        h = np.bincount(np.clip(np.rint(g.to_numpy()).astype(np.int64), LO, HI) - LO, minlength=HI - LO + 1)
        dst[k] = dst[k] + h if k in dst else h
for i in range(90):
    d = date(2025, 1, 1) + timedelta(days=i)
    with zipfile.ZipFile(f'{TOP}/{d}_top.csv.zip') as zf:
        with zf.open(zf.namelist()[0]) as f:
            x = pd.read_csv(f, usecols=['source_type', 'slot', 'slot_t_ms', 'value', 'relay', 'builder_pubkey'], low_memory=False)
    for c in ('value', 'slot', 'slot_t_ms'): x[c] = pd.to_numeric(x[c], errors='coerce')
    x = x.dropna(subset=['value', 'slot', 'slot_t_ms']); n += len(x)
    add(A, x); add(L, x[x['builder_pubkey'].isin(late)])
def med(h):
    c = np.cumsum(h); return (int(np.searchsorted(c, c[-1] / 2)) + LO, int(c[-1])) if c[-1] else (float('nan'), 0)
print('rows', n, '(paper: 86,805,657)')
print('relay,source_type,n,median_ms,late_n,late_median_ms')
for k in sorted(A):
    m, c = med(A[k]); lm, lc = med(L[k]) if k in L else (float('nan'), 0)
    print(f'{k[0]},{k[1]},{c},{m},{lc},{lm}')
for st in sorted({k[1] for k in A}):
    m, c = med(sum(v for k, v in A.items() if k[1] == st)); ls = [v for k, v in L.items() if k[1] == st]
    lm, lc = med(sum(ls)) if ls else (float('nan'), 0)
    print(f'ALL,{st},{c},{m},{lc},{lm}')
