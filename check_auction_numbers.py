"""Checks the operator, market, auction and relay-trace numbers printed in the paper against work/all_bids_numbers.csv, work/market_numbers.csv and
work/relay_trace_numbers.csv (written by 6_robustness/06, 07 and 08).  Usage: python3 check_auction_numbers.py [path/to/pbs_paper.tex]"""
import sys, os, re, pandas as pd
import paths
tex = open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(paths.ROOT, '4_paper', 'pbs_paper.tex')).read().replace('{,}', ',').replace('$-$', '-')
N = pd.read_csv(paths.WORK + 'all_bids_numbers.csv', index_col=0)['value']
C = []  # (key, decimals, scale)
def add(k, nd=0, pat=None): C.append((k, nd, pat))
for k in [
          'archive days read', 'slots with the winning block found',
          'builder keys bidding', 'operators bidding (D)', 'operators bidding (B)', 'operators bidding (L)', 'operators bidding (M)', 'operators with more than one key (L)', 'keys held by multi-key operators (L)', 'weeks in the HHI-margin test',
          'summary bids median', 'summary bids p10', 'summary bids p90', 'summary bids mean', 'summary keys median', 'summary operators median', 'summary winning bid ms median', 'summary winning bid ms p10', 'summary winning bid ms p90', 'summary winning bid ms mean', 'summary next bid gap ms median', 'summary next bid gap ms p90']: add(k)
for p in ['Sep-Dec 2024', 'Jan-Apr 2025', 'May-Dec 2025', 'Jan-May 2026']: add(f'agnostic operators bidding per slot, {p}', 1)
for k in ['winning block found among the recorded bids (%)',
          'a rival operator had a higher standing bid (B, % of slots)', 'a rival operator had a higher standing bid (D, % of slots)', 'a rival operator had a higher standing bid (L, % of slots)', 'a rival operator had a higher standing bid (M, % of slots)', 'a rival operator had a higher standing bid (KEY, % of slots)',
          '... higher by more than 1% (L, % of slots)', 'winner more than 1% ahead of every rival operator (% of slots)', 'winner more than 5% ahead (% of slots)', 'no rival operator within 1% (% of slots)', 'exactly one rival within 1% (% of slots)', 'two or more rivals within 1% (% of slots)',
          'bid share of the five largest winners (%)', 'win share of the five largest winners (%)', 'summary keys mean', 'summary operators mean']: add(k, 1)
for k in ['median standing margin (B, %)', 'median standing margin (D, %)', 'median standing margin (L, %)', 'median standing margin (M, %)', 'median margin over any lower rival bid (B, %)', 'median margin over any lower rival bid (D, %)', 'median margin over any lower rival bid (L, %)', 'median margin over any lower rival bid (M, %)',
          'summary stand % mean', 'summary stand % p10', 'summary stand % p90', 'summary any % mean', 'summary any % p10', 'summary any % p90', 'rho value ~ operators bidding', 'rho value ~ operators bidding CI low', 'rho value ~ operators bidding CI high',
          'rho value ~ keys bidding', 'rho value ~ standing margin', 'rho value ~ standing margin CI low', 'rho value ~ standing margin CI high', 'rho value ~ bids', 'rho operators bidding ~ standing margin',
          ]: add(k, 3)
for k in ['standing margin p25 (%)', 'standing margin p75 (%)', 'no rival operator among the candidates (% of slots)']: add(k, 2)
for q in ['2024Q3', '2024Q4', '2025Q1', '2025Q2', '2025Q3', '2025Q4', '2026Q1', '2026Q2']:
    add(f'quarter {q} slots'); add(f'quarter {q} bids'); add(f'quarter {q} keys'); add(f'quarter {q} ops'); add(f'quarter {q} stand_L', 3); add(f'quarter {q} rival_higher', 1); add(f'operators bidding per day, {q}') if q in ('2024Q3', '2026Q2') else None
for t in ['Low', 'Mid', 'High']:
    add(f'tercile {t} ops'); add(f'tercile {t} keys'); add(f'tercile {t} bids'); add(f'tercile {t} stand_L', 3); add(f'tercile {t} rival_higher', 1); add(f'tercile {t} riv1', 2); add(f'tercile {t} wt')
for o in ['Titan', 'buildernet', 'beaverbuild.org', 'Quasar', 'rsync', 'BTCS', 'Eureka Builder', 'bobthebuilder']:
    add(f'operator {o} keys'); add(f'operator {o} bid_share', 1); add(f'operator {o} wins_per_1000', 2); add(f'operator {o} t_median')
    N[f'operator {o} after_start pct'] = 100 * N[f'operator {o} after_start']; N[f'operator {o} above_winner pct'] = 100 * N[f'operator {o} above_winner']; add(f'operator {o} after_start pct', 1); add(f'operator {o} above_winner pct', 1)
    add(f'operator {o} win_share', 1)
for X in ['B', 'D']:
    add(f'validation {X} operators'); add(f'validation {X} operators with two or more labels')
    for c in ['purity_bids', 'completeness_bids', 'completeness_wins']: N[f'validation {X} {c} pct'] = 100 * N[f'validation {X} {c}']; add(f'validation {X} {c} pct', 1)
R = pd.read_csv(paths.WORK + 'relay_trace_numbers.csv', index_col=0)['value']
for k_ in R.index: N['RT ' + k_] = R[k_]
for k_, nd in [('sample days', 0), ('slots in the same-relay check', 0), ('archive: rival live bid higher (% of these slots)', 1), ('same relay: rival live bid higher on a delivering relay (% of slots)', 1), ('same relay: higher and at least 100 ms old (% of slots)', 1),
               ('age of the higher rival bid: median (ms)', 0), ('age of the higher rival bid: p25 (ms)', 0), ('age of the higher rival bid: p75 (ms)', 0), ('share of higher rival bids less than 50 ms old (%)', 0), ('share of higher rival bids less than 100 ms old (%)', 0),
               ('archive-negative slots: higher rival bid found on a delivering relay (%)', 1), ('archive-negative slots: found on another relay only (%)', 1), ('archive-negative slots: not found on any covered relay (%)', 1),
               ('median margin on the delivering relay (%)', 2), ('median archive standing margin, same slots (%)', 2), ('rank correlation of the two margins', 2)]: add('RT ' + k_, nd)
for q in ['2024Q4', '2025Q1', '2025Q2', '2025Q3', '2025Q4', '2026Q1', '2026Q2']:
    add(f'RT quarter {q} slots'); add(f'RT quarter {q} archive', 1); add(f'RT quarter {q} same', 1); add(f'RT quarter {q} same100', 1)
M = pd.read_csv(paths.WORK + 'market_numbers.csv', index_col=0)['value']
for k_ in M.index: N['MK ' + k_] = M[k_]
for k_, nd in [('MEV-Boost slots, all relays', 0), ('share of all slots in the period (%)', 1), ('sample share of MEV-Boost slots (%)', 1), ('sample slots found in the market set (%)', 2), ('builder keys with a delivered block, market', 0), ('operators with a delivered block, market (L)', 0),
               ('market HHI by key', 0), ('market HHI by operator (shared blocks only)', 0), ('market HHI by operator (shared blocks and value ties)', 0), ('market HHI by operator (labels and shared blocks)', 0), ('market HHI by operator (BuilderNet partners merged)', 0),
               ('market top-5 key share (%)', 1), ('market top-1 share (L, %)', 1), ('market top-3 share (L, %)', 1), ('market top-5 share (L, %)', 1), ('market top-2 share (M, %)', 1), ('market top-3 share (M, %)', 1), ('sample HHI by operator (L)', 0), ('weeks', 0),
               ('weekly market HHI by operator: min', 0), ('weekly market HHI by operator: median', 0), ('weekly market HHI by operator: max', 0), ('weekly market HHI by key: min', 0), ('weekly market HHI by key: max', 0),
               ('median standing margin, re-weighted (%)', 2), ('rival live bid higher, re-weighted (%)', 1), ('no rival within 1%, re-weighted (%)', 1), ('lagged market operator HHI ~ weekly median margin: rho', 3), ('lagged market operator HHI ~ weekly median margin: p', 2)]: add('MK ' + k_, nd)
for b in ['Titan', 'BuilderNet', 'beaverbuild', 'Quasar', 'rsync']:
    add(f'MK market share {b} (%)', 1); add(f'MK winner {b} median', 2); add(f'MK winner {b} higher', 1); add(f'MK winner {b} none', 1); add(f'MK winner {b} sample_share', 1)
    for q in ['2024Q3', '2024Q4', '2025Q1', '2025Q2', '2025Q3', '2025Q4', '2026Q1', '2026Q2']: add(f'MK market share {b} {q}', 1)
for b in ['Titan', 'Quasar']: add(f'MK sample share {b} (%)', 1)
for q in ['2024Q3', '2024Q4', '2025Q1', '2025Q2', '2025Q3', '2025Q4', '2026Q1', '2026Q2']:
    add(f'MK quarter {q} slots'); add(f'MK quarter {q} key'); add(f'MK quarter {q} L'); add(f'MK quarter {q} M'); add(f'MK quarter {q} margin reweighted', 3)
for k_, nd in [('old higher bids from one of the five largest operators (%)', 0), ('old higher bids from bobthebuilder (%)', 0), ('old higher bids from bloXroute (%)', 0), ('old higher bids: median excess over the winning bid (%)', 0), ('old higher bids from the five largest: median excess (%)', 1),
               ('slots with an old higher bid from one of the five largest operators (%)', 2), ('slots with an old higher bid, excluding 15 December 2024 (%)', 1), ('15 December 2024, BloXroute: old higher bids (% of slot-relay pairs)', 0)]: add('RT ' + k_, nd)
bad = 0
for k, nd, _ in C:
    v = N[k] + 1e-9; s = f'{v:,.{nd}f}'; alts = {s, s.replace(',', '')}
    if abs(v) < 1 and nd >= 2: alts |= {s.replace('0.', '.')}
    ok = any(re.search(r'(?<![\d.,])' + re.escape(a) + r'(?![\d])', tex) for a in alts)
    if not ok: bad += 1; print('MISSING', k, s)
print(f'{len(C) - bad}/{len(C)} numbers found in the paper')
