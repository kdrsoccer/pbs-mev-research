"""Single source of truth for where things live. Every script imports this.
Override the bid-archive location with the BIDARCHIVE environment variable."""
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data') + os.sep          # inputs: bid_auction_stats.parquet, eth_price_hourly.parquet
WORK = os.path.join(ROOT, 'work') + os.sep          # intermediates written by 2_analysis/
FIG  = os.path.join(ROOT, 'figures') + os.sep       # figures written by 3_figures/
REF  = os.path.join(ROOT, 'reference') + os.sep     # reference CSVs used for comparison printouts and by 5_bid_level/
SRC  = DATA + 'bid_auction_stats.parquet'           # merged delivered payloads + per-slot bid statistics
ETH  = DATA + 'eth_price_hourly.parquet'
LEAD = DATA + 'lead_margin_by_slot.parquet'     # precomputed output of 6_robustness/03_lead_margin.py (needs the full bid archive to rebuild)
CUT  = DATA + 'winner_cutoff_by_slot.parquet'    # precomputed output of 6_robustness/04_winner_cutoff.py (needs the full bid archive to rebuild)
AB   = DATA + 'all_bids_by_slot.parquet'         # per-slot statistics from the archive's all-bids files (notebook in 7_all_bids/)
ABK  = DATA + 'all_bids_keys.parquet'            # per builder key per day: bids, wins, timing
OPS  = DATA + 'operator_map.csv'                 # builder key -> operator, under each linking rule
DELIVERED = DATA + os.path.join('raw_extended', 'delivered_all_relays.parquet')
BIDARCHIVE = os.environ.get('BIDARCHIVE', DATA + os.path.join('bidarchive', 'top'))
for _d in (WORK, FIG):
    os.makedirs(_d, exist_ok=True)
