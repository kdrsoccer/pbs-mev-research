#!/usr/bin/env bash
# Reproduce every table and figure that can be rebuilt from the files in data/.
# Usage:  bash run_all.sh          (about 15 minutes)
set -e
cd "$(dirname "$0")"; mkdir -p logs work figures
run() { echo ">>> $1"; ( cd "$(dirname "$1")" && python3 "$(basename "$1")" ) > "logs/$(basename "$1" .py).txt" 2> "logs/$(basename "$1" .py).err"; }
run 2_analysis/01_load_dedupe.py
run 2_analysis/02_regimes_temporal_concentration.py
run 2_analysis/03_auction.py
run 2_analysis/04_regression_eth_timeseries.py
run 2_analysis/05_garch_level.py
run 3_figures/fig_baseline.py
run 3_figures/fig_expansion.py
run 3_figures/fig_garch_temporal_margin.py
cp -n expected_output/relay_timing_q1_2025.csv work/ 2>/dev/null || true
run 5_bid_level/03_fig_relay_timing.py       # from expected_output/relay_timing_q1_2025.csv; 5_bid_level/01 and 05 need the bid archive
run 6_robustness/01_volatility_eth_relay_checks.py
run 7_all_bids/label_operators.py            # operators from public labels and shared blocks; validation of the behavioural rules
run 6_robustness/06_all_bids_analysis.py     # the auction from all recorded bids
run 6_robustness/07_relay_trace_check.py     # same-relay check of the higher rival bids (24 sample days)
run 6_robustness/08_market_concentration.py  # concentration on all MEV-Boost blocks; sample against market; re-weighted margins
python3 check_numbers.py | tee logs/check_numbers.txt | tail -1
python3 check_robustness_numbers.py | tee logs/check_robustness_numbers.txt | tail -1
python3 check_auction_numbers.py | tee logs/check_auction_numbers.txt | tail -1
echo "Done. Output in logs/, figures in figures/. To rebuild the PDF: bash build_paper.sh"
