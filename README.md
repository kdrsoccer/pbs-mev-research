# Empirical Analysis of Proposer-Builder Separation on Ethereum

Code and data for the paper *Empirical Analysis of Proposer-Builder Separation on Ethereum: Auction Dynamics, MEV Volatility Clustering, and Market Concentration* (`4_paper/pbs_paper.pdf`).

## What the paper uses

| Data | Source | In this repository |
|---|---|---|
| Delivered payloads from five MEV-Boost relays, Sep 2024 to May 2026 (1,271,079 slots) | relay data APIs, `1_collect/01_collect_delivered.py` | `data/bid_auction_stats.parquet` (391 MB, not in git; see below) |
| Every bid recorded for 1,219,842 of those slots (3.70 billion bids) | Relayscan bid archive, all-bids files, `7_all_bids/pbs_all_bids.ipynb` | reduced to `data/all_bids_by_slot.parquet`, `data/all_bids_keys.parquet`, `data/all_bids_intermediate/` |
| Delivered payloads on all relays (4.1 million MEV-Boost blocks) and builder labels | Data Always, [mevboost-data](https://github.com/dataalways/mevboost-data) | `data/market_delivered.parquet`, `data/builder_labels.csv` |
| Per-relay bid traces for 24 sample days | ethPandaOps Xatu, `7_all_bids/pbs_relay_traces.ipynb` | `data/relay_trace_*.parquet` |
| Hourly ETH/USD | Coinbase, `1_collect/04_fetch_eth_price.py` | `data/eth_price_hourly.parquet` |
| Top-bid files of the bid archive (Q1 2025 timing by relay and channel; top-bid record statistics) | Relayscan bid archive, `1_collect/02_download_bidarchive.py` | outputs only: `expected_output/relay_timing_q1_2025.csv`, `data/winner_cutoff_by_slot.parquet` |

`data/bid_auction_stats.parquet` exceeds GitHub's file-size limit. Rebuild it with `1_collect/01_collect_delivered.py` and `1_collect/03_process_bids.py`, or ask the author for the file (sha256 in `data/README.md`). Everything else needed to reproduce the paper's numbers is in `data/`.

## Reproduce

```bash
pip install -r requirements.txt
bash run_all.sh        # about 15 minutes; writes work/, logs/, figures/
bash build_paper.sh    # compiles 4_paper/pbs_paper.pdf
```

`run_all.sh` ends with three checks of the numbers printed in the paper against recomputed values. Expected last lines:

```
69 of 69 match the paper.
31 of 31 match the paper.
393/393 numbers found in the paper
```

## Layout

| Folder | Contents |
|---|---|
| `1_collect/` | Data collection: delivered payloads, bid-archive download, per-slot bid statistics, ETH price. |
| `2_analysis/` | Deduplication, volatility regimes, seasonality, ETH-price regressions, level GARCH. `student_t_diagnostic/` holds the Student-t fits on the differenced series discussed in Section 5.2, with their outputs. |
| `3_figures/` | Figures for Sections 5.1, 5.2, 5.4 and 5.6. |
| `5_bid_level/` | Bid timing by relay and by collection channel in the Q1 2025 top-bid files (Section 5.6.4). `01` and `05` need the top-bid files; their output is in `expected_output/`. |
| `6_robustness/` | `01` robustness of the volatility result and the ETH-volatility comparison; `04`, `05` what the top-bid files record (Section 3.2); `06` the auction from all bids (Sections 5.5, 5.6.3); `07` same-relay check (Section 5.5.2); `08` concentration on all MEV-Boost blocks and re-weighted margins (Sections 5.3, 5.5). |
| `7_all_bids/` | The pass over the all-bids files, operator construction from labels and shared blocks, and the per-relay trace reduction. See `7_all_bids/README.md`. |
| `data/` | Inputs. See `data/README.md`. |
| `expected_output/` | Logs and number tables from the last full run. |
| `4_paper/` | LaTeX source, figures and PDF. |

## What cannot be rebuilt from this repository alone

- The all-bids pass (`7_all_bids/pbs_all_bids.ipynb`) downloads about 800 GB from the bid archive. It was run on Google Colab; its reduced outputs are in `data/`.
- The per-relay trace reduction downloads 24 files of 1 to 3 GB each from ethPandaOps.
- The top-bid-file scripts need the archive's top-bid files (about 40 GB).
