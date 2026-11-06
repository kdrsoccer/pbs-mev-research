# Empirical Analysis of Proposer-Builder Separation on Ethereum

Code and data pipeline for *Empirical Analysis of Proposer-Builder Separation on Ethereum: Auction Dynamics, MEV Volatility Clustering, and Market Concentration* (Rothwell, 2026).

**Paper:** `Paper/pbs_paper_v3.pdf` · **arXiv:** (link once posted)

## What's in the paper

1,271,079 delivered MEV-Boost payloads from six relays (Sept 2024 – May 2026), matched to 1,219,842 slots of bid-level data from the Relayscan bid archive. Main results:

- Block values are ~7× higher in high-volatility regimes (Spearman ρ = 0.49).
- Block-value log returns show extreme volatility persistence (GARCH(1,1) α+β = 0.98, Student-t ν ≈ 4), robust to GJR and winsorization. ETH/USD explains none of it (R² = 0.0001).
- Builder HHI fell from ~2,100 to 600–1,000; pooled Gini = 0.87 across 227 builders.
- Bid margins compress with block value and, more strongly, with relay participation. Builder participation per slot *declined* (17 → 9–11) while delivered-block concentration also fell — consolidation, not entry.
- A relay-side bid-timestamp artifact (Ultrasound vs. Titan/Eden offsets of several seconds) that invalidates naive cross-relay timing analysis.

## Repository layout

```
.
├── 01_data_collector.py          # Pull delivered payloads from six relay REST APIs (~2–3 h)
├── 02_analysis.py                # Volatility regimes, temporal patterns, HHI/Gini
├── garch_extended_robustness.py  # GARCH(1,1), GJR, winsorized, subperiod fits
├── regenerate_figures.py         # Figures 1–5
├── process_bids_v2.py            # Build bid_auction_stats.parquet from the Relayscan bid archive
├── auction_analysis.py           # Auction microstructure (Figures 6–10)
├── expansion_analysis.py         # ETH decomposition, margin regression, relay/builder analysis (Figures 11–17)
├── requirements.txt
├── Paper/
│   ├── pbs_paper_v3.tex
│   └── figures/
└── data/                         # not tracked — see below
```

## Data

Nothing large is committed. Three inputs are needed to reproduce everything:

| File | Size | How to get it |
|---|---|---|
| `data/raw_extended/delivered_all_relays.parquet` | ~340 MB | `python 01_data_collector.py` (queries Flashbots, Ultrasound, BloXroute, Aestus, Agnostic, Eden) |
| `bidarchive/top/YYYY-MM-DD_top.csv.zip` | ~1 GB/month | Daily "top bids" dumps from [bidarchive.relayscan.io](https://bidarchive.relayscan.io) |
| `eth_price_hourly.parquet` | <1 MB | Coinbase ETH-USD hourly candles |

Then build the matched bid dataset:

```bash
python process_bids_v2.py      # -> bid_auction_stats.parquet (~390 MB)
```

**Known data issues handled in the code**
- `relay.ethgas.com` injects ~11,000 ETH placeholder bids from Dec 2025; `process_bids_v2.py` filters them before aggregating `top_bid_wei`.
- The stored `datetime`/`hour`/`dayofweek` columns in early parquet builds carried a ~653-day offset from using the Merge timestamp as genesis. All analysis derives time from `slot` directly: `t = 1606824023 + 12·slot`.
- Relay bid timestamps are not comparable across relays (Section 5.6.3 of the paper).

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python 02_analysis.py
python garch_extended_robustness.py
python regenerate_figures.py
python auction_analysis.py
python expansion_analysis.py

cd Paper && pdflatex pbs_paper_v3.tex && pdflatex pbs_paper_v3.tex
```

Scripts currently hard-code `/Users/kylerothwell/pbs_research/` paths at the top of each file; edit those to your checkout before running.

## Citation

```bibtex
@misc{rothwell2026pbs,
  title  = {Empirical Analysis of Proposer-Builder Separation on Ethereum:
            Auction Dynamics, MEV Volatility Clustering, and Market Concentration},
  author = {Rothwell, Kyle D.},
  year   = {2026},
  note   = {Working paper, Stevens Institute of Technology}
}
```

## License

MIT for code. Data belongs to the respective relay operators and Relayscan.
