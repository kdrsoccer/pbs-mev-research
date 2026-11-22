# data/

| File | What |
|---|---|
| `bid_auction_stats.parquet` (391 MB, **not in git**) | Delivered payloads from five relays merged with per-slot statistics of the bid archive's top-bid files. One row per (slot, relay record); deduplicate on `slot`. sha256 `de237a3b5cb839c3dc8f9547d0f639ddc24d632b02880d30cd5134ba02325826` |
| `eth_price_hourly.parquet` | Hourly ETH/USD from Coinbase. |
| `delivered_slim.parquet` | Slot, value, block-hash prefix and builder key of the 1,219,842 slots of the auction sample: the input of the all-bids notebook. |
| `all_bids_by_slot.parquet` | Per slot, from every bid in the archive's all-bids files: `has_win`, `wt` (winning bid arrival, ms from slot start), `n_all` (bids), `nb_all` (keys), `nop_all` (operators, within-day shared-block links), `wnext_gap` (ms to the winning key's next bid), and for each operator definition X: `stand_X` (margin over the best live bid of another operator when the winning bid arrived), `any_X` (margin over the best bid at or below the winning value from another operator), `riv1_X` (other operators with a live bid within 1%). X = L (labels and shared blocks, the paper's baseline), M (L with beaverbuild, BuilderNet and Flashbots as one), B (shared blocks only), D (shared blocks and value ties), F (looser variant, not used), KEY. |
| `all_bids_keys.parquet` | Per builder key per day: bids, slots, median bid time, share after slot start, share above the winning value, wins. |
| `all_bids_intermediate/` | Outputs of the all-bids notebook and of the two extraction cells: the inputs of `7_all_bids/relink.py` and `7_all_bids/label_operators.py`. |
| `operator_map.csv` | Builder key to operator code under each definition, and the key's public label. |
| `builder_labels.csv` | For each of the 803 builder keys: `map_name` (Data Always `mapping_pubkey_builder_name.csv`), `xd_label` (modal `builder_label` of the key's delivered blocks), `xd_purity`, `blocks`. |
| `market_delivered.parquet` | One row per slot with an MEV-Boost delivered payload on any relay, 10 Sep 2024 to 2 Jun 2026: `slot`, `builder_pubkey`, `builder_label`, `n_relays`, `n_keys`. From Data Always `mevboost-data`. |
| `relay_trace_keys.parquet` | Per slot, relay and builder key on 24 sample days, from ethPandaOps Xatu `mev_relay_bid_trace`: the key's latest bid on that relay when the winning block reached that relay (`live_w`, `tlive_w`) and when it first reached any relay (`live_g`, `tlive_g`), `vmax`, `vmax_pre`, `isw`. Values are ratios to the winning value; times are ms from slot start. Rows kept: keys whose best bid is at least 90% of the winning value, and the winner. |
| `relay_trace_relays.parquet` | Per slot and relay: bids, keys, top bid, `tw` (when the winning block reached that relay), `twg` (earliest across relays). |
| `relay_trace_delivered.parquet` | Which relays delivered each sample slot (Data Always `mevboost-data`). |
| `winner_cutoff_by_slot.parquet`, `lead_margin_by_slot.parquet` | Per-slot statistics of the top-bid files with each slot cut at the delivered value (`6_robustness/04_winner_cutoff.py`); used for the statements about the top-bid files in Section 3.2. |
