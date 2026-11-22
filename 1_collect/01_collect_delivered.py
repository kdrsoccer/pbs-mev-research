"""
PBS Research Pipeline — Module 1: Data Collection
Kyle Rothwell | Stevens Institute of Technology

Pulls builder bid traces and delivered payload data from multiple
MEV-Boost relays and saves to local parquet files for analysis.

Run this first. Estimated collection time: 30–60 minutes for 100k slots.

Usage:
    python 01_data_collector.py --slots 50000 --output data/raw
"""

import requests
import pandas as pd
import numpy as np
import time
import os
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("collection.log")
    ]
)
log = logging.getLogger(__name__)

# ── Relay endpoints ────────────────────────────────────────────────────────────
RELAYS = {
    "flashbots":  "https://boost-relay.flashbots.net",
    "ultrasound": "https://relay.ultrasound.money",
    "agnostic":   "https://agnostic-relay.net",
    "aestus":     "https://mainnet.aestus.live",
    "bloxroute":  "https://bloxroute.regulated.blxrbdn.com",
    "eden":       "https://relay.edennetwork.io",
}

DELIVERED_PATH  = "/relay/v1/data/bidtraces/proposer_payload_delivered"
BUILDER_PATH    = "/relay/v1/data/bidtraces/builder_blocks_received"
MAX_LIMIT       = 100   # relay max per request
RATE_LIMIT_SLEEP = 0.25 # seconds between requests per relay
REQUEST_TIMEOUT  = 15


def fetch_page(
    base_url: str,
    endpoint: str,
    cursor_slot: Optional[int] = None,
    limit: int = MAX_LIMIT,
    relay_name: str = ""
) -> list[dict]:
    """Fetch one page of relay data. Returns list of records or empty list."""
    params = {"limit": limit}
    if cursor_slot is not None:
        params["cursor"] = cursor_slot

    url = base_url + endpoint
    try:
        r = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "pbs-research-bot/1.0"}
        )
        if r.status_code == 200:
            return r.json() or []
        else:
            log.warning(f"{relay_name} {r.status_code} at cursor={cursor_slot}")
            return []
    except requests.exceptions.RequestException as e:
        log.warning(f"{relay_name} request error: {e}")
        return []


def collect_delivered_payloads(
    relay_name: str,
    base_url: str,
    n_slots: int = 50_000
) -> pd.DataFrame:
    """
    Collect proposer_payload_delivered data (winning bids only).
    Returns one row per slot.
    """
    records = []
    cursor  = None
    fetched = 0

    log.info(f"[{relay_name}] Starting delivered payload collection (target={n_slots})")

    while fetched < n_slots:
        page = fetch_page(base_url, DELIVERED_PATH, cursor, relay_name=relay_name)
        if not page:
            break

        records.extend(page)
        fetched += len(page)

        # Cursor for next page = slot of oldest record in this page
        cursor = int(page[-1]["slot"])

        if fetched % 1000 == 0:
            log.info(f"[{relay_name}] Collected {fetched} delivered payloads, cursor slot={cursor}")

        time.sleep(RATE_LIMIT_SLEEP)

        if len(page) < MAX_LIMIT:
            break   # reached end of available data

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # ── Type conversions ───────────────────────────────────────────────────────
    df["slot"]          = df["slot"].astype(int)
    df["block_number"]  = df["block_number"].astype(int)
    df["gas_used"]      = df["gas_used"].astype(int)
    df["gas_limit"]     = df["gas_limit"].astype(int)
    df["num_tx"]        = df["num_tx"].astype(int)
    # value is in wei (string) — convert to ETH float
    df["value_wei"]     = df["value"].astype(float)
    df["value_eth"]     = df["value_wei"] / 1e18
    df["relay"]         = relay_name

    # ── Timestamp from slot number ─────────────────────────────────────────────
    # Ethereum Merge slot 0 = 2022-09-15 06:42:59 UTC, 12s per slot
    MERGE_TIMESTAMP = 1663224179
    df["timestamp"] = MERGE_TIMESTAMP + df["slot"] * 12
    df["datetime"]  = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["hour"]      = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["date"]      = df["datetime"].dt.date

    df = df.drop(columns=["value"], errors="ignore")
    df = df.sort_values("slot", ascending=False).reset_index(drop=True)

    log.info(f"[{relay_name}] Collected {len(df)} delivered payloads")
    return df


def collect_builder_bids(
    relay_name: str,
    base_url: str,
    n_slots: int = 20_000
) -> pd.DataFrame:
    """
    Collect builder_blocks_received — ALL bids per slot, not just winners.
    This is the key dataset for bidding strategy analysis (Q1, Q2, Q3).
    """
    records = []
    cursor  = None
    fetched = 0

    log.info(f"[{relay_name}] Starting builder bid collection (target slots ~{n_slots})")

    while fetched < n_slots * 5:   # ~5 bids per slot on average
        page = fetch_page(base_url, BUILDER_PATH, cursor, relay_name=relay_name)
        if not page:
            break

        records.extend(page)
        fetched += len(page)
        cursor   = int(page[-1]["slot"])

        if fetched % 5000 == 0:
            log.info(f"[{relay_name}] Collected {fetched} builder bids, cursor slot={cursor}")

        time.sleep(RATE_LIMIT_SLEEP)

        if len(page) < MAX_LIMIT:
            break

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["slot"]         = df["slot"].astype(int)
    df["block_number"] = df["block_number"].astype(int)
    df["value_wei"]    = df["value"].astype(float)
    df["value_eth"]    = df["value_wei"] / 1e18
    df["relay"]        = relay_name

    MERGE_TIMESTAMP = 1663224179
    df["timestamp"]  = MERGE_TIMESTAMP + df["slot"] * 12
    df["datetime"]   = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    # ── Timing features (critical for Q1 — bid timing relative to slot deadline)
    # timestamp_ms is when the relay received the bid
    if "timestamp_ms" in df.columns:
        df["timestamp_ms"]    = df["timestamp_ms"].astype(float)
        df["bid_recv_time_ms"] = df["timestamp_ms"]
        # Slot deadline is at slot_start + 4000ms (4 seconds into slot)
        df["slot_start_ms"]    = (df["timestamp"] * 1000).astype(float)
        df["ms_into_slot"]     = df["bid_recv_time_ms"] - df["slot_start_ms"]
        # Negative = bid arrived before slot started (pre-slot bidding)
        df["ms_before_deadline"] = 4000 - df["ms_into_slot"]

    df = df.drop(columns=["value"], errors="ignore")
    df = df.sort_values(["slot", "timestamp_ms" if "timestamp_ms" in df.columns else "slot"])
    df = df.reset_index(drop=True)

    log.info(f"[{relay_name}] Collected {len(df)} builder bids across ~{df['slot'].nunique()} slots")
    return df


def run_collection(output_dir: str, n_delivered: int, n_bids: int):
    """Main collection loop across all relays."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_delivered = []
    all_bids      = []

    for relay_name, base_url in RELAYS.items():
        log.info(f"\n{'='*60}")
        log.info(f"Collecting from relay: {relay_name}")
        log.info(f"{'='*60}")

        # Delivered payloads
        df_del = collect_delivered_payloads(relay_name, base_url, n_delivered)
        if not df_del.empty:
            path = out / f"delivered_{relay_name}.parquet"
            df_del.to_parquet(path, index=False)
            log.info(f"Saved {len(df_del)} delivered payloads → {path}")
            all_delivered.append(df_del)

        # Builder bids
        df_bids = collect_builder_bids(relay_name, base_url, n_bids)
        if not df_bids.empty:
            path = out / f"bids_{relay_name}.parquet"
            df_bids.to_parquet(path, index=False)
            log.info(f"Saved {len(df_bids)} builder bids → {path}")
            all_bids.append(df_bids)

        time.sleep(1.0)  # be polite between relays

    # ── Merge across relays ────────────────────────────────────────────────────
    if all_delivered:
        combined_del = pd.concat(all_delivered, ignore_index=True)
        # Deduplicate: keep one record per slot (highest value across relays)
        combined_del = (
            combined_del
            .sort_values(["slot", "value_eth"], ascending=[True, False])
            .drop_duplicates(subset="slot", keep="first")
            .sort_values("slot", ascending=False)
            .reset_index(drop=True)
        )
        path = out / "delivered_all_relays.parquet"
        combined_del.to_parquet(path, index=False)
        log.info(f"\nCombined delivered payloads: {len(combined_del)} unique slots → {path}")

    if all_bids:
        combined_bids = pd.concat(all_bids, ignore_index=True)
        combined_bids = combined_bids.drop_duplicates(
            subset=["slot", "builder_pubkey", "value_wei"]
        ).reset_index(drop=True)
        path = out / "bids_all_relays.parquet"
        combined_bids.to_parquet(path, index=False)
        log.info(f"Combined builder bids: {len(combined_bids)} unique bids → {path}")

    log.info("\nCollection complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PBS Research Data Collector")
    parser.add_argument("--delivered", type=int, default=50_000,
                        help="Number of delivered payload slots to collect per relay")
    parser.add_argument("--bids", type=int, default=20_000,
                        help="Approximate number of bid slots to collect per relay")
    parser.add_argument("--output", type=str, default="data/raw",
                        help="Output directory for parquet files")
    args = parser.parse_args()

    run_collection(args.output, args.delivered, args.bids)
