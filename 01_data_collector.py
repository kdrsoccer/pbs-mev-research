"""
PBS MEV Research — Data Collector
Collects delivered payload data from MEV-Boost relays.
Compatible with Windows, Mac, and Linux.
"""

import requests
import pandas as pd
import time
import logging
import os
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("collection.log"),
        logging.StreamHandler()
    ]
)

BEACON_GENESIS = 1606824023

RELAYS = {
    "flashbots":   "https://boost-relay.flashbots.net",
    "ultrasound":  "https://relay.ultrasound.money",
    "bloxroute":   "https://bloxroute.regulated.blxrbdn.com",
    "aestus":      "https://mainnet.aestus.live",
    "agnostic":    "https://agnostic-relay.net",
}

def fetch_delivered_payloads(relay_name, base_url, limit=100, cursor=None):
    url = f"{base_url}/relay/v1/data/bidtraces/proposer_payload_delivered"
    params = {"limit": limit}
    if cursor:
        params["cursor"] = cursor
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.warning(f"{relay_name}: {e}")
        return []

def collect_relay(relay_name, base_url, max_records=200000, output_dir="data/raw_extended"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"delivered_{relay_name}.parquet"

    records = []
    cursor = None
    batch = 0

    logging.info(f"Starting collection: {relay_name}")

    while len(records) < max_records:
        batch += 1
        data = fetch_delivered_payloads(relay_name, base_url, limit=100, cursor=cursor)

        if not data:
            logging.info(f"{relay_name}: No more data at batch {batch}")
            break

        records.extend(data)
        logging.info(f"{relay_name}: {len(records):,} records collected")

        if len(data) < 100:
            break

        cursor = data[-1].get("slot")
        time.sleep(0.5)

    if records:
        df = pd.DataFrame(records)
        df["value_eth"] = pd.to_numeric(df["value"], errors="coerce") / 1e18
        df["slot"] = pd.to_numeric(df["slot"], errors="coerce")
        df["timestamp"] = BEACON_GENESIS + df["slot"] * 12
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df.to_parquet(output_path, index=False)
        logging.info(f"{relay_name}: Saved {len(df):,} records to {output_path}")

    return len(records)

def deduplicate(output_dir="data/raw_extended"):
    parquet_files = list(Path(output_dir).glob("delivered_*.parquet"))
    if not parquet_files:
        logging.error("No parquet files found to deduplicate")
        return

    dfs = []
    for f in parquet_files:
        if "all_relays" in f.name:
            continue
        df = pd.read_parquet(f)
        df["relay"] = f.stem.replace("delivered_", "")
        dfs.append(df)
        logging.info(f"Loaded {f.name}: {len(df):,} rows")

    combined = pd.concat(dfs, ignore_index=True)
    combined["slot"] = pd.to_numeric(combined["slot"], errors="coerce")
    combined["value_eth"] = pd.to_numeric(combined["value_eth"], errors="coerce")
    combined = combined.sort_values(["slot", "value_eth"], ascending=[True, False])
    deduped = combined.drop_duplicates(subset=["slot"], keep="first")
    deduped = deduped.sort_values("slot").reset_index(drop=True)

    out_path = Path(output_dir) / "delivered_all_relays.parquet"
    deduped.to_parquet(out_path, index=False)
    logging.info(f"Deduplicated: {len(deduped):,} unique slots -> {out_path}")

if __name__ == "__main__":
    for name, url in RELAYS.items():
        collect_relay(name, url, max_records=200000)
        time.sleep(2)
    deduplicate()
    logging.info("Collection complete")
