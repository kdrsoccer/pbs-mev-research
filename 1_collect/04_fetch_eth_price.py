"""Hourly ETH/USD closes from the Coinbase Exchange public API, 2024-09-01 .. 2026-05-31 -> data/eth_price_hourly.parquet.
This is the code that produced the file in June 2026 (run inline then; only the output path is changed). It returned 15,302 candles.
data/eth_price_hourly.parquet is this script's output; rerunning will overwrite it."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import paths
import requests, time
import pandas as pd
from datetime import datetime, timedelta
BASE = "https://api.exchange.coinbase.com/products/ETH-USD/candles"
all_candles = []
start, end = datetime(2024, 9, 1), datetime(2026, 5, 31, 23, 0)
chunk_start = start
while chunk_start < end:                                  # 300 candles max per call at 1h granularity
    chunk_end = min(chunk_start + timedelta(hours=299), end)
    params = {"granularity": 3600, "start": chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ"), "end": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ")}
    r = requests.get(BASE, params=params)
    if r.status_code == 429:
        print("  Rate limited, waiting 5s..."); time.sleep(5); continue
    if r.status_code != 200:
        print(f"  Error {r.status_code}: {r.text[:100]}"); chunk_start = chunk_end + timedelta(hours=1); continue
    candles = r.json()
    if candles:
        all_candles.extend(candles); print(f"  Through {chunk_end:%Y-%m-%d}, total: {len(all_candles):,} candles")
    chunk_start = chunk_end + timedelta(hours=1); time.sleep(0.3)
df = pd.DataFrame(all_candles, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])   # Coinbase column order
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
df['eth_close'] = df['close'].astype(float); df['eth_high'] = df['high'].astype(float); df['eth_low'] = df['low'].astype(float)
df = df[['datetime', 'eth_close', 'eth_high', 'eth_low']].drop_duplicates('datetime').sort_values('datetime').reset_index(drop=True)
df.to_parquet(paths.ETH, index=False)
print(f"\nSaved {len(df):,} hourly candles to {paths.ETH}   (original run: 15,302)")
print(f"Range: {df['datetime'].min()} to {df['datetime'].max()}")
