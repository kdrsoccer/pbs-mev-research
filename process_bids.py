import pandas as pd
import numpy as np
import zipfile
from pathlib import Path
from datetime import date, timedelta
import warnings
warnings.filterwarnings('ignore')

BIDARCHIVE_DIR = Path('/Users/kylerothwell/pbs_research/bidarchive/top')
DELIVERED_PATH = Path('/Users/kylerothwell/Downloads/pbs_research/pbs_research/data/raw_extended/delivered_all_relays.parquet')
OUTPUT_PATH    = Path('/Users/kylerothwell/pbs_research/bid_auction_stats.parquet')

print("Loading delivered payloads...")
delivered = pd.read_parquet(DELIVERED_PATH, columns=['slot', 'value_wei', 'builder_pubkey', 'relay'])
valid_slots = set(delivered['slot'].values)
print(f"  {len(valid_slots):,} unique slots to match")

start = date(2024, 9, 1)
end   = date(2026, 5, 31)

all_stats = []
d = start
total_days = (end - start).days + 1
day_count = 0

while d <= end:
    ym    = d.strftime('%Y-%m')
    fname = f"{d.strftime('%Y-%m-%d')}_top.csv.zip"
    fpath = BIDARCHIVE_DIR / fname
    day_count += 1

    if not fpath.exists():
        d += timedelta(days=1)
        continue

    try:
        with zipfile.ZipFile(fpath) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, usecols=['slot', 'value', 'builder_pubkey', 'relay', 'slot_t_ms'])

        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['slot'] = pd.to_numeric(df['slot'], errors='coerce')
        df['slot_t_ms'] = pd.to_numeric(df['slot_t_ms'], errors='coerce')
        df = df.dropna(subset=['value', 'slot'])
        df['slot'] = df['slot'].astype(int)
        df['value'] = df['value'].astype(float)

        df = df[df['slot'].isin(valid_slots)]

        if len(df) == 0:
            d += timedelta(days=1)
            continue

        def second_largest(x):
            s = x.nlargest(2)
            return s.iloc[1] if len(s) >= 2 else np.nan

        stats = df.groupby('slot').agg(
            n_bids          = ('value', 'count'),
            n_builders      = ('builder_pubkey', 'nunique'),
            n_relays        = ('relay', 'nunique'),
            top_bid_wei     = ('value', 'max'),
            second_bid_wei  = ('value', second_largest),
            median_bid_wei  = ('value', 'median'),
            earliest_bid_ms = ('slot_t_ms', 'min'),
            latest_bid_ms   = ('slot_t_ms', 'max'),
        ).reset_index()

        stats['bid_margin'] = (stats['top_bid_wei'] - stats['second_bid_wei']) / stats['top_bid_wei']

        all_stats.append(stats)

        if day_count % 30 == 0:
            print(f"  Processed {day_count}/{total_days} days, {sum(len(s) for s in all_stats):,} slots so far...")

    except Exception as e:
        print(f"  Error on {fname}: {e}")

    d += timedelta(days=1)

print("\nCombining auction stats...")
auction_stats = pd.concat(all_stats, ignore_index=True)
print(f"  {len(auction_stats):,} slots with bid data")

print("Merging with delivered payloads...")
delivered_full = pd.read_parquet(DELIVERED_PATH)
merged = delivered_full.merge(auction_stats, on='slot', how='left')
print(f"  Merged dataset: {len(merged):,} rows")
print(f"  Slots with bid data: {merged['n_bids'].notna().sum():,}")
print(f"  Match rate: {merged['n_bids'].notna().mean()*100:.1f}%")

merged.to_parquet(OUTPUT_PATH, index=False)
print(f"\nSaved to {OUTPUT_PATH}")
print("\nSample:")
print(merged[['slot', 'n_bids', 'n_builders', 'top_bid_wei', 'second_bid_wei', 'bid_margin', 'earliest_bid_ms', 'latest_bid_ms']].dropna().head(5).to_string())
