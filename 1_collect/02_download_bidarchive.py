"""Download the Relayscan bid archive "top" files for 2024-09-01 .. 2026-05-31 into data/bidarchive/top/.
About 608 files of 60-70 MB each (~40 GB). Resumable: existing files are skipped.
URL layout checked 2026-11-15:
  https://bidarchive.relayscan.io/ethereum/mainnet/YYYY-MM/YYYY-MM-DD_top.csv.zip
NOTE: the original download was done by hand; this script documents it and has not been executed end to end."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import paths
import time, requests
from datetime import date, timedelta
from pathlib import Path
BASE = 'https://bidarchive.relayscan.io/ethereum/mainnet'
out = Path(paths.BIDARCHIVE); out.mkdir(parents=True, exist_ok=True)
d, end, missing = date(2024, 9, 1), date(2026, 5, 31), []
while d <= end:
    name = f'{d:%Y-%m-%d}_top.csv.zip'; dest = out / name
    if not dest.exists():
        url = f'{BASE}/{d:%Y-%m}/{name}'
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                if r.status_code == 404: missing.append(name)
                else:
                    r.raise_for_status(); tmp = dest.with_suffix('.part')
                    with open(tmp, 'wb') as f:
                        for chunk in r.iter_content(1 << 20): f.write(chunk)
                    tmp.rename(dest); print('got', name, f'{dest.stat().st_size / 1e6:.0f} MB', flush=True)
        except Exception as e:
            print('FAILED', name, e, flush=True); missing.append(name)
        time.sleep(0.2)
    d += timedelta(days=1)
print(f'done. {len(list(out.glob("*_top.csv.zip")))} files present, {len(missing)} missing or failed:', missing)
