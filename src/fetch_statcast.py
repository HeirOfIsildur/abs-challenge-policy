"""All 2026 regular-season pitches from Baseball Savant Statcast Search (5-day chunks).
usage: python src/fetch_statcast.py data 2026  ->  data/statcast_2026.parquet"""
import io, sys, time, datetime as dt, urllib.request
from pathlib import Path
import pandas as pd

URL = ("https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfGT=R%7C&hfSea={s}%7C&player_type=pitcher"
       "&type=details&game_date_gt={a}&game_date_lt={b}")
UA = {"User-Agent": "Mozilla/5.0 (research script)"}
out = Path(sys.argv[1]); y = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
out.mkdir(parents=True, exist_ok=True)
d, end, parts = dt.date(y, 3, 15), dt.date(y, 10, 5), []
while d <= end:
    b = min(d + dt.timedelta(days=4), end)
    for k in range(4):
        try:
            raw = urllib.request.urlopen(urllib.request.Request(URL.format(s=y, a=d, b=b), headers=UA), timeout=300).read()
            if raw.strip():
                parts.append(pd.read_csv(io.BytesIO(raw), low_memory=False))
            break
        except Exception as e:
            print("retry", d, e, flush=True); time.sleep(10 * (k + 1))
    print(d, sum(len(p) for p in parts), flush=True)
    d = b + dt.timedelta(days=1)
df = pd.concat(parts, ignore_index=True).drop_duplicates(["game_pk", "at_bat_number", "pitch_number"])
df.to_parquet(out / f"statcast_{y}.parquet")
print(y, len(df), "pitches")
