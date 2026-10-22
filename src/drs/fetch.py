"""Cricsheet ball-by-ball JSON (men's and women's; parse.py keeps men's).
usage: python src/drs/fetch.py data  ->  data/cricsheet/{tests,odis,t20s,ipl}/*.json"""
import io, sys, zipfile, urllib.request
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data") / "cricsheet"
for f in ("tests", "odis", "t20s", "ipl"):
    raw = urllib.request.urlopen(f"https://cricsheet.org/downloads/{f}_json.zip", timeout=600).read()
    (out / f).mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(raw)).extractall(out / f)
    print(f, len(list((out / f).glob("*.json"))), "matches")
