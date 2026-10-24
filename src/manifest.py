"""provenance of a build -- git commit, package versions, raw-input checksums and
retrieval dates, and the result files with the time they were written.
usage: python src/manifest.py  -> results/manifest.json"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

RAW = ["statcast_2026.parquet", "feed_pitches_2026.parquet", "feed_games_2026.parquet", "statcast_post2026.parquet",
       "ext_post2026.parquet", "ext_mlb2026feed.parquet", "ext_aaa2025.parquet", "ext_spring2026.parquet"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    raw = {f: dict(sha256=sha(f"data/{f}"), modified=time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(f"data/{f}"))))
           for f in RAW if os.path.exists(f"data/{f}")}
    res = {p.name: dict(written=time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)), sha256=sha(p))
           for p in sorted(Path("results").glob("*.json"))}        # result checksums as well
    def tree_sha(d):                                 # one checksum per Cricsheet folder: names and contents of its files
        h = hashlib.sha256()
        for q in sorted(Path(d).rglob("*")):
            if q.is_file():
                h.update(str(q.relative_to(d)).encode()); h.update(sha(q).encode())
        return h.hexdigest()
    cric = {p.name: dict(sha256=tree_sha(p), files=sum(1 for q in p.rglob("*") if q.is_file()))
            for p in sorted(Path("data/cricsheet").iterdir()) if p.is_dir()} if Path("data/cricsheet").exists() else {}
    out = dict(commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain", "--", "src", "paper", "wolfram")),
               python=sys.version.split()[0], packages=Path("requirements-lock.txt").read_text().split(),
               raw_inputs=raw, cricket_raw_inputs=cric, results=res,
               note="raw data are public and not redistributed (src/fetch_*.py re-fetch them; feeds can be revised, "
                    "so the checksums identify the snapshot used)")
    json.dump(out, open("results/manifest.json", "w"), indent=1); print(out["commit"], len(raw), "raw inputs")
