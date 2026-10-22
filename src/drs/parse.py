"""Ball-level state table for men's Tests, ODIs, T20Is and IPL from Cricsheet JSON.
One row per delivery: match/innings state before the ball, runs and wickets on the ball, the review (if any) and the
match result from the batting team's side.  T20Is and IPL are pooled as format 'T20' (comp keeps them apart).
Match-level review metadata goes to a second table: event / season / city / teams, whether a TV umpire is
listed, the number of recorded reviews and Cricsheet's own flag info.missing = ["reviews"] (the scorer did not record
reviews for that match).  build.py uses it to keep DRS matches with zero reviews and to drop matches whose review
reporting is missing.
usage: python src/drs/parse.py  ->  data/drs_balls.parquet, data/drs_matches.parquet"""
import json, glob
import numpy as np, pandas as pd

SRC = {"tests": "Test", "odis": "ODI", "t20s": "T20", "ipl": "T20"}
rows, meta = [], []
for d_, fmt in SRC.items():
    for f in sorted(glob.glob(f"data/cricsheet/{d_}/*.json")):
        d = json.load(open(f)); info = d["info"]
        if info.get("gender") != "male" or not d.get("innings"): continue
        mid = f.split("/")[-1][:-5]; date = info["dates"][0]
        out = info.get("outcome", {}); win = out.get("winner"); res = out.get("result")  # draw / tie / no result
        dl = "method" in out
        ev = info.get("event", {}); ev = ev.get("name", "") if isinstance(ev, dict) else str(ev)
        miss = [m for m in info.get("missing", []) if isinstance(m, str)]
        nrev = sum("review" in b for inn in d["innings"] if not inn.get("super_over")
                   for ov in inn.get("overs", []) for b in ov["deliveries"])
        meta.append((fmt, d_, mid, date, ev, str(info.get("season", "")), info.get("city", ""), " v ".join(sorted(info["teams"])),
                     "tv_umpires" in info.get("officials", {}), "reviews" in miss, nrev, d["meta"].get("data_version", "")))
        teams = info["teams"]; tot = {t: 0 for t in teams}   # match aggregate per team (Tests: lead)
        mo = 0                                                 # legal balls bowled in the match so far
        for ii, inn in enumerate(d["innings"]):
            if inn.get("super_over"): break
            bat = inn["team"]; bowl = [t for t in teams if t != bat][0]
            pts = 1.0 if win == bat else 0.0 if win else 0.5 if res in ("draw", "tie") else np.nan
            tgt = inn.get("target", {}).get("runs", np.nan)
            r = w = lb = 0
            for ov in inn.get("overs", []):
                for b in ov["deliveries"]:
                    ex = b.get("extras", {}); legal = not ("wides" in ex or "noballs" in ex)
                    rv = b.get("review")
                    wk = [x["kind"] for x in b.get("wickets", [])]
                    rows.append((fmt, d_, mid, date, ii, bat, bowl, r, w, lb, mo, tot[bat] - tot[bowl], tgt,
                                 info.get("overs", np.nan), b["runs"]["total"], len(wk), wk[0] if wk else "",
                                 (rv or {}).get("by", ""), (rv or {}).get("decision", ""), bool((rv or {}).get("umpires_call")),
                                 pts, dl))
                    r += b["runs"]["total"]; w += len([k for k in wk if k not in ("retired hurt", "retired not out")])
                    lb += legal; mo += legal; tot[bat] += b["runs"]["total"]
cols = ["fmt", "comp", "match", "date", "inn", "bat", "bowl", "runs", "wkts", "balls", "match_balls", "lead", "target",
        "max_overs", "r_ball", "n_wk", "wk_kind", "rev_by", "rev_dec", "rev_uc", "pts", "dl"]
df = pd.DataFrame(rows, columns=cols)
df["date"] = pd.to_datetime(df.date)
df.to_parquet("data/drs_balls.parquet")
mt = pd.DataFrame(meta, columns=["fmt", "comp", "match", "date", "event", "season", "city", "teams", "tv_umpire", "reviews_missing",
                                 "n_reviews", "data_version"])
mt["date"] = pd.to_datetime(mt.date)
mt.to_parquet("data/drs_matches.parquet")
print(df.groupby("fmt").agg(balls=("match", "size"), matches=("match", "nunique"), reviews=("rev_by", lambda s: (s != "").sum())))
