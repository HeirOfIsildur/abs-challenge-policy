"""Independent recount of DRS reviews from raw Cricsheet JSON (does not import src/).
Scope mirrors the paper: men's Tests/ODIs/T20Is from 2017-10-01, IPL from 2018-01-01, DRS matches (final rule: a recorded review, or a reviewed match of the same event and season between the same
teams or at the same city; reviews-missing matches excluded),
'technical failure' reviews dropped.  Counts per format: batting-side out decisions (lbw / caught / c&b not given on
a fielding review, plus overturned batting reviews), batting reviews by outcome, fielding reviews by outcome, and
reviews taken with none left under the allowance schedule (official ICC / IPL dates, see audit_drs.md).
usage: .venv/bin/python audit/drs/recount.py  ->  audit/drs/recount.json"""
import json, glob, collections, datetime as dt

SRC = {"tests": "Test", "odis": "ODI", "t20s": "T20", "ipl": "T20"}
REV = {"lbw", "caught", "caught and bowled"}

def allowance(comp, date):
    # ICC: Oct 2017 playing conditions (Tests 2, ODI 1, T20I 1; umpire's call retains); interim COVID regulation
    # (first used ENG v WI, 8 Jul 2020) adds one per innings in every format.  IPL: 1 until 2021, 2 from 2022.
    if comp == "tests": return 2 if date < "2020-07-01" else 3
    if comp == "ipl": return 1 if date < "2022-01-01" else 2
    return 1 if date < "2020-07-01" else 2

C = collections.defaultdict(collections.Counter); types = collections.Counter(); viol = collections.Counter()
# The sample rule of the paper, coded here independently of src/drs/build.py. A match is a DRS match if it
# has a recorded review, or none but another match of the same competition, event and season between the same teams
# or at the same city has one; matches flagged info.missing = reviews are excluded.
def meta(d_, f):
    d = json.load(open(f)); info = d["info"]
    ev = info.get("event", {}); ev = ev.get("name", "") if isinstance(ev, dict) else str(ev)
    miss = any("review" in str(x) for x in info.get("missing", []))
    nrev = sum(1 for inn in d.get("innings", []) if not inn.get("super_over") for ov in inn.get("overs", [])
               for b in ov["deliveries"] if "review" in b and b["review"].get("decision") != "technical failure")
    return d, info, (d_, ev, str(info.get("season", ""))), "|".join(sorted(info.get("teams", []))), info.get("city", ""), miss, nrev

M = {}
for d_ in SRC:
    start = "2018-01-01" if d_ == "ipl" else "2017-10-01"
    for f in sorted(glob.glob(f"data/cricsheet/{d_}/*.json")):
        d, info, key, teams, city, miss, nrev = meta(d_, f)
        if info.get("gender") != "male" or info["dates"][0] < start or not d.get("innings"): continue
        M[f] = (key, teams, city, miss, nrev)
rev_teams = {(k, t) for k, t, c, ms, n in M.values() if n > 0 and not ms}
rev_city = {(k, c) for k, t, c, ms, n in M.values() if n > 0 and not ms and c}
KEEP = {f for f, (k, t, c, ms, n) in M.items() if not ms and (n > 0 or (k[1] and ((k, t) in rev_teams or (c and (k, c) in rev_city))))}

for d_, fmt in SRC.items():
    start = "2018-01-01" if d_ == "ipl" else "2017-10-01"
    for f in sorted(glob.glob(f"data/cricsheet/{d_}/*.json")):
        if f not in KEEP: continue
        d = json.load(open(f)); info = d["info"]; date = info["dates"][0]
        c = C[fmt]; c["matches"] += 1
        A = allowance(d_, date)
        for inn in d["innings"]:
            if inn.get("super_over"): break
            bat = inn["team"]; left = collections.Counter()
            for ov in inn.get("overs", []):
                for b in ov["deliveries"]:
                    r = b.get("review"); wk = [w["kind"] for w in b.get("wickets", [])]
                    if r and r.get("decision") == "technical failure": continue
                    if r: types[r.get("type", "")] += 1
                    side = None if not r else ("bat" if r["by"] == bat else "bowl")
                    dec = r and r["decision"]; uc = bool(r and r.get("umpires_call"))
                    if r:
                        if left[side] >= A: viol[(fmt, side)] += 1
                        if dec == "struck down" and not uc: left[side] += 1
                        o = "success" if dec == "upheld" else "uc" if uc else "lost"
                        c[f"{side}_rev"] += 1; c[f"{side}_{o}"] += 1
                    # batting-side opportunity
                    if side == "bat" and dec == "upheld": c["bat_decisions"] += 1
                    elif wk and wk[0] in REV and not (side == "bowl" and dec == "upheld"):
                        if side == "bat" or side is None or side == "bowl": c["bat_decisions"] += 1
                        if wk[0] == "lbw": c["bat_dec_lbw"] += 1
                        if side == "bat" and wk[0] == "lbw": c["bat_rev_lbw_failed"] += 1
out = dict(counts={k: dict(v) for k, v in C.items()}, review_types=dict(types),
           none_left={f"{a}_{b}": n for (a, b), n in viol.items()})
json.dump(out, open("audit/drs/recount.json", "w"), indent=1)
print(json.dumps(out, indent=1))
