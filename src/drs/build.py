"""Review opportunities with their match state, reviews left and outcome.

Scope: men's DRS matches in the retained-umpire's-call era: Tests, ODIs and T20Is from 1 Oct 2017, IPL from 2018.
Sample (HONZA-122; it used to be "matches with >= 1 recorded review", which selects on the outcome): a match is a DRS
match if it has a recorded review, or if it has none but DRS was in use around it -- another match of the same
competition, event and season, between the same teams or at the same city, has a recorded review (DRS is set per
series / tournament and host).  Matches that Cricsheet flags as info.missing = ["reviews"] (review reporting missing)
are excluded whatever the rest says.  Counts -> results/drs/sample.json.  Allowances (unsuccessful reviews per team-innings, read off the data, never exceeded):
Test 2 -> 3 from Jul 2020; ODI and T20I 1 -> 2 from Jul 2020; IPL 1 -> 2 from 2022.  A review that ends in
umpire's call is kept (no reviews left is decremented).

Batting side ("bat"): the opportunity set is observed -- every out decision the batter could review, i.e. every
lbw / caught / caught-and-bowled dismissal that was not given on a fielding-side review, plus every batting review
that was upheld (decision overturned, no wicket).  kind = lbw / caught, unknown ('') for overturned decisions.
Fielding side ("bowl"): only the reviews are observed (not-out appeals that were not reviewed leave no trace).
outcome: none / success / uc (failed, umpire's call, kept) / lost.
usage: python src/drs/build.py  ->  data/drs_opps.parquet, data/drs_innings.parquet"""
import numpy as np, pandas as pd
from scipy.optimize import brentq

import json

def drs_sample():
    """match-level DRS sample from the parse.py metadata; returns the match ids kept and the counts"""
    mt = pd.read_parquet("data/drs_matches.parquet")
    mt = mt[(mt.date >= "2017-10-01") & ((mt.comp != "ipl") | (mt.date >= "2018-01-01"))].copy()
    mt["rev"] = mt.n_reviews > 0
    ok = mt[mt.rev & ~mt.reviews_missing]
    grp = lambda x: x.comp + "|" + x.event + "|" + x.season
    by_teams = set(grp(ok) + "|" + ok.teams); by_city = set(grp(ok) + "|" + ok.city)
    near = (mt.event != "") & ((grp(mt) + "|" + mt.teams).isin(by_teams) | ((mt.city != "") & (grp(mt) + "|" + mt.city).isin(by_city)))
    mt["status"] = np.select([mt.reviews_missing, mt.rev, near], ["excluded_reviews_missing", "reviewed", "added_zero_review"],
                             "no_drs_zero_review")
    keep = set(mt.match[mt.status.isin(["reviewed", "added_zero_review"])])
    cnt = {}
    for fmt, g in mt.groupby("fmt"):
        st = g.status.value_counts().to_dict()
        k = g[g.status.isin(["reviewed", "added_zero_review"])]
        x = k.n_reviews.values; mpos = x[x > 0].mean()           # zero-truncated Poisson: implied share of zero-review matches
        lam = brentq(lambda l: l / (1 - np.exp(-l)) - mpos, 1e-6, 100) if mpos > 1 else 0.0
        cnt[fmt] = dict(matches=int(len(k)), reviewed=int(st.get("reviewed", 0)), added_zero_review=int(st.get("added_zero_review", 0)),
                        excluded_reviews_missing=int(st.get("excluded_reviews_missing", 0)),
                        not_drs_zero_review=int(st.get("no_drs_zero_review", 0)),
                        added_without_tv_umpire=int((g.status.eq("added_zero_review") & ~g.tv_umpire).sum()),
                        zero_share_obs=round(float((x == 0).mean()), 4), zero_share_ztpoisson=round(float(np.exp(-lam)), 4),
                        by_comp={c: {k_: int(v) for k_, v in h.status.value_counts().items()} for c, h in g.groupby("comp")})
    return keep, cnt

b = pd.read_parquet("data/drs_balls_g.parquet")
b = b[(b.date >= "2017-10-01") & ((b.comp != "ipl") | (b.date >= "2018-01-01"))]
b = b[b.rev_dec != "technical failure"]
keep, cnt = drs_sample()
old = b.groupby("match").rev_by.transform(lambda s: (s != "").any())        # the old rule, for the record
for fmt in cnt: cnt[fmt]["old_rule_matches"] = int(b[old & (b.fmt == fmt)].match.nunique())
b = b[b.match.isin(keep)].reset_index(drop=True)
for fmt in cnt: cnt[fmt]["matches_with_balls"] = int(b[b.fmt == fmt].match.nunique())
json.dump(cnt, open("results/drs/sample.json", "w"), indent=1)
print(json.dumps(cnt, indent=1))

def allowance(comp, date):
    if comp == "tests": return np.where(date < "2020-07-01", 2, 3)
    if comp == "ipl": return np.where(date < "2022-01-01", 1, 2)
    return np.where(date < "2020-07-01", 1, 2)
b["A"] = 0
for c, k in b.groupby("comp").groups.items():
    b.loc[k, "A"] = allowance(c, b.loc[k, "date"].values.astype("datetime64[D]").astype(str))

# reviews left before each ball, per innings and side
b["key"] = b.match + "_" + b.inn.astype(str)
for side, who in (("bat", b.bat), ("bowl", b.bowl)):
    rev = (b.rev_by == who)
    lost = (rev & (b.rev_dec == "struck down") & ~b.rev_uc).astype(int)
    b[f"c_{side}"] = b.A - (lost.groupby(b.key).cumsum() - lost)
    b[f"rev_{side}"] = rev
viol = {s: int(((b[f"c_{s}"] <= 0) & b[f"rev_{s}"]).sum()) for s in ("bat", "bowl")}
print("reviews taken with none left:", viol)

REV = {"lbw", "caught", "caught and bowled"}
def outcome(rev, d):
    return np.select([~rev, d.rev_dec == "upheld", d.rev_uc], ["none", "success", "uc"], "lost")

bat = b[(b.wk_kind.isin(REV) & ~(b.rev_bowl & (b.rev_dec == "upheld")) & ~(b.rev_bat & (b.rev_dec == "upheld")))
        | (b.rev_bat & (b.rev_dec == "upheld"))].copy()
bat = bat[~(bat.rev_bat & (bat.n_wk > 0) & ~bat.wk_kind.isin(REV))]
bat["side"] = "bat"; bat["c"] = bat.c_bat
bat["kind"] = np.where(bat.n_wk > 0, np.where(bat.wk_kind == "lbw", "lbw", "caught"), "")
bat["outcome"] = outcome(bat.rev_bat.values, bat)
bowl = b[b.rev_bowl].copy()
bowl["side"] = "bowl"; bowl["c"] = bowl.c_bowl; bowl["kind"] = ""
bowl["outcome"] = outcome(np.ones(len(bowl), bool), bowl)
cols = ["fmt", "comp", "match", "date", "inn", "key", "bat", "bowl", "A", "c", "wkts", "balls", "match_balls", "g", "V0",
        "side", "kind", "outcome"]
opp = pd.concat([bat[cols], bowl[cols]]).reset_index(drop=True)
opp["oid"] = np.arange(len(opp))
opp.to_parquet("data/drs_opps.parquet")

# innings-level walk for the batting-side DP: from each wickets-down state the next event is a reviewable decision
# (opportunity), a non-reviewable wicket, or the end of the innings.
ev = []
for key, d in b.groupby("key", sort=False):
    w = 0; f = d.iloc[0]
    for r in d.itertuples():
        is_opp = (r.wk_kind in REV and not (r.rev_bowl and r.rev_dec == "upheld")) or (r.rev_bat and r.rev_dec == "upheld")
        if is_opp and not (r.rev_bat and r.n_wk > 0 and r.wk_kind not in REV):
            ev.append((f.fmt, f.match, key, w, "opp")); w += 0 if (r.rev_bat and r.rev_dec == "upheld") else 1
        elif r.n_wk > 0 and r.wk_kind not in ("retired hurt", "retired not out"):
            ev.append((f.fmt, f.match, key, w, "other")); w += r.n_wk
    if w < 10: ev.append((f.fmt, f.match, key, w, "end"))
pd.DataFrame(ev, columns=["fmt", "match", "key", "w", "ev"]).to_parquet("data/drs_innings.parquet")
s = opp.groupby(["fmt", "side", "outcome"]).size().unstack(fill_value=0)
print(s); print(opp.groupby("fmt").match.nunique())
