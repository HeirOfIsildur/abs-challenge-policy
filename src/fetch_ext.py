"""External validation feeds: Triple-A 2025 and MLB spring training 2026, straight from the Stats API live
feed (no Statcast there). One row per pitch with the tracking at the front of the plate (pX, pZ, strike-zone
top/bottom, initial velocity/acceleration), the pre-pitch game state and the ABS challenge fields.

usage: python fetch_ext.py <datadir> <tag> <sportId> <gameType> <start> <end>
  e.g. python src/fetch_ext.py data aaa2025 11 R 2025-03-01 2025-10-15
       python src/fetch_ext.py data spring2026 1 S 2026-02-01 2026-03-31
-> <datadir>/ext_<tag>.parquet, <datadir>/ext_games_<tag>.parquet"""
import sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd, requests

API = "https://statsapi.mlb.com/api"
S = requests.Session()


def get(url, tries=5):
    for k in range(tries):
        try:
            r = S.get(url, timeout=60)
            if r.ok:
                return r.json()
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2 * (k + 1))
    return None


def games(sport, gt, start, end):
    d = get(f"{API}/v1/schedule?sportId={sport}&gameType={gt}&startDate={start}&endDate={end}")
    return sorted({g["gamePk"] for dd in d["dates"] for g in dd["games"] if g["status"]["codedGameState"] == "F"})


def game_rows(pk):
    d = get(f"{API}/v1.1/game/{pk}/feed/live")
    if d is None:
        return pk, None, None
    gd = d["gameData"]
    abs_ = gd.get("absChallenges") or {}
    g = dict(game_pk=pk, date=gd["datetime"].get("officialDate"), home_id=gd["teams"]["home"]["id"],
             away_id=gd["teams"]["away"]["id"], has=abs_.get("hasChallenges"),
             **{f"{s}_{k}": (abs_.get(s) or {}).get(k) for s in ("home", "away")
                for k in ("usedSuccessful", "usedFailed", "remaining")})
    rows = []
    hs = as_ = 0; bases = outs = 0; last_half = None
    for p in d["liveData"]["plays"]["allPlays"]:
        a, m = p["about"], p["matchup"]
        half = (a["inning"], a["isTopInning"])
        if half != last_half:                      # new half-inning: empty bases (runner on 2nd in extras)
            bases = 2 if a["inning"] >= 10 else 0; outs = 0; last_half = half
        pitches = [e for e in p["playEvents"] if e.get("isPitch")]
        balls = strikes = 0
        for e in pitches:
            det = e.get("details", {}); pdd = e.get("pitchData") or {}; co = pdd.get("coordinates") or {}
            r = e.get("reviewDetails")
            if r is None and e is pitches[-1] and (p.get("reviewDetails") or {}).get("reviewType") == "MJ":
                r = p["reviewDetails"]
            if r is not None and r.get("reviewType") != "MJ":
                r = None
            rows.append(dict(game_pk=pk, at_bat_number=a["atBatIndex"] + 1, pitch_number=e.get("pitchNumber"),
                             inning=a["inning"], top=a["isTopInning"], code=det.get("code"),
                             balls=balls, strikes=strikes, outs_when_up=min(outs, 2), bases=bases,
                             home_score=hs, away_score=as_,
                             batter=m["batter"]["id"], pitcher=m["pitcher"]["id"], stand=m["batSide"]["code"],
                             plate_x=co.get("pX"), plate_z=co.get("pZ"), vx0=co.get("vX0"), vy0=co.get("vY0"),
                             vz0=co.get("vZ0"), ax=co.get("aX"), ay=co.get("aY"), az=co.get("aZ"),
                             sz_top=pdd.get("strikeZoneTop"), sz_bot=pdd.get("strikeZoneBottom"),
                             ch=r is not None, ch_over=bool(r.get("isOverturned")) if r else False,
                             ch_team=r.get("challengeTeamId") if r else None,
                             ch_player=((r.get("player") or {}).get("id")) if r else None))
            balls, strikes = e["count"]["balls"], e["count"]["strikes"]
        # state after the play = state before the next PA of the same half (mid-PA steals are ignored)
        res = p["result"]; hs, as_ = res.get("homeScore", hs), res.get("awayScore", as_)
        outs = p["count"]["outs"]
        bases = (1 if m.get("postOnFirst") else 0) | (2 if m.get("postOnSecond") else 0) | (4 if m.get("postOnThird") else 0)
    return pk, rows, g


if __name__ == "__main__":
    out = Path(sys.argv[1]); tag = sys.argv[2]; sport, gt, start, end = int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
    pks = games(sport, gt, start, end)
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(game_rows, pks))
    bad = [pk for pk, r, _ in res if r is None]
    df = pd.DataFrame([row for _, r, _ in res if r for row in r])
    df.to_parquet(out / f"ext_{tag}.parquet")
    gm = pd.DataFrame([g for _, _, g in res if g]); gm.to_parquet(out / f"ext_games_{tag}.parquet")
    print(tag, len(pks), "games", len(df), "pitches", int(df.ch.sum()), "challenges;",
          "games with challenges:", int(gm.has.fillna(False).sum()), "failed fetches:", len(bad), flush=True)
