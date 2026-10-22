"""All 2026 MLB regular-season pitches from the Stats API live feed, with ABS challenge fields.

usage: python fetch_challenges.py <datadir> [season]
Writes <datadir>/feed_pitches_<y>.parquet (one row per pitch) and feed_games_<y>.parquet
(gameData.absChallenges per game). A play-level reviewDetails is attached to the PA's last pitch.
Join to Statcast on (game_pk, at_bat_number = atBatIndex + 1, pitch_number).
"""
import sys, time, json
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


def games(y):
    d = get(f"{API}/v1/schedule?sportId=1&gameType=R&startDate={y}-03-01&endDate={y}-11-30")
    return sorted({g["gamePk"] for dd in d["dates"] for g in dd["games"]
                   if g["status"]["codedGameState"] == "F"})


def rev(r):
    if not r:
        return {}
    return dict(ch=True, ch_over=bool(r.get("isOverturned")), ch_team=r.get("challengeTeamId"),
                ch_player=(r.get("player") or {}).get("id"), ch_type=r.get("reviewType"))


def game_rows(pk):
    d = get(f"{API}/v1.1/game/{pk}/feed/live")
    if d is None:
        return pk, None, None
    gd = d["gameData"]
    abs_ = gd.get("absChallenges") or {}
    g = dict(game_pk=pk, date=gd["datetime"].get("officialDate"),
             home_id=gd["teams"]["home"]["id"], away_id=gd["teams"]["away"]["id"],
             has=abs_.get("hasChallenges"),
             **{f"{s}_{k}": (abs_.get(s) or {}).get(k) for s in ("home", "away")
                for k in ("usedSuccessful", "usedFailed", "remaining")})
    rows = []
    for p in d["liveData"]["plays"]["allPlays"]:
        a, m = p["about"], p["matchup"]
        pitches = [e for e in p["playEvents"] if e.get("isPitch")]
        prev = {}
        for e in p["playEvents"]:
            if not e.get("isPitch"):
                if "reviewDetails" in e and e["reviewDetails"].get("reviewType") == "MJ":
                    prev = e["reviewDetails"]  # rare: review logged on a non-pitch event
                continue
            det = e.get("details", {})
            r = e.get("reviewDetails")
            if r is None and e is pitches[-1] and p.get("reviewDetails", {}).get("reviewType") == "MJ":
                r = p["reviewDetails"]
            if r is not None and r.get("reviewType") != "MJ":
                r = None
            rows.append(dict(game_pk=pk, at_bat_number=a["atBatIndex"] + 1, pitch_number=e.get("pitchNumber"),
                             inning=a["inning"], top=a["isTopInning"], code=det.get("code"),
                             call=det.get("description"), balls_after=e["count"]["balls"],
                             strikes_after=e["count"]["strikes"], outs=e["count"]["outs"],
                             t0=e.get("startTime"), batter=m["batter"]["id"], pitcher=m["pitcher"]["id"],
                             pa_event=p["result"].get("eventType"), **rev(r)))
    return pk, rows, g


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    y = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
    pks = games(y)
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(game_rows, pks))
    bad = [pk for pk, r, _ in res if r is None]
    df = pd.DataFrame([row for _, r, _ in res if r for row in r])
    df.to_parquet(out / f"feed_pitches_{y}.parquet")
    pd.DataFrame([g for _, _, g in res if g]).to_parquet(out / f"feed_games_{y}.parquet")
    print(y, len(pks), "games", len(df), "pitches", int(df.ch.fillna(False).sum()), "challenges; failed:", bad, flush=True)
