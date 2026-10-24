"""HONZA-88 geometry audit: which y-plane each source reports plate_x / plate_z in, and which geometry reproduces the
2026 ABS challenge rulings.  Needs data/called_2026_oldgeom.parquet (pre-HONZA-88 build: raw and projected
coordinates side by side), data/statcast_2026.parquet, data/statcast_sample_{2024,2025}.parquet (one Savant day
each), the Stats API feed tables data/ext_{mlb2026feed,aaa2025,spring2026,post2026}.parquet and
data/statcast_post2026.parquet.  Fetches batter heights (Stats API people) and re-reads the live feed of every game
with a mismatch.
usage: python src/geometry_audit.py  -> results/geometry_audit.json, results/geometry_mismatches.csv"""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, requests
sys.path.insert(0, "src")
import geometry as G

API = "https://statsapi.mlb.com/api"
S = requests.Session()
KEY = ["game_pk", "at_bat_number", "pitch_number"]


def get(url, tries=4):
    for k in range(tries):
        try:
            r = S.get(url, timeout=60)
            if r.ok:
                return r.json()
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2 * (k + 1))
    return None


def plane_test(sc):
    """anchor the trajectory on the release point (same 9-parameter fit) and find the y at which it passes through
    the reported (plate_x, plate_z)."""
    s = sc[sc.plate_x.notna() & sc.vy0.notna() & sc.release_pos_y.notna()]
    tr = G.time_to(s.vy0, s.ay, s.release_pos_y)
    x50 = s.release_pos_x - (s.vx0 * tr + 0.5 * s.ax * tr ** 2); z50 = s.release_pos_z - (s.vz0 * tr + 0.5 * s.az * tr ** 2)
    out = dict(n=len(s))
    for nm, y in (("front", G.Y_FRONT_FT), ("mid", G.Y_MID_FT), ("back", 0.0)):
        t = G.time_to(s.vy0, s.ay, y)
        dz = 12 * (z50 + s.vz0 * t + 0.5 * s.az * t * t - s.plate_z); dx = 12 * (x50 + s.vx0 * t + 0.5 * s.ax * t * t - s.plate_x)
        out[nm] = dict(dz_mean=float(dz.mean()), dz_sd=float(dz.std()), dx_mean=float(dx.mean()), dx_sd=float(dx.std()))
    ys = np.linspace(-1, 3, 401)
    Z = np.stack([z50 + s.vz0 * G.time_to(s.vy0, s.ay, y) + 0.5 * s.az * G.time_to(s.vy0, s.ay, y) ** 2 for y in ys], 1)
    ystar = ys[np.abs(Z - s.plate_z.values[:, None]).argmin(1)]
    out["ystar_ft_quantiles_5_25_50_75_95"] = np.percentile(ystar, [5, 25, 50, 75, 95]).round(3).tolist()
    return out


def heights(ids):
    ids = sorted(set(int(i) for i in ids if pd.notna(i)))
    rows = []
    for k in range(0, len(ids), 150):
        d = get(f"{API}/v1/people?personIds={','.join(map(str, ids[k:k + 150]))}")
        for p in (d or {}).get("people", []):
            h = p.get("height") or ""
            try:
                ft, inch = h.replace('"', "").split("' ")
                rows.append(dict(batter=p["id"], name=p.get("fullName"), height_in=12 * int(ft) + int(inch)))
            except ValueError:
                pass
    return pd.DataFrame(rows)


def rep(abs_s, px, pz, top, bot, **kw):
    pred = G.abs_strike(px, pz, top, bot, **kw)
    return float((pred == abs_s).mean()), int((pred != abs_s).sum())


def feed_events(pk):
    """per-pitch outcome text, review details and timing from the live feed of one game."""
    d = get(f"{API}/v1.1/game/{pk}/feed/live")
    rows = []
    if d is None:
        return rows
    for p in d["liveData"]["plays"]["allPlays"]:
        pitches = [e for e in p["playEvents"] if e.get("isPitch")]
        for e in pitches:
            r = e.get("reviewDetails") or ((p.get("reviewDetails") or {}) if e is pitches[-1] else {})
            if r.get("reviewType") != "MJ":
                continue
            det = e.get("details", {}); pdd = e.get("pitchData") or {}
            rows.append(dict(game_pk=pk, at_bat_number=p["about"]["atBatIndex"] + 1, pitch_number=e.get("pitchNumber"),
                             call_text=det.get("description"), play_text=p["result"].get("description"),
                             review_overturned=r.get("isOverturned"), review_in_progress=r.get("inProgress"),
                             review_player=(r.get("player") or {}).get("fullName"), t_start=e.get("startTime"),
                             t_end=e.get("endTime"), feed_zone=pdd.get("zone"), feed_sz_top=pdd.get("strikeZoneTop"),
                             feed_sz_bot=pdd.get("strikeZoneBottom"), review_on_last_pitch_only=e.get("reviewDetails") is None))
    return rows


def classify(r):
    """tracking disagreement: Statcast and the (projected) feed differ by > 0.5 in on this pitch, and the feed location
    agrees with the ruling (Statcast was revised after the live call); boundary tie: both sources agree and the pitch
    is within 0.5 in of the boundary; mislabel: both sources agree, the pitch is > 0.5 in from the boundary and the
    recorded ruling contradicts it; other: anything else."""
    feed_ok = (r.d_feed_mid > 0) == r.abs_s
    if abs(r.d_statcast - r.d_feed_mid) > 0.5:
        return "tracking disagreement (feed agrees with ruling)" if feed_ok else "other"
    if abs(r.d_statcast) <= 0.5:
        return "boundary tie (|d| <= 0.5 in, sources agree)"
    return "mislabel"


if __name__ == "__main__":
    res = {}
    # 1. reference plane of plate_x/plate_z: Statcast 2024 / 2025 (one day each) and the full 2026 season
    sc = pd.read_parquet("data/statcast_2026.parquet")
    res["plane"] = {"statcast_2026": plane_test(sc)}
    for y in (2024, 2025):
        p = Path(f"data/statcast_sample_{y}.parquet")
        if p.exists():
            res["plane"][f"statcast_{y}_one_day"] = plane_test(pd.read_parquet(p))
    print(json.dumps(res["plane"], indent=1), flush=True)

    # 2. challenged pitches: the ABS ruling is observed
    m = pd.read_parquet("data/called_2026_oldgeom.parquet")
    # HONZA-122 (Fable review): drop feed-Statcast mis-joins (the Statcast description contradicts the feed's call
    # code, so the location is another pitch's; build.py); 6 of the 11 earlier mismatches were mis-joins
    mj = ~((m.code.eq("C") & m.description.eq("called_strike")) | (m.code.isin(["B", "*B"]) & m.description.isin(["ball", "blocked_ball"])))
    res["misjoined_challenged_dropped"] = int((mj & m.ch & m.plate_x.notna()).sum()); m = m[~mj | m.plate_x.isna()]
    c = m[m.ch & m.plate_x.notna()].copy()
    c["abs_s"] = np.where(c.ch_over, ~c.orig_strike, c.orig_strike)
    sc_p = sc[KEY + ["p_throws", "game_date", "release_pos_x", "release_pos_y", "release_pos_z"]]
    c = c.merge(sc_p, on=KEY, how="left")
    f = pd.read_parquet("data/ext_mlb2026feed.parquet")
    fc = ["plate_x", "plate_z", "sz_top", "sz_bot", "vx0", "vy0", "vz0", "ax", "ay", "az"]
    c = c.merge(f[KEY + fc].rename(columns={k: k + "_f" for k in fc}), on=KEY, how="left")
    pf = c[[k + "_f" for k in fc]].rename(columns=lambda k: k[:-2])
    c["px_fmid"], c["pz_fmid"] = G.abs_location(pf, "feed")
    c["d_statcast"], c["edge_sc"] = G.signed_distance(c.plate_x, c.plate_z, c.sz_top, c.sz_bot)
    c["d_old"], _ = G.signed_distance(c.px_mid, c.pz_mid, c.sz_top, c.sz_bot)
    c["d_feed_raw"], _ = G.signed_distance(c.plate_x_f, c.plate_z_f, c.sz_top_f, c.sz_bot_f)
    c["d_feed_mid"], _ = G.signed_distance(c.px_fmid, c.pz_fmid, c.sz_top_f, c.sz_bot_f)
    # feed vs Statcast on the same pitches
    j = c[c.plate_x_f.notna()]
    res["feed_vs_statcast"] = dict(
        n=len(j), dz_raw_in=float(12 * (j.plate_z_f - j.plate_z).mean()), dz_raw_sd=float(12 * (j.plate_z_f - j.plate_z).std()),
        dx_raw_in=float(12 * (j.plate_x_f - j.plate_x).mean()), dz_feed_projected_to_mid_in=float(12 * (j.pz_fmid - j.plate_z).mean()),
        dz_feed_projected_sd=float(12 * (j.pz_fmid - j.plate_z).std()),
        dz_feed_projected_abs_p99=float(np.percentile(12 * np.abs(j.pz_fmid - j.plate_z), 99)),
        dx_feed_projected_to_mid_in=float(12 * (j.px_fmid - j.plate_x).mean()),
        sz_top_diff_in=float(12 * (j.sz_top_f - j.sz_top).abs().mean()), sz_bot_diff_in=float(12 * (j.sz_bot_f - j.sz_bot).abs().mean()),
        sz_identical_share=float(((j.sz_top_f - j.sz_top).abs() < 0.01 / 12).mean()),
        trajectory_identical_share=float(((j.vz0_f - j.vz0).abs() < 1e-3).mean()))
    # where along y does the feed's pZ lie? anchor on the Statcast release point (identical trajectory parameters)
    jj = j[(j.vz0_f - j.vz0).abs() < 1e-3].dropna(subset=["release_pos_y"])
    tr = G.time_to(jj.vy0, jj.ay, jj.release_pos_y); z50 = jj.release_pos_z - (jj.vz0 * tr + 0.5 * jj.az * tr ** 2)
    ys = np.linspace(-1, 3, 401)
    Z = np.stack([z50 + jj.vz0 * G.time_to(jj.vy0, jj.ay, y) + 0.5 * jj.az * G.time_to(jj.vy0, jj.ay, y) ** 2 for y in ys], 1)
    res["feed_vs_statcast"]["feed_ystar_ft_quantiles_5_25_50_75_95"] = np.percentile(ys[np.abs(Z - jj.plate_z_f.values[:, None]).argmin(1)], [5, 25, 50, 75, 95]).round(3).tolist()

    # 3. rule reproduction under each geometry
    rr = {}
    for nm, (x, z, t, b) in dict(statcast_raw=(c.plate_x, c.plate_z, c.sz_top, c.sz_bot),
                                 statcast_projected_old=(c.px_mid, c.pz_mid, c.sz_top, c.sz_bot)).items():
        rr[nm] = dict(zip(("share", "mismatches"), rep(c.abs_s, x, z, t, b)), n=len(c))
        rr[nm + "_no_ball_radius"] = dict(zip(("share", "mismatches"), rep(c.abs_s, x, z, t, b, R=0.0)), n=len(c))
    # HONZA-120: ball-radius sweep on raw Statcast (round-ball rule) and the square-corner rule used up to HONZA-114
    rr["statcast_raw_radius_sweep"] = {f"{r_in:.2f}": rep(c.abs_s, c.plate_x, c.plate_z, c.sz_top, c.sz_bot, R=r_in / 12)[1]
                                       for r_in in (1.35, 1.40, 1.43, 1.44, 1.45, 1.46, 1.47, 1.50, 1.55)}
    def square_corner(x, z, t, b, R=G.BALL_RADIUS_FT, hw=G.PLATE_HALF_WIDTH_FT):
        x, z, t, b = (np.asarray(v, float) for v in (x, z, t, b))
        return (hw + R - np.abs(x) > 0) & (np.minimum(z - (b - R), (t + R) - z) > 0)
    sq = square_corner(c.plate_x, c.plate_z, c.sz_top, c.sz_bot)
    rr["statcast_raw_square_corners"] = dict(share=float((sq == c.abs_s).mean()), mismatches=int((sq != c.abs_s).sum()), n=len(c),
                                             relabelled_challenged=int((sq != G.abs_strike(c.plate_x, c.plate_z, c.sz_top, c.sz_bot)).sum()))
    a_ = m[m.plate_x.notna() & m.sz_top.notna()]
    rr["statcast_raw_square_corners"]["relabelled_called"] = int((square_corner(a_.plate_x, a_.plate_z, a_.sz_top, a_.sz_bot)
                                                                  != G.abs_strike(a_.plate_x, a_.plate_z, a_.sz_top, a_.sz_bot)).sum())
    rr["statcast_raw_square_corners"]["n_called"] = len(a_)
    rr["feed_raw_front"] = dict(zip(("share", "mismatches"), rep(j.abs_s, j.plate_x_f, j.plate_z_f, j.sz_top_f, j.sz_bot_f)), n=len(j))
    rr["feed_projected_mid"] = dict(zip(("share", "mismatches"), rep(j.abs_s, j.px_fmid, j.pz_fmid, j.sz_top_f, j.sz_bot_f)), n=len(j))
    # best constant vertical / horizontal shift of the zone (grid): where do the rulings flip?
    for nm, (x, z) in dict(statcast_raw=(c.plate_x, c.plate_z), statcast_projected_old=(c.px_mid, c.pz_mid)).items():
        best = {}
        for e, lab in (("bottom", "sz_bot"), ("top", "sz_top")):
            sh = np.arange(-2.0, 2.01, 0.05)
            acc = [((G.abs_strike(x, z, c.sz_top + (s / 12 if e == "top" else 0), c.sz_bot + (s / 12 if e == "bottom" else 0))) == c.abs_s).mean() for s in sh]
            best[f"{e}_line_shift_in"] = float(sh[int(np.argmax(acc))]); best[f"{e}_best_share"] = float(max(acc))
        rr[nm + "_best_line_shift"] = best
    res["reproduction"] = rr

    # by edge (nearest edge in the ABS plane), batter side, month, challenger
    mid = (c.sz_top + c.sz_bot) / 2
    inside = np.where(c.stand == "R", c.plate_x < 0, c.plate_x > 0)            # toward the batter
    armside = np.where(c.p_throws == "R", c.plate_x > 0, c.plate_x < 0)        # pitcher's arm side (catcher's view +x = 3B side)
    c["edge4"] = np.where(c.edge_sc == "vert", np.where(c.plate_z > mid, "top", "bottom"), np.where(inside, "inside (lik 'arm')", "outside (lik 'glove')"))
    c["edge_pitcher"] = np.where(c.edge_sc == "vert", np.where(c.plate_z > mid, "top", "bottom"), np.where(armside, "pitcher arm side", "pitcher glove side"))
    c["month"] = pd.to_datetime(c.game_date).dt.strftime("%Y-%m")
    c["ok_raw"] = G.abs_strike(c.plate_x, c.plate_z, c.sz_top, c.sz_bot) == c.abs_s
    c["ok_old"] = G.abs_strike(c.px_mid, c.pz_mid, c.sz_top, c.sz_bot) == c.abs_s
    by = {}
    for key in ("edge4", "edge_pitcher", "stand", "month", "role"):
        t = c.groupby(key).agg(n=("ok_raw", "size"), raw=("ok_raw", "mean"), projected=("ok_old", "mean"), mismatches_raw=("ok_raw", lambda s: int((~s).sum())))
        by[key] = {str(k): {kk: (float(v) if kk != "n" and kk != "mismatches_raw" else int(v)) for kk, v in row.items()} for k, row in t.iterrows()}
    res["by"] = by

    # 4. zone lines vs listed height
    hts = heights(sc.batter.unique())
    hts.to_parquet("data/people_heights.parquet")
    zs = sc[sc.sz_top.notna()].groupby("batter").agg(sz_top=("sz_top", "median"), sz_bot=("sz_bot", "median"),
                                                     sz_top_sd=("sz_top", "std"), n=("sz_top", "size")).reset_index().merge(hts, on="batter")
    dt_ = 12 * zs.sz_top - G.SZ_TOP_FRAC * zs.height_in; db_ = 12 * zs.sz_bot - G.SZ_BOT_FRAC * zs.height_in
    res["zone_vs_height"] = dict(batters=len(zs), top_mean_in=float(dt_.mean()), top_sd_in=float(dt_.std()), bot_mean_in=float(db_.mean()),
                                 bot_sd_in=float(db_.std()), share_within_025in=float(((dt_.abs() <= 0.25) & (db_.abs() <= 0.25)).mean()),
                                 max_abs_in=float(max(dt_.abs().max(), db_.abs().max())),
                                 within_batter_sd_in_median=float(12 * zs.sz_top_sd.median()),
                                 off_batters=zs.loc[(dt_.abs() > 0.25) | (db_.abs() > 0.25), ["batter", "name", "height_in", "sz_top", "sz_bot", "n"]].assign(sz_top=lambda d: 12 * d.sz_top, sz_bot=lambda d: 12 * d.sz_bot).to_dict("records"))
    print(json.dumps(res["zone_vs_height"], default=str)[:600], flush=True)

    # 5. other feeds: is the feed convention (front of plate) the same everywhere?  Feed geometry only (no Statcast).
    ext = {}
    for tag in ("mlb2026feed", "aaa2025", "spring2026", "post2026"):
        p = Path(f"data/ext_{tag}.parquet")
        if not p.exists():
            continue
        e = pd.read_parquet(p)
        e = e[e.ch.fillna(False).astype(bool) & e.code.isin(["C", "B", "*B"]) & e.plate_x.notna() & e.sz_top.notna() & e.vy0.notna()].copy()
        final = e.code.eq("C"); orig = np.where(e.ch_over, ~final, final); a_s = np.where(e.ch_over, ~orig, orig)
        px, pz = G.abs_location(e, "feed")
        ext[tag] = dict(n=len(e), raw=dict(zip(("share", "mismatches"), rep(a_s, e.plate_x, e.plate_z, e.sz_top, e.sz_bot))),
                        projected_to_mid=dict(zip(("share", "mismatches"), rep(a_s, px, pz, e.sz_top, e.sz_bot))))
    sp = Path("data/statcast_post2026.parquet")
    if sp.exists() and Path("data/ext_post2026.parquet").exists():
        e = pd.read_parquet("data/ext_post2026.parquet")
        e = e[e.ch.fillna(False).astype(bool) & e.code.isin(["C", "B", "*B"])][KEY + ["code", "ch_over"]]
        e = e.merge(pd.read_parquet(sp)[KEY + ["plate_x", "plate_z", "sz_top", "sz_bot", "vx0", "vy0", "vz0", "ax", "ay", "az", "release_pos_x", "release_pos_y", "release_pos_z"]], on=KEY).dropna(subset=["plate_x"])
        final = e.code.eq("C"); orig = np.where(e.ch_over, ~final, final); a_s = np.where(e.ch_over, ~orig, orig)
        ext["post2026_statcast"] = dict(n=len(e), raw=dict(zip(("share", "mismatches"), rep(a_s, e.plate_x, e.plate_z, e.sz_top, e.sz_bot))))
        res["plane"]["statcast_post2026"] = plane_test(pd.read_parquet(sp))
    res["other_feeds"] = ext
    print(json.dumps(ext, indent=1), flush=True)

    # 6. the mismatches under the raw rule, one by one
    mm = c[~c.ok_raw].copy()
    with ThreadPoolExecutor(8) as ex:
        ev = pd.DataFrame([r for rows in ex.map(feed_events, mm.game_pk.unique()) for r in rows])
    mm = mm.merge(ev, on=KEY, how="left").merge(hts[["batter", "height_in"]], on="batter", how="left")
    mm["sz_top_vs_height_in"] = 12 * mm.sz_top - G.SZ_TOP_FRAC * mm.height_in
    mm["sz_bot_vs_height_in"] = 12 * mm.sz_bot - G.SZ_BOT_FRAC * mm.height_in
    # the batter's own usual zone (season median) vs the zone on this pitch
    mm = mm.merge(zs[["batter", "sz_top", "sz_bot"]].rename(columns={"sz_top": "sz_top_med", "sz_bot": "sz_bot_med"}), on="batter", how="left")
    mm["sz_top_vs_own_median_in"] = 12 * (mm.sz_top - mm.sz_top_med); mm["sz_bot_vs_own_median_in"] = 12 * (mm.sz_bot - mm.sz_bot_med)
    mm["class"] = mm.apply(classify, axis=1)
    mm["ruling"] = np.where(mm.abs_s, "strike", "ball"); mm["orig_call"] = np.where(mm.orig_strike, "strike", "ball")
    cols = ["game_pk", "game_date", "inning", "at_bat_number", "pitch_number", "stand", "role", "orig_call", "ch_over", "ruling",
            "edge4", "d_statcast", "d_old", "d_feed_raw", "d_feed_mid", "plate_x", "plate_z", "sz_top", "sz_bot",
            "height_in", "sz_top_vs_height_in", "sz_bot_vs_height_in", "sz_top_vs_own_median_in", "sz_bot_vs_own_median_in",
            "call_text", "play_text", "review_overturned", "review_player", "t_start", "t_end", "class"]
    mm[cols].sort_values("d_statcast").to_csv("results/geometry_mismatches.csv", index=False)
    res["mismatches"] = dict(n=len(mm), by_class=mm["class"].value_counts().to_dict(),
                             abs_d_statcast_quantiles_50_90_max=np.percentile(mm.d_statcast.abs(), [50, 90, 100]).round(3).tolist(),
                             within_1in=int((mm.d_statcast.abs() <= 1).sum()), within_05in=int((mm.d_statcast.abs() <= 0.5).sum()))
    # the near-boundary benchmark: how many challenged pitches lie within 0.5 in of the boundary at all
    res["mismatches"]["challenged_within_05in"] = int((c.d_statcast.abs() <= 0.5).sum())
    res["mismatches"]["mismatch_rate_within_05in"] = float((~c.ok_raw[c.d_statcast.abs() <= 0.5]).mean())
    res["mismatches"]["mismatch_rate_beyond_1in"] = float((~c.ok_raw[c.d_statcast.abs() > 1]).mean())
    dis = (c.d_statcast - c.d_feed_mid).abs() > 0.5
    res["mismatches"]["challenged_statcast_feed_disagree_05in"] = int(dis.sum())
    res["mismatches"]["disagree_feed_right_statcast_wrong"] = int((dis & ~c.ok_raw & ((c.d_feed_mid > 0) == c.abs_s)).sum())
    res["mismatches"]["disagree_statcast_right_feed_wrong"] = int((dis & c.ok_raw & ((c.d_feed_mid > 0) != c.abs_s)).sum())
    res["mismatches"]["boundary_ties_d_positive_ruled_ball"] = int((~c.ok_raw & ~dis & (c.d_statcast > 0)).sum())
    res["mismatches"]["boundary_ties_d_negative_ruled_strike"] = int((~c.ok_raw & ~dis & (c.d_statcast <= 0)).sum())
    json.dump(res, open("results/geometry_audit.json", "w"), indent=1, default=str)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(mm[cols].sort_values("d_statcast").to_string())
