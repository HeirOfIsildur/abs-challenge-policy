"""Guardrails for the ABS geometry: the convention that broke once must not break silently again.

Data tests read data/ (not committed) and are skipped when it is absent; unit tests always run.
run: python -m pytest tests  (or make check)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import geometry as G

DATA = ROOT / "data"


def need(*names):
    miss = [n for n in names if not (DATA / n).exists()]
    if miss:
        pytest.skip(f"data not present: {miss}")


# ---------- unit tests (no data) ----------

def test_convention_constant():
    assert G.SOURCE_PLANE_Y_FT["statcast2026"] == G.Y_MID_FT == 17 / 24
    assert G.SOURCE_PLANE_Y_FT["feed"] == G.Y_FRONT_FT == 17 / 12


def test_statcast_2026_is_not_projected():
    df = pd.DataFrame(dict(plate_x=[0.3, -0.7], plate_z=[1.6, 3.2], vx0=[5.0, -6.0], vy0=[-130.0, -125.0], vz0=[-4.0, -7.0],
                           ax=[-10.0, 8.0], ay=[28.0, 25.0], az=[-20.0, -30.0]))
    px, pz = G.abs_location(df, "statcast2026")
    assert np.array_equal(px, df.plate_x.values) and np.array_equal(pz, df.plate_z.values)


def test_feed_projection_drops_the_ball():
    # a typical 93 mph fastball: front -> midpoint lowers it by about an inch
    px, pz = G.project(0.0, 2.5, 0.0, -136.0, -5.0, 0.0, 28.0, -16.0, G.Y_FRONT_FT)
    assert -1.3 < 12 * (pz - 2.5) < -0.6
    # projecting there and back is the identity
    x2, z2 = G.project(px, pz, 0.0, -136.0, -5.0, 0.0, 28.0, -16.0, G.Y_MID_FT, G.Y_FRONT_FT)
    assert abs(x2) < 1e-9 and abs(z2 - 2.5) < 1e-9


def test_signed_distance():
    top, bot = 3.4, 1.7
    hw, R = G.PLATE_HALF_WIDTH_FT, G.BALL_RADIUS_FT
    d, e = G.signed_distance([0.0, hw + R + 1 / 12, 0.0, 0.0], [2.55, 2.55, bot - R - 2 / 12, bot - R + 0.5 / 12], top, bot)
    assert np.allclose(d, [12 * (hw + R), -1.0, -2.0, 0.5]) and list(e) == ["horiz", "horiz", "vert", "vert"]


def test_corner_contact_is_round():
    """The ball is round, so at a corner it touches the zone only within R of the corner point."""
    top, bot = 3.4, 1.7
    hw, R = G.PLATE_HALF_WIDTH_FT, G.BALL_RADIUS_FT
    u = 1 / np.sqrt(2)
    # ball centre on the diagonal off the top-right corner, at distance r from the corner point
    for r_in, want in ((1.0, R * 12 - 1.0), (1.45, 0.0), (1.6, R * 12 - 1.6)):
        r = r_in / 12
        d, _ = G.signed_distance(hw + u * r, top + u * r, top, bot)
        assert np.isclose(d, want), (r_in, d, want)
    # inside the old square corner but more than R from the corner point: a ball, not a strike
    d, _ = G.signed_distance(hw + 0.9 * R, bot - 0.9 * R, top, bot)
    assert d < 0 and not G.abs_strike(hw + 0.9 * R, bot - 0.9 * R, top, bot)
    assert np.isclose(d, 12 * (R - np.hypot(0.9 * R, 0.9 * R)))
    # off a single edge nothing changes: d = R - (distance to that edge)
    d, _ = G.signed_distance(0.0, top + 0.5 / 12, top, bot)
    assert np.isclose(d, 12 * R - 0.5)
    # rounded-corner region is symmetric in the four corners
    xs, zs = np.array([1, -1, 1, -1]) * (hw + 0.05), np.array([top + 0.05, top + 0.05, bot - 0.05, bot - 0.05])
    d4, _ = G.signed_distance(xs, zs, top, bot)
    assert np.allclose(d4, d4[0]) and np.isclose(d4[0], 12 * (R - np.hypot(0.05, 0.05)))


# ---------- data tests ----------

def test_rule_reproduces_2026_rulings():
    need("called_2026.parquet")
    m = pd.read_parquet(DATA / "called_2026.parquet")
    c = m[m.ch & m.plate_x.notna()]
    ruling = np.where(c.ch_over, ~c.orig_strike, c.orig_strike)
    share = (G.abs_strike(c.plate_x, c.plate_z, c.sz_top, c.sz_bot) == ruling).mean()
    assert len(c) > 10000 and share >= 0.998, share
    # the pipeline's own distance must be the raw-coordinate distance (no projection anywhere); mis-joined
    # rows (Statcast description contradicts the feed code) carry no distance
    c = c[c.dist_in.notna()]
    assert ((c.dist_in > 0).values == G.abs_strike(c.plate_x, c.plate_z, c.sz_top, c.sz_bot)).all()


def test_no_midpoint_projection_in_build():
    need("called_2026.parquet")
    m = pd.read_parquet(DATA / "called_2026.parquet", columns=["plate_x", "plate_z", "px_mid", "pz_mid"]).dropna()
    assert np.array_equal(m.px_mid.values, m.plate_x.values) and np.array_equal(m.pz_mid.values, m.plate_z.values)


def test_statcast_2026_reports_the_midpoint_plane():
    """guards against Savant changing the convention again: the trajectory, anchored on the release point, passes
    through (plate_x, plate_z) at y = 17/24 ft, not at the front of the plate."""
    need("statcast_2026.parquet")
    s = pd.read_parquet(DATA / "statcast_2026.parquet", columns=["plate_x", "plate_z", "vx0", "vy0", "vz0", "ax", "ay", "az",
                                                                  "release_pos_x", "release_pos_y", "release_pos_z"]).dropna().sample(20000, random_state=0)
    tr = G.time_to(s.vy0, s.ay, s.release_pos_y); z50 = s.release_pos_z - (s.vz0 * tr + 0.5 * s.az * tr ** 2)
    def dz(y):
        t = G.time_to(s.vy0, s.ay, y); return 12 * (z50 + s.vz0 * t + 0.5 * s.az * t * t - s.plate_z)
    assert abs(dz(G.Y_MID_FT).mean()) < 0.05 and dz(G.Y_MID_FT).std() < 0.1
    assert abs(dz(G.Y_FRONT_FT).mean()) > 0.8


def test_zone_lines_are_53_5_and_27_percent_of_height():
    need("statcast_2026.parquet", "people_heights.parquet")
    sc = pd.read_parquet(DATA / "statcast_2026.parquet", columns=["batter", "sz_top", "sz_bot"]).dropna()
    # both lines come from one height on every pitch
    h_top, h_bot = 12 * sc.sz_top / G.SZ_TOP_FRAC, 12 * sc.sz_bot / G.SZ_BOT_FRAC
    assert (np.abs(h_top - h_bot) < 0.01).mean() > 0.995
    # and that height is the listed height: lines within 0.25 in (certified heights differ from the whole-inch
    # listed height by up to half an inch, i.e. 0.535 x 0.5 = 0.27 in on the top line)
    z = sc.groupby("batter")[["sz_top", "sz_bot"]].median().reset_index().merge(pd.read_parquet(DATA / "people_heights.parquet"), on="batter")
    dt = np.abs(12 * z.sz_top - G.SZ_TOP_FRAC * z.height_in); db = np.abs(12 * z.sz_bot - G.SZ_BOT_FRAC * z.height_in)
    assert len(z) > 500
    assert ((dt <= 0.25) & (db <= 0.25)).mean() > 0.9 and dt.max() <= 0.27 + 1e-6 and db.max() <= 0.25
    assert abs((12 * z.sz_top - G.SZ_TOP_FRAC * z.height_in).mean()) < 0.05


def test_feed_tables_use_the_front_plane():
    """external feeds (Triple-A, spring, MLB feed) are projected: the projected rule must win."""
    for tag in ("mlb2026feed", "aaa2025", "spring2026"):
        if not (DATA / f"ext_{tag}.parquet").exists():
            continue
        e = pd.read_parquet(DATA / f"ext_{tag}.parquet")
        e = e[e.ch.fillna(False).astype(bool) & e.code.isin(["C", "B", "*B"]) & e.plate_x.notna() & e.sz_top.notna() & e.vy0.notna()]
        final = e.code.eq("C").values; orig = np.where(e.ch_over, ~final, final); ruling = np.where(e.ch_over, ~orig, orig)
        px, pz = G.abs_location(e, "feed")
        assert (G.abs_strike(px, pz, e.sz_top, e.sz_bot) == ruling).mean() >= 0.99, tag
