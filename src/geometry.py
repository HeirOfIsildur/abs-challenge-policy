"""ABS strike-zone geometry shared by build.py, build_ext.py, geometry_audit.py and tests/test_geometry.py.

ABS rule (2026): the zone is a 2-D rectangle in the plane through the middle of the plate (y = 8.5 in from the back
tip), 17 in wide, between sz_bot = 27 % and sz_top = 53.5 % of the batter's certified height; a pitch is a strike
when any part of the ball (radius 1.45 in) touches it.

Which y-plane a source reports plate_x / plate_z in (results/geometry_audit.md):
  * Statcast 2026 reports the location already in the ABS plane (the trajectory evaluated at y = 17/24 ft; residual
    SD 0.035 in). The pre-geometry-fix pipeline treated it as front-of-plate and projected again, lowering every pitch
    by 0.98 in on average; that double projection was the "ABS bottom 1.02 in / top 0.82 in below Statcast" offset.
  * The Stats API live feed (pX / pZ; MLB 2026, Triple-A 2025, spring and postseason 2026) reports the front of the
    plate (y = 17/12 ft), so feed coordinates are projected to the midpoint.
Raw-coordinate rule on Statcast 2026 reproduces 99.90 % of the 10,551 challenge rulings (99.65 % with the square-corner
distance used in earlier versions); the projected one about 85 %."""
import numpy as np

PLATE_HALF_WIDTH_FT = 17 / 24
BALL_RADIUS_FT = 1.45 / 12
Y_MID_FT = 17 / 24            # ABS evaluation plane
Y_FRONT_FT = 17 / 12          # front edge of the plate
Y0_FT = 50.0                  # Statcast / Stats API trajectory reference (vx0, vy0, vz0 given at y = 50 ft)
SZ_TOP_FRAC, SZ_BOT_FRAC = 0.535, 0.27

# THE convention: y-plane (ft) in which each source reports plate_x / plate_z. See results/geometry_audit.md.
SOURCE_PLANE_Y_FT = {
    "statcast2026": Y_MID_FT,     # already at the ABS plane -> no projection (geometry fix)
    "feed": Y_FRONT_FT,           # Stats API pX / pZ at the front of the plate -> project to the midpoint
}


def time_to(vy0, ay, y):
    """time (s) from y = 50 ft to y (ball travels toward -y)."""
    a = 0.5 * ay; b = vy0; c = Y0_FT - y
    return (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)


def project(plate_x, plate_z, vx0, vy0, vz0, ax, ay, az, y_from, y_to=Y_MID_FT):
    """move a reported location from plane y_from to plane y_to along the constant-acceleration trajectory."""
    t1, t2 = time_to(vy0, ay, y_from), time_to(vy0, ay, y_to)
    dx = vx0 * (t2 - t1) + 0.5 * ax * (t2 ** 2 - t1 ** 2)
    dz = vz0 * (t2 - t1) + 0.5 * az * (t2 ** 2 - t1 ** 2)
    return plate_x + dx, plate_z + dz


def abs_location(df, source):
    """(px, pz) in the ABS plane for a frame with plate_x/plate_z and the 9-parameter trajectory."""
    y = SOURCE_PLANE_Y_FT[source]
    if y == Y_MID_FT:
        return df.plate_x.values.astype(float), df.plate_z.values.astype(float)
    px, pz = project(df.plate_x.values, df.plate_z.values, df.vx0.values, df.vy0.values, df.vz0.values,
                     df.ax.values, df.ay.values, df.az.values, y)
    return np.asarray(px, float), np.asarray(pz, float)


def signed_distance(px, pz, sz_top, sz_bot, R=BALL_RADIUS_FT, hw=PLATE_HALF_WIDTH_FT):
    """signed distance (in) from the ball's edge to the zone rectangle (+ = the ball touches it = ABS strike) and the
    nearest edge ('vert' = top/bottom line, 'horiz' = a side). The ball is round: R + (signed distance of the ball
    centre to the rectangle), so the strike region is the rectangle grown by R with rounded corners.
    the old version grew each edge by R with square corners, which counted a ball
    diagonally off a corner as touching it (353 called pitches relabelled; 26 of 37 ruling mismatches resolved)."""
    px, pz, sz_top, sz_bot = (np.asarray(v, float) for v in (px, pz, sz_top, sz_bot))
    dx = hw - np.abs(px)
    dz = np.minimum(pz - sz_bot, sz_top - pz)
    d0 = np.where((dx >= 0) & (dz >= 0), np.minimum(dx, dz), -np.hypot(np.minimum(dx, 0), np.minimum(dz, 0)))
    return 12 * (R + d0), np.where(dz < dx, "vert", "horiz")


def abs_strike(px, pz, sz_top, sz_bot, **kw):
    return signed_distance(px, pz, sz_top, sz_bot, **kw)[0] > 0
