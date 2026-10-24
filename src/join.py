"""feed x Statcast join of CALLED pitches on the pre-pitch count.

The old join on (game, at-bat, pitch number), screened by description agreement, attached the wrong Statcast pitch when
one source counted an automatic ball/strike as a pitch and the other did not: two successive balls (or strikes) passed
the screen with another pitch's location (94 count inconsistencies in 2026, median 15 in apart). Within a plate
appearance the count only rises, so at most one called pitch has a given pre-pitch count and call type; that key is
unique. Statcast's balls/strikes are pre-pitch and its description is the final call, as is the feed's code (an
overturned pitch agrees in both)."""
import numpy as np, pandas as pd

CALLED_SC = ["called_strike", "ball", "blocked_ball"]
KEY = ["game_pk", "at_bat_number", "pre_b", "pre_s", "stk"]


def count_join(fc, sc, cols):
    """fc: called feed pitches with game_pk, at_bat_number, code, pre_b, pre_s (pre-pitch count). sc: Statcast with
    cols (must include balls, strikes, description, pitch_number). Returns fc with the Statcast cols of the unique
    count-consistent called pitch (balls/strikes = the pre-pitch count), sc_pitch_number, misjoin (no such pitch:
    unlocated) and renumbered (the two sources number the pitch differently)."""
    fc = fc.assign(stk=fc.code.eq("C").values)
    cs = sc[sc.description.isin(CALLED_SC)][list(dict.fromkeys(["game_pk", "at_bat_number"] + cols))].copy()
    cs["stk"] = cs.description.eq("called_strike")
    cs = cs.rename(columns={"pitch_number": "sc_pitch_number", "balls": "pre_b", "strikes": "pre_s"})
    cs = cs[~cs.duplicated(KEY, keep=False)]             # a non-unique key stays unlocated (none in 2026)
    fc = fc.drop(columns=[c for c in cs.columns if c in fc.columns and c not in KEY])
    m = fc.merge(cs, on=KEY, how="left", indicator=True)
    m["balls"], m["strikes"] = m.pre_b.astype(int), m.pre_s.astype(int)
    m["misjoin"] = m._merge.ne("both")
    m["renumbered"] = m.sc_pitch_number.notna() & (m.sc_pitch_number != m.pitch_number)
    return m.drop(columns="_merge")


def pre_count_from_after(fc):
    """feed pre-pitch count of a called pitch from its post-pitch count (balls_after, strikes_after)."""
    isk = fc.code.eq("C").values
    return (np.where(isk, fc.balls_after, fc.balls_after - 1).astype(int),
            np.where(isk, fc.strikes_after - 1, fc.strikes_after).astype(int))
