"""HONZA-133: in a resample, cross-fitting folds are drawn over the original games (all copies share a fold)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")


def test_copies_share_a_fold(monkeypatch):
    import crossfit as CF
    m = pd.DataFrame(dict(game_pk=np.repeat(np.arange(40), 3), orig_game=np.repeat(np.arange(40) % 13, 3), x=0.0))
    sc = pd.DataFrame(dict(game_pk=np.arange(40)))
    seen = []
    monkeypatch.setattr(CF, "engine", lambda s, w=None: tuple(s.game_pk.values))
    def stakes_under(eng, rows):
        seen.append((set(eng), rows))
        return rows.assign(g_signed=0.0, _row=0)
    monkeypatch.setattr(CF, "stakes_under", stakes_under)
    out = CF.crossfit_frame(m, sc, np.random.default_rng(0))
    folds = out.groupby("orig_game").fold.nunique()
    assert (folds == 1).all()
    for f in range(CF.K):                         # the decision engine never contains a copy of a scored game
        out_games, rows = seen[2 * f]; in_games, _ = seen[2 * f + 1]
        assert set(rows.game_pk) == in_games and not (out_games & in_games)
        assert not (set(m[m.game_pk.isin(out_games)].orig_game) & set(rows.orig_game))
