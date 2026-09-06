"""Figure generation. These are cheap smoke checks plus one real regression."""
import random

import pytest

from pmbench.baselines import Row
from pmbench.report import _label_ends, build_all


class FakeAx:
    def __init__(self, ylim=(0.0, 1.0)):
        self._ylim = ylim
        self.calls = []

    def get_ylim(self):
        return self._ylim

    def annotate(self, text, xy=None, **kw):
        self.calls.append((text, xy[1]))


def test_label_placement_terminates_on_adversarial_floats():
    """Regression: the de-collision pass was a `while target - placed[-1] <
    min_gap` loop. `placed[-1] + min_gap - placed[-1]` can round to fractionally
    less than min_gap, and the loop then never exits — the whole report hung."""
    rng = random.Random(0)
    entries = [(rng.random(), f"s{i}", "#000000") for i in range(40)]
    ax = FakeAx()
    _label_ends(ax, entries, min_gap=0.05)     # must return, not spin
    assert len(ax.calls) == 40


def test_labels_are_separated_and_ordered():
    ax = FakeAx()
    _label_ends(ax, [(0.50, "a", "#000"), (0.505, "b", "#111"), (0.51, "c", "#222")],
                min_gap=0.05)
    ys = [y for _, y in ax.calls]
    assert all(b - a >= 0.049 for a, b in zip(ys, ys[1:]))


def test_identical_positions_do_not_collide():
    ax = FakeAx()
    _label_ends(ax, [(0.3, "a", "#000"), (0.3, "b", "#111")], min_gap=0.05)
    ys = sorted(y for _, y in ax.calls)
    assert ys[1] - ys[0] >= 0.049


def rows(n=60, seed=0):
    rng = random.Random(seed)
    cats = ["Politics", "Crypto", "Sports"]
    out = []
    for i in range(n):
        p = rng.random()
        out.append(Row(f"m{i}", cats[i % 3], p, rng.random(),
                       1 if rng.random() < p else 0,
                       horizon_days=rng.choice([3.0, 20.0, 60.0, 200.0]),
                       ensemble_sd=rng.random() * 0.2,
                       config_name="cfg-a" if i % 2 else "cfg-b"))
    return out


def test_build_all_writes_every_figure_and_its_table(tmp_path):
    written = build_all(rows(), tmp_path)
    names = {p.name for p in written}
    for stem in ("reliability", "baselines", "abstention", "horizon", "category"):
        assert f"{stem}.png" in names, stem
        assert f"{stem}.csv" in names, stem
    assert all(p.stat().st_size > 0 for p in written)


def test_build_all_survives_a_single_config(tmp_path):
    rs = rows()
    for r in rs:
        r.config_name = "only"
    assert build_all(rs, tmp_path)


def test_build_all_survives_missing_market_prices(tmp_path):
    rs = rows()
    for r in rs:
        r.market_price = None
    assert build_all(rs, tmp_path)
