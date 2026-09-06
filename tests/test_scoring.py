import math

import numpy as np
import pytest

from pmbench.scoring import (
    apply_platt, brier, brier_decomposition, expected_calibration_error,
    fit_platt, fit_temperature, log_loss, overconfidence, reliability_bins, summarise,
)


def test_brier_and_log_loss_known_values():
    assert brier(1.0, 1) == 0.0
    assert brier(0.0, 1) == 1.0
    assert brier(0.5, 1) == pytest.approx(0.25)
    assert log_loss(1.0, 1) == pytest.approx(0.0, abs=1e-5)
    assert log_loss(0.5, 0) == pytest.approx(math.log(2))


def test_log_loss_clips_instead_of_exploding():
    assert math.isfinite(log_loss(0.0, 1))
    assert log_loss(0.0, 1) > 10


def test_always_half_scores_exactly_quarter():
    ys = [1, 0, 1, 1, 0]
    assert sum(brier(0.5, y) for y in ys) / len(ys) == pytest.approx(0.25)


def _calibrated(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, n)
    y = (rng.uniform(size=n) < p).astype(int)
    return list(p), list(y)


def test_calibrated_forecaster_has_small_ece_and_zero_overconfidence():
    p, y = _calibrated()
    assert expected_calibration_error(p, y) < 0.03
    assert abs(overconfidence(p, y)) < 0.03


def test_overconfident_forecaster_is_detected_and_platt_says_so():
    """Push a calibrated forecaster's probabilities toward 0 and 1 and check
    that both the overconfidence statistic and the Platt slope notice."""
    p, y = _calibrated()
    sharp = [min(max(0.5 + (x - 0.5) * 1.9, 1e-4), 1 - 1e-4) for x in p]
    assert overconfidence(sharp, y) > 0.05
    a, _ = fit_platt(sharp, y)
    assert a < 1.0                       # too extreme -> slope shrinks them
    assert fit_temperature(sharp, y) > 1.0


def test_platt_correction_reduces_brier_on_miscalibrated_data():
    p, y = _calibrated(seed=3)
    sharp = [min(max(0.5 + (x - 0.5) * 1.9, 1e-4), 1 - 1e-4) for x in p]
    a, b = fit_platt(sharp, y)
    before = sum(brier(x, t) for x, t in zip(sharp, y)) / len(y)
    after = sum(brier(apply_platt(x, a, b), t) for x, t in zip(sharp, y)) / len(y)
    assert after < before


def test_murphy_decomposition_reconstructs_brier():
    p, y = _calibrated(seed=7)
    rel, res, unc = brier_decomposition(p, y, n_bins=10)
    direct = sum(brier(a, b) for a, b in zip(p, y)) / len(p)
    # Exact only in the limit of the binning; 10 bins gets close.
    assert rel - res + unc == pytest.approx(direct, abs=0.01)


def test_reliability_bins_cover_every_prediction():
    p, y = _calibrated(seed=11)
    assert sum(b.n for b in reliability_bins(p, y, 10)) == len(p)


def test_reliability_bins_handle_endpoints():
    bins = reliability_bins([0.0, 1.0], [0, 1], 10)
    assert sum(b.n for b in bins) == 2


def test_summarise_on_empty_is_nan_not_crash():
    s = summarise([], [])
    assert s.n == 0 and math.isnan(s.brier)
