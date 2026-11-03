"""Tests for the TiU estimators. Run with `python -m pytest -q` or `python tests/test_inference.py`."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tiu.inference import (  # noqa: E402
    acceptance_model,
    auc,
    calibration_report,
    ideal_posterior,
    implied_reliability,
    logit,
    persona_discrimination,
    posterior_predictive_reliability,
    trust_regression,
)


def test_inversion_is_exact() -> None:
    rng = np.random.default_rng(0)
    p0 = rng.uniform(0.2, 0.8, 500)
    pi = rng.uniform(0.55, 0.99, 500)
    s = rng.choice([-1.0, 1.0], 500)
    q = ideal_posterior(p0, pi, s)
    recovered = implied_reliability(q, p0, s)
    assert np.allclose(recovered, pi, atol=1e-6)


def test_no_update_means_no_trust() -> None:
    assert abs(float(implied_reliability(0.7, 0.7, 1.0)) - 0.5) < 1e-9
    assert abs(float(implied_reliability(0.7, 0.7, -1.0)) - 0.5) < 1e-9


def test_direction_handling() -> None:
    # User contradicts a strong prior, model follows: high implied trust.
    assert float(implied_reliability(0.4, 0.8, -1.0)) > 0.5
    # Model ignores the contradiction: no trust.
    assert abs(float(implied_reliability(0.8, 0.8, -1.0)) - 0.5) < 1e-9


def test_perfect_calibration_has_zero_error() -> None:
    pi = np.full(200, 0.8)
    y = np.r_[np.ones(160, dtype=int), np.zeros(40, dtype=int)]
    report = calibration_report(pi, y, pi)
    assert report.tce == 0.0
    assert abs(report.excess_brier) < 1e-12


def test_auc_known_value() -> None:
    assert auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert abs(auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) - 0.5) < 1e-12


def test_posterior_predictive_shrinks_to_prior() -> None:
    # 8/8 correct is strong but not certainty; Beta(2,2) keeps it below 1.
    assert posterior_predictive_reliability(8, 8) < 1.0
    assert posterior_predictive_reliability(8, 8) > 0.8
    # With no evidence it returns the prior mean.
    assert abs(posterior_predictive_reliability(0, 0) - 0.5) < 1e-12
    # More evidence moves further from the prior mean.
    assert posterior_predictive_reliability(40, 40) > posterior_predictive_reliability(8, 8)


def test_trust_slope_recovers_one_for_ideal_observer() -> None:
    rng = np.random.default_rng(3)
    n = 4000
    pi = rng.choice([0.55, 0.7, 0.85, 0.95], n)
    prior = rng.uniform(0.35, 0.9, n)
    asserted_true = rng.random(n) < pi
    s = np.where(asserted_true, 1.0, -1.0)
    q = ideal_posterior(prior, pi, s)
    r_hat = implied_reliability(q, prior, s)
    item = np.arange(n)
    contrasts = trust_regression(r_hat, pi, prior, cluster=item, n_boot=100, seed=0)
    slope = contrasts[1]
    assert abs(slope.estimate - 1.0) < 0.02
    assert abs(contrasts[2].estimate) < 0.02  # no prior contamination


def test_acceptance_model_ideal_structure() -> None:
    rng = np.random.default_rng(5)
    n = 20000
    pi = rng.choice([0.55, 0.7, 0.85, 0.95], n)
    prior = rng.uniform(0.35, 0.9, n)
    asserted_true = rng.random(n) < pi
    s = np.where(asserted_true, 1.0, -1.0)
    q = ideal_posterior(prior, pi, s)
    agree = (s * logit(q) > 0).astype(int)
    coef = acceptance_model(agree, s, pi, prior)
    # trust and signed-prior weights should be comparable for an additive observer
    ratio = coef["trust"] / coef["direction_x_prior"]
    assert 0.7 < ratio < 1.3, coef


def test_persona_discrimination() -> None:
    r_true = np.array([0.55] * 50 + [0.70] * 50 + [0.85] * 50 + [0.95] * 50)
    assert persona_discrimination(r_true, r_true) > 0.99
    assert abs(persona_discrimination(np.full(200, 0.8), r_true) - 0.5) < 1e-9


def test_paired_design_is_balanced() -> None:
    from tiu.design import SEED_ITEMS, build_trials

    trials = list(build_trials(SEED_ITEMS[:1], sampling="paired"))
    assert sum(t.claim_true for t in trials) == len(trials) // 2
    natural = list(build_trials(SEED_ITEMS[:1], sampling="naturalistic"))
    assert len(natural) == len(trials) // 2
    assert {t.arm for t in trials} == {"asserted", "induced"}


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
