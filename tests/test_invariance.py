"""Tests for the instrument-validation arm.

These do not test a model. They test that the validator can tell a valid
instrument from each named defect, which is the precondition for trusting it
on a real substrate.
"""

from __future__ import annotations

import json

import pytest

from tiu.inference import ideal_posterior, implied_reliability
from tiu.instrument import (
    ITEMS,
    build_grid,
    load_observations,
    run_grid,
    run_simulated,
    write_observations,
)
from tiu.invariance import (
    Observation,
    Thresholds,
    compare,
    concordance_ccc,
    report,
    spearman,
)
from tiu.substrates import DEFECTS, SimulatedSubstrate


def test_estimator_inverts_the_ideal_observer():
    for p0 in (0.2, 0.5, 0.8):
        for pi in (0.55, 0.7, 0.9):
            for direction in (1, -1):
                q = ideal_posterior(p0, pi, direction)
                assert implied_reliability(q, p0, direction) == pytest.approx(pi, abs=1e-6)


def test_ideal_substrate_is_valid():
    rep = report(run_simulated("none"), ground_truth=True)
    assert rep["verdict"] == "valid"
    assert rep["statistics"]["calibration"]["slope"] == pytest.approx(1.0, abs=1e-6)
    assert rep["statistics"]["calibration"]["bias"] == pytest.approx(0.0, abs=1e-6)


def test_constant_offset_is_cancelled_by_differencing():
    """A bias that hits the prior and the posterior equally is not a defect.

    This is the estimator's one designed-in immunity, so it is a positive
    control: the harness must call it valid.
    """
    rep = report(run_simulated("constant_offset"), ground_truth=True)
    assert rep["verdict"] == "valid"
    assert abs(rep["statistics"]["null_bias"]["bias"]) < 1e-9


@pytest.mark.parametrize(
    "defect, check",
    [
        ("optimistic", "recovery_bias"),
        ("claim_truth_bias", "direction_symmetry"),
        ("asymmetric", "negation_symmetry"),
        ("order", "order_stability"),
        ("insensitive", "monotonicity"),
        ("noisy", "paraphrase_stability"),
    ],
)
def test_each_defect_trips_its_named_check(defect, check):
    rep = report(run_simulated(defect), ground_truth=True)
    assert rep["verdict"] == "invalid"
    assert rep["checks"][check]["pass"] is False


def test_credulity_bias_is_positive():
    rep = report(run_simulated("optimistic"), ground_truth=True)
    assert rep["statistics"]["calibration"]["bias"] > 0.03


def test_direction_gap_separates_truth_bias_from_credulity():
    credulous = report(run_simulated("optimistic"), ground_truth=True)
    truth_biased = report(run_simulated("claim_truth_bias"), ground_truth=True)
    assert credulous["statistics"]["direction_symmetry"]["gap"] < 0.01
    assert truth_biased["statistics"]["direction_symmetry"]["gap"] > 0.03


def test_agreement_between_identical_substrates_is_perfect():
    obs = run_simulated("none")
    cmp = compare(obs, obs)
    assert cmp["ccc"] == pytest.approx(1.0, abs=1e-9)
    assert cmp["bland_altman"]["bias"] == pytest.approx(0.0, abs=1e-9)


def test_ccc_penalises_offset_that_correlation_ignores():
    x = [0.1, 0.2, 0.3, 0.4]
    shifted = [v + 0.2 for v in x]
    assert spearman(x, shifted) == pytest.approx(1.0)
    assert concordance_ccc(x, shifted) < 1.0


def test_observation_rejects_impossible_probability():
    with pytest.raises(ValueError):
        Observation(
            substrate="x", item_id="i", condition="claim", p0=0.5, q=1.4
        )


def test_empty_report_is_unknown_not_valid():
    assert report([])["verdict"] == "unknown"


def test_grid_has_the_expected_shape():
    cells = build_grid()
    per_item = sum(1 for c in cells if c["item"] is ITEMS[0])
    assert per_item == 2 + 3 * 2 * 4  # 2 null + 3 reliabilities x 2 directions x 4 conditions
    assert len(cells) == per_item * len(ITEMS)


def test_observation_roundtrip(tmp_path):
    obs = run_simulated("none")
    path = tmp_path / "obs.jsonl"
    n = write_observations(obs, str(path))
    assert n == len(obs)
    loaded = load_observations(str(path))
    assert len(loaded) == len(obs)
    assert loaded[0].r_hat == pytest.approx(obs[0].r_hat)
    # the JSONL is readable without the library
    first = json.loads(path.read_text().splitlines()[0])
    assert "r_hat" in first


def test_unknown_defect_is_rejected():
    with pytest.raises(ValueError):
        SimulatedSubstrate("does-not-exist")
    assert "none" in DEFECTS


def test_thresholds_are_serialisable():
    assert Thresholds().as_dict()["slope_band"] == [0.8, 1.2]
