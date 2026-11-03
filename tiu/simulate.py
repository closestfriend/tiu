"""Simulation study: is the estimator identifiable, and does it separate policies?

We generate synthetic models with known trust policies, push them through the
same recovery pipeline used on real transcripts, and check that (a) implied
reliability recovers the policy, and (b) the metrics order policies the way a
researcher would want.

Data-generating process (this matters): a user of reliability pi asserts the
true proposition with probability pi, otherwise the negation. So the *asserted
proposition* is true with probability pi, which is what makes implied
reliability a forecast of user correctness and makes AUC meaningful. Crossing
truth with reliability instead would fix marginal accuracy at 50% and quietly
destroy the estimand -- one of several traps this harness is built to avoid.

Run: python -m tiu.simulate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .inference import (
    acceptance_model,
    calibration_report,
    ideal_posterior,
    implied_reliability,
    logit,
    sigmoid,
    trust_regression,
)


def policy_ideal(pi: np.ndarray, prior: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return pi


def policy_anchored(pi: np.ndarray, prior: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Treats every user as highly reliable, regardless of evidence."""
    return np.full_like(pi, 0.92)


def policy_compressed(pi: np.ndarray, prior: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Distinguishes users, but by half the correct amount."""
    return 0.5 + 0.5 * (pi - 0.5)


def policy_affect_driven(pi: np.ndarray, prior: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ignores reliability and keys off an uninformative affective cue."""
    return np.where(rng.random(len(pi)) < 0.5, 0.90, 0.55)


def policy_rationalising(pi: np.ndarray, prior: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Infers reliability partly from its own prior: 'I thought so, so you must be right.'

    This is the failure mode the prior-contamination diagnostic exists to catch,
    and it is invisible to every agreement-rate benchmark.
    """
    return sigmoid(logit(pi) + 0.8 * logit(prior))


def run(n_items: int = 500, n_reps: int = 4, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    reliabilities = np.array([0.55, 0.70, 0.85, 0.95])
    policies = {
        "ideal": policy_ideal,
        "anchored": policy_anchored,
        "compressed": policy_compressed,
        "affect_driven": policy_affect_driven,
        "rationalising": policy_rationalising,
    }

    item = np.repeat(np.arange(n_items), len(reliabilities) * n_reps)
    pi = np.tile(np.repeat(reliabilities, n_reps), n_items)
    prior = rng.uniform(0.35, 0.90, size=len(item))
    # user's claim is correct with probability pi
    asserted_true = rng.random(len(item)) < pi
    direction = np.where(asserted_true, 1.0, -1.0)

    out = {}
    for name, fn in policies.items():
        r_used = fn(pi, prior, rng)
        q = ideal_posterior(prior, r_used, direction)
        q = sigmoid(logit(q) + rng.normal(0, 0.35, size=len(q)))  # elicitation noise
        q = np.clip(q, 1e-3, 1 - 1e-3)
        r_hat = implied_reliability(q, prior, direction)
        # accept the asserted proposition
        agree = (direction * logit(q) > 0).astype(int)
        user_correct = asserted_true.astype(int)

        report = calibration_report(r_hat, user_correct, pi)
        contrasts = trust_regression(r_hat, pi, prior, cluster=item, n_boot=300, seed=seed)
        acc = acceptance_model(agree, direction, pi, prior)
        out[name] = {
            **report.as_dict(),
            "recovered_mean_r": float(np.mean(r_hat)),
            "trust_slope": round(contrasts[1].estimate, 3),
            "trust_slope_ci": [round(contrasts[1].lo, 3), round(contrasts[1].hi, 3)],
            "prior_contamination": round(contrasts[2].estimate, 3),
            "prior_contamination_ci": [round(contrasts[2].lo, 3), round(contrasts[2].hi, 3)],
            "acceptance": {k: round(v, 3) for k, v in acc.items()},
        }
    return out


def main() -> None:
    results = run()
    cols = ["tce", "bias", "excess_brier", "auc", "dprime", "criterion",
            "recovered_mean_r", "trust_slope", "prior_contamination"]
    print(f"{'policy':<15}" + "".join(f"{c[:14]:>15}" for c in cols))
    for name, r in results.items():
        print(f"{name:<15}" + "".join(f"{r[c]:>15.3f}" for c in cols))
    print()
    for name, r in results.items():
        print(name, "trust_slope CI", r["trust_slope_ci"],
              "prior_contam CI", r["prior_contamination_ci"])
    print()
    for name, r in results.items():
        print(f"{name:<15} acceptance {r['acceptance']}")

    path = Path(__file__).resolve().parent.parent / "results" / "simulation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    sys.exit(main())
