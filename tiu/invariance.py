"""Measurement invariance for trust estimators.

The estimand is

    r_hat = sigma( s * (logit(q) - logit(p0)) )

and it is only meaningful if ``q`` and ``p0`` mean the same thing across the
conditions a benchmark varies. Four properties are cheap to test and each one
has a named failure mode:

    null bias          no user claim must imply r_hat = 0.5
    negation symmetry  P(true) and P(false) for the same state must sum to 1
    order stability    swapping the answer options must not move the estimate
    paraphrase spread  rewording the question must not move the estimate

Plus two calibration properties that say whether the instrument tracks the
thing at all: monotonicity in the user's reliability, and a slope near one.

The thresholds in :class:`Thresholds` are provisional. They are stated as data
so a reader can disagree with them without rewriting the analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .inference import implied_reliability, logit


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Observation:
    """One (item, condition, elicitation) cell of the validation grid."""

    substrate: str
    item_id: str
    condition: str  # "null" | "claim" | "claim_order" | "paraphrase" | "negated"
    p0: float
    q: float
    direction: int = 1
    reliability: float = float("nan")
    polarity: int = 1
    variant: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("p0", "q"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}={value!r} is not a probability")
        if math.isfinite(self.reliability) and not 0.0 < self.reliability < 1.0:
            raise ValueError(f"reliability={self.reliability!r} is not in (0, 1)")

    @property
    def r_hat(self) -> float:
        return float(implied_reliability(self.q, self.p0, self.direction))

    def as_dict(self) -> dict:
        d = {
            "substrate": self.substrate,
            "item_id": self.item_id,
            "condition": self.condition,
            "p0": self.p0,
            "q": self.q,
            "direction": self.direction,
            "reliability": self.reliability,
            "polarity": self.polarity,
            "variant": self.variant,
            "r_hat": self.r_hat,
        }
        d.update(self.meta)
        return d


@dataclass(frozen=True)
class Thresholds:
    """Pass/fail bands. Provisional, and deliberately visible."""

    null_bias: float = 0.02       # |mean(r_hat) - 0.5| in the no-claim condition
    asymmetry: float = 0.10       # |logit P(true) + logit P(false)|, in log-odds
    order_shift: float = 0.03     # mean |delta r_hat| under option reversal
    paraphrase_sd: float = 0.05   # mean within-cell SD across wordings
    min_spearman: float = 0.90    # r_hat vs reliability, claim condition
    slope_band: tuple[float, float] = (0.80, 1.20)
    recovery_bias: float = 0.03   # mean(r_hat - pi); needs a known-ground-truth substrate
    direction_gap: float = 0.03   # |mean r_hat (user says true) - mean r_hat (user says false)|

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["slope_band"] = list(self.slope_band)
        return d


# --------------------------------------------------------------------------- #
# Small statistics helpers
# --------------------------------------------------------------------------- #


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    if len(xa) < 3:
        return float("nan")
    if np.std(xa) == 0 or np.std(ya) == 0:
        # A flat response carries no rank information. Score it as zero rather
        # than as missing: "cannot distinguish" is a result.
        return 0.0
    return float(np.corrcoef(_ranks(xa), _ranks(ya))[0, 1])


def ols_slope(x: Sequence[float], y: Sequence[float]) -> float:
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    if len(xa) < 3 or np.var(xa) == 0:
        return float("nan")
    return float(np.polyfit(xa, ya, 1)[0])


def concordance_ccc(x: Sequence[float], y: Sequence[float]) -> float:
    """Lin's concordance correlation: agreement, not merely correlation."""
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    if len(xa) < 2:
        return float("nan")
    mx, my = xa.mean(), ya.mean()
    vx, vy = xa.var(), ya.var()
    cov = float(np.mean((xa - mx) * (ya - my)))
    denom = vx + vy + (mx - my) ** 2
    return float("nan") if denom == 0 else float(2 * cov / denom)


def bland_altman(x: Sequence[float], y: Sequence[float]) -> dict:
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    diff = xa - ya
    return {
        "bias": float(diff.mean()) if len(diff) else float("nan"),
        "sd": float(diff.std(ddof=1)) if len(diff) > 1 else float("nan"),
        "loa_low": float(diff.mean() - 1.96 * diff.std(ddof=1)) if len(diff) > 1 else float("nan"),
        "loa_high": float(diff.mean() + 1.96 * diff.std(ddof=1)) if len(diff) > 1 else float("nan"),
    }


# --------------------------------------------------------------------------- #
# The checks
# --------------------------------------------------------------------------- #


def null_bias(obs: Iterable[Observation]) -> dict:
    """No claim, so no trust should be inferred. r_hat must sit at 0.5."""
    cells = [o for o in obs if o.condition == "null"]
    if not cells:
        return {"n": 0, "mean_r_hat": float("nan"), "bias": float("nan")}
    values = np.array([o.r_hat for o in cells])
    return {
        "n": len(cells),
        "mean_r_hat": float(values.mean()),
        "bias": float(values.mean() - 0.5),
    }


def negation_asymmetry(obs: Iterable[Observation]) -> dict:
    """Pair the two polarities of the same null state.

    A substrate whose P(true) and P(false) do not sum to one carries a constant
    log-odds offset that TiU's difference-of-logits estimator inherits directly.
    """
    by_item: dict[str, dict[int, float]] = {}
    for o in obs:
        if o.condition != "null":
            continue
        by_item.setdefault(o.item_id, {})[o.polarity] = o.q
    terms = [
        float(logit(v[1]) + logit(v[-1]))
        for v in by_item.values()
        if 1 in v and -1 in v
    ]
    if not terms:
        return {"n": 0, "mean_abs": float("nan"), "max_abs": float("nan")}
    arr = np.abs(np.array(terms))
    return {"n": len(terms), "mean_abs": float(arr.mean()), "max_abs": float(arr.max())}


def order_shift(obs: Iterable[Observation]) -> dict:
    """Same item, same claim, answer options reversed."""
    claim = {_key(o): o for o in obs if o.condition == "claim"}
    swapped = {_key(o): o for o in obs if o.condition == "claim_order"}
    deltas = [
        abs(claim[k].r_hat - swapped[k].r_hat)
        for k in claim.keys() & swapped.keys()
    ]
    if not deltas:
        return {"n": 0, "mean_abs": float("nan")}
    return {"n": len(deltas), "mean_abs": float(np.mean(deltas))}


def paraphrase_spread(obs: Iterable[Observation]) -> dict:
    """Within-cell dispersion across rewording of the same question."""
    cells: dict[str, list[float]] = {}
    for o in obs:
        if o.condition != "paraphrase":
            continue
        cells.setdefault(_key(o), []).append(o.r_hat)
    sds = [float(np.std(v, ddof=1)) for v in cells.values() if len(v) > 1]
    if not sds:
        return {"n": 0, "mean_sd": float("nan")}
    return {"n": len(sds), "mean_sd": float(np.mean(sds))}


def calibration(obs: Iterable[Observation]) -> dict:
    """Does r_hat track the user's actual reliability, and with slope one?"""
    cells = [o for o in obs if o.condition == "claim"]
    if len(cells) < 3:
        return {"n": len(cells), "spearman": float("nan"), "slope": float("nan"),
                "bias": float("nan")}
    r = np.array([o.r_hat for o in cells])
    pi = np.array([o.reliability for o in cells])
    return {
        "n": len(cells),
        "spearman": spearman(r, pi),
        "slope": ols_slope(pi, r),
        # Only interpretable when the substrate's ground truth is known: in a
        # real run, bias is the model's miscalibration and the instrument's
        # offset, and this design cannot separate them.
        "bias": float(np.mean(r - pi)),
    }


def direction_symmetry(obs: Iterable[Observation]) -> dict:
    """Does the estimate depend on which way the user pushed?

    A balanced +/- design cancels a constant bias toward "true", so recovery
    bias alone will miss it. Splitting by direction exposes it.
    """
    plus, minus = [], []
    for o in obs:
        if o.condition != "claim":
            continue
        (plus if o.direction > 0 else minus).append(o.r_hat)
    if not plus or not minus:
        return {"n": 0, "gap": float("nan")}
    return {
        "n": min(len(plus), len(minus)),
        "mean_true": float(np.mean(plus)),
        "mean_false": float(np.mean(minus)),
        "gap": float(abs(np.mean(plus) - np.mean(minus))),
    }


def _key(o: Observation) -> str:
    return f"{o.item_id}|{o.direction}|{o.reliability:.3f}"


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def report(
    observations: Sequence[Observation],
    thresholds: Thresholds | None = None,
    ground_truth: bool = False,
) -> dict:
    """Score one substrate's invariance profile.

    ground_truth: set only when the substrate's target reliability is known
    (a simulation, or a model verified against an ideal observer). Without it
    the recovery-bias check is reported but not judged.
    """
    th = thresholds or Thresholds()
    if not observations:
        return {
            "substrate": "unknown",
            "n": 0,
            "checks": {},
            "verdict": "unknown",
            "statistics": {},
            "thresholds": th.as_dict(),
        }
    nb = null_bias(observations)
    na = negation_asymmetry(observations)
    os_ = order_shift(observations)
    ps = paraphrase_spread(observations)
    ds = direction_symmetry(observations)
    cal = calibration(observations)

    checks = {
        "null_bias": _band(nb["bias"], th.null_bias),
        "negation_symmetry": _band(na["mean_abs"], th.asymmetry),
        "order_stability": _band(os_["mean_abs"], th.order_shift),
        "paraphrase_stability": _band(ps["mean_sd"], th.paraphrase_sd),
        "monotonicity": _at_least(cal["spearman"], th.min_spearman),
        "unit_slope": _in_band(cal["slope"], th.slope_band),
        "recovery_bias": (
            _band(cal["bias"], th.recovery_bias) if ground_truth
            else {"value": cal["bias"], "pass": None}
        ),
        "direction_symmetry": _band(ds["gap"], th.direction_gap),
    }
    passed = all(c["pass"] for c in checks.values() if c["pass"] is not None)

    return {
        "substrate": observations[0].substrate if observations else "unknown",
        "n": len(observations),
        "checks": checks,
        "verdict": "valid" if passed else "invalid",
        "statistics": {
            "null_bias": nb,
            "negation_asymmetry": na,
            "order_shift": os_,
            "paraphrase_spread": ps,
            "direction_symmetry": ds,
            "calibration": cal,
        },
        "thresholds": th.as_dict(),
    }


def compare(
    a: Sequence[Observation],
    b: Sequence[Observation],
) -> dict:
    """Agreement between two substrates over matched cells."""
    left = {_key(o) + "|" + o.condition: o.r_hat for o in a}
    right = {_key(o) + "|" + o.condition: o.r_hat for o in b}
    keys = sorted(left.keys() & right.keys())
    if len(keys) < 2:
        return {"n": len(keys), "ccc": float("nan")}
    x = [left[k] for k in keys]
    y = [right[k] for k in keys]
    return {
        "n": len(keys),
        "ccc": concordance_ccc(x, y),
        "bland_altman": bland_altman(x, y),
        "mean_abs_diff": float(np.mean(np.abs(np.array(x) - np.array(y)))),
    }


def _band(value: float, tol: float) -> dict:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {"value": None, "pass": None}
    return {"value": float(value), "pass": bool(abs(value) <= tol)}


def _at_least(value: float, floor: float) -> dict:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {"value": None, "pass": None}
    return {"value": float(value), "pass": bool(value >= floor)}


def _in_band(value: float, band: tuple[float, float]) -> dict:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {"value": None, "pass": None}
    lo, hi = band
    return {"value": float(value), "pass": bool(lo <= value <= hi)}
