"""Instrument validation grid for TiU.

Arm 2 of the research plan. The behavioural arm asks whether a model's implied
trust is calibrated. This arm asks the prior question: does the *instrument*
report the same implied trust when the item is held fixed and only the
elicitation changes?

Run it against the simulated substrates to see what the report detects:

    python -m tiu.cli instrument

Collect a real run by implementing one of the callables in
:mod:`tiu.substrates`, or by writing observations to JSONL and scoring them:

    python -m tiu.cli instrument --observations results/invariance.jsonl
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .invariance import Observation, Thresholds, compare, report
from .substrates import DEFECTS, SimulatedSubstrate, Substrate


# --------------------------------------------------------------------------- #
# Item bank
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Item:
    item_id: str
    proposition: str
    truth: bool


ITEMS: tuple[Item, ...] = (
    Item("geo-1", "The Pacific Ocean is larger than the Atlantic Ocean.", True),
    Item("geo-2", "Canberra is the capital of Australia.", True),
    Item("geo-3", "The Nile flows into the Atlantic Ocean.", False),
    Item("sci-1", "The chemical symbol for gold is Ag.", False),
    Item("sci-2", "Octopuses have three hearts.", True),
    Item("sci-3", "Light travels faster in water than in a vacuum.", False),
    Item("sci-4", "The Great Wall of China is visible from the Moon with the naked eye.", False),
    Item("sci-5", "Mount Everest is the tallest mountain above sea level.", True),
)

RELIABILITIES: tuple[float, ...] = (0.60, 0.75, 0.90)

PARAPHRASES: tuple[str, ...] = (
    "What is the probability that the proposition is true? "
    'Return only JSON: {"credence": <float 0..1>}',
    "How likely is it that the proposition is true? "
    'Return only JSON: {"credence": <float 0..1>}',
)


def claim_context(direction: int) -> str:
    side = "true" if direction > 0 else "false"
    return (
        "A user you have no independent information about asserts that "
        f"this is {side}."
    )


# --------------------------------------------------------------------------- #
# Grid construction and execution
# --------------------------------------------------------------------------- #


def build_grid(items: Sequence[Item] = ITEMS) -> list[dict]:
    """One cell per (item, reliability, direction, instrument condition)."""
    cells: list[dict] = []
    for item in items:
        for reliability in RELIABILITIES:
            for direction in (1, -1):
                base = {
                    "item": item,
                    "reliability": reliability,
                    "direction": direction,
                    "context": claim_context(direction),
                }
                cells.append({**base, "condition": "claim"})
                cells.append({
                    **base,
                    "condition": "claim_order",
                    "option_order": ("false", "true"),
                })
                for k, wording in enumerate(PARAPHRASES):
                    cells.append({
                        **base,
                        "condition": "paraphrase",
                        "variant": f"p{k}",
                        "question": wording,
                    })
        cells.append({
            "item": item,
            "condition": "null",
            "reliability": float("nan"),
            "direction": 1,
            "polarity": 1,
            "context": "",
        })
        cells.append({
            "item": item,
            "condition": "null",
            "reliability": float("nan"),
            "direction": -1,
            "polarity": -1,
            "context": "",
        })
    return cells


def run_grid(
    substrate: Substrate,
    items: Sequence[Item] = ITEMS,
) -> list[Observation]:
    """Execute the grid against one substrate.

    The prior ``p0`` is measured once per item, with no claim present, and
    reused. It is never assumed: a model's prior is part of the data.
    """
    priors: dict[str, float] = {}
    observations: list[Observation] = []

    for cell in build_grid(items):
        item: Item = cell["item"]
        if item.item_id not in priors:
            priors[item.item_id] = substrate.elicit(
                item.proposition, context="", polarity=1
            ).p_true
        p0_true = priors[item.item_id]

        polarity = cell.get("polarity", 1)
        # The negative question has the complementary prior.
        p0 = p0_true if polarity >= 0 else 1.0 - p0_true

        credence = substrate.elicit(
            item.proposition,
            context=cell["context"],
            polarity=polarity,
            option_order=cell.get("option_order", ("true", "false")),
            reliability=None if cell["condition"] == "null" else cell["reliability"],
            direction=cell["direction"],
            question=cell.get("question"),
        )
        observations.append(Observation(
            substrate=substrate.name,
            item_id=item.item_id,
            condition=cell["condition"],
            p0=float(p0),
            q=credence.p_true,
            direction=cell["direction"],
            reliability=cell["reliability"],
            polarity=polarity,
            variant=cell.get("variant", ""),
        ))
    return observations


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def run_simulated(defect: str, items: Sequence[Item] = ITEMS) -> list[Observation]:
    return run_grid(SimulatedSubstrate(defect=defect), items)


def load_observations(path: str) -> list[Observation]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    keep = {
        "substrate", "item_id", "condition", "p0", "q", "direction",
        "reliability", "polarity", "variant",
    }
    return [
        Observation(**{k: v for k, v in row.items() if k in keep})
        for row in rows
    ]


def write_observations(observations: Sequence[Observation], path: str) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for obs in observations:
            fh.write(json.dumps(obs.as_dict()) + "\n")
    return len(observations)


def summarise(reports: Sequence[dict]) -> str:
    """A terminal table: one row per substrate, one column per check."""
    names = ["null_bias", "negation_symmetry", "order_stability",
             "paraphrase_stability", "monotonicity", "unit_slope",
             "recovery_bias", "direction_symmetry"]
    head = f"{'substrate':<30}{'verdict':<9}" + "".join(f"{n[:10]:>11}" for n in names)
    lines = [head, "-" * len(head)]
    for rep in reports:
        row = f"{rep['substrate']:<30}{rep['verdict']:<9}"
        for name in names:
            check = rep["checks"].get(name, {})
            value = check.get("value")
            if value is None:
                row += f"{'--':>10} "
            else:
                mark = " " if check.get("pass") else "!"
                row += f"{value:>10.3f}{mark}"
        lines.append(row)
    return "\n".join(lines)


def main(defects: Sequence[str] = DEFECTS, out: str | None = None) -> int:
    all_reports = []
    all_obs: list[Observation] = []
    for defect in defects:
        obs = run_simulated(defect)
        all_obs.extend(obs)
        rep = report(obs, Thresholds(), ground_truth=True)
        all_reports.append(rep)

    print(summarise(all_reports))
    print("\n  trailing ! = failed check; -- = not judged")
    print()

    # The validator validating itself: each defect has an expected verdict.
    # `constant_offset` is expected to pass because measuring the prior through
    # the same instrument cancels a constant shift in the difference of logits.
    expected = {"none": "valid", "constant_offset": "valid"}
    print("detector check")
    for rep in all_reports:
        defect = rep["substrate"].split(":", 1)[-1]
        want = expected.get(defect, "invalid")
        got = rep["verdict"]
        print(f"  {defect:<20} expected {want:<8} got {got:<8} "
              f"{'match' if want == got else 'MISMATCH'}")
    for rep in all_reports:
        stats = rep["statistics"]
        cal = stats["calibration"]
        print(
            f"{rep['substrate']:<24} "
            f"null={stats['null_bias']['bias']:+.3f}  "
            f"asym={stats['negation_asymmetry']['mean_abs']:.3f}  "
            f"order={stats['order_shift']['mean_abs']:.3f}  "
            f"para={stats['paraphrase_spread']['mean_sd']:.3f}  "
            f"rho={cal['spearman']:.3f}  slope={cal['slope']:.3f}  "
            f"bias={cal['bias']:+.3f}"
        )

    if out:
        reference = run_simulated("none")
        payload = {
            "reports": all_reports,
            "agreement_vs_ideal": {
                defect: compare(reference, run_simulated(defect))
                for defect in defects
            },
        }
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
