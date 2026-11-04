"""Command line entry points.

    python -m tiu.cli build-design --out data/design.jsonl
    python -m tiu.cli analyze --results results/transcripts.jsonl
    python -m tiu.cli simulate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .design import SEED_ITEMS, build_trials, write_jsonl
from .inference import (
    calibration_report,
    implied_reliability,
    persona_discrimination,
    trust_regression,
)


def cmd_build_design(args: argparse.Namespace) -> None:
    trials = build_trials(SEED_ITEMS, history_n=args.history, sampling=args.sampling)
    n = write_jsonl(trials, args.out)
    print(f"wrote {n} trials to {args.out}")
    print(f"  items={len(SEED_ITEMS)}  arms=asserted,induced  "
          f"families=7  reliabilities=4  stakes=2  sampling={args.sampling}")


def cmd_analyze(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in Path(args.results).read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit("no rows")

    r_hat = np.array([
        implied_reliability(r["posterior"], r["prior"], 1.0 if r["claim_true"] else -1.0)
        for r in rows
    ])
    r_true = np.array([r["target_reliability"] for r in rows])
    correct = np.array([bool(r["claim_true"]) for r in rows], dtype=int)
    prior = np.array([r["prior"] for r in rows])
    cluster = np.array([r["item_id"] for r in rows])

    report = calibration_report(r_hat, correct, r_true)
    print("## Calibration")
    for key, value in report.as_dict().items():
        print(f"  {key:>14}: {value:.4f}" if isinstance(value, float) else f"  {key:>14}: {value}")
    print(f"  {'auc_persona':>14}: {persona_discrimination(r_hat, r_true):.4f}")
    print("  note: auc/dprime/criterion require naturalistic sampling; "
          "auc_persona is valid for matched pairs.")

    print("\n## Trust regression (ideal: slope=1, contamination=0)")
    for c in trust_regression(r_hat, r_true, prior, cluster=cluster, n_boot=args.boot):
        flag = "*" if c.significant else " "
        print(f"  {c.name:<22} {c.estimate:+.3f}  [{c.lo:+.3f}, {c.hi:+.3f}] {flag}")

    print("\n## By cue family")
    families = sorted({r["cue_family"] for r in rows})
    for family in families:
        idx = [i for i, r in enumerate(rows) if r["cue_family"] == family]
        sub = calibration_report(r_hat[idx], correct[idx], r_true[idx])
        print(f"  {family:<12} TCE={sub.tce:.3f}  bias={sub.bias:+.3f}  AUC={sub.auc:.3f}")


def cmd_instrument(args: argparse.Namespace) -> None:
    from .instrument import load_observations, main as inst_main, summarise
    from .invariance import report
    from .substrates import DEFECTS

    if args.observations:
        rep = report(load_observations(args.observations), ground_truth=False)
        print(summarise([rep]))
        return

    defects = DEFECTS if args.defect == "all" else (args.defect,)
    inst_main(defects, args.out)


def cmd_simulate(args: argparse.Namespace) -> None:
    from .simulate import main as sim_main

    sim_main()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tiu")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-design", help="emit the factorial design as JSONL")
    p.add_argument("--out", default="data/design.jsonl")
    p.add_argument("--history", type=int, default=8, help="verified prior claims in the induced arm")
    p.add_argument("--sampling", choices=["paired", "naturalistic"], default="paired",
                   help="paired = both polarities per item; naturalistic = polarity ~ Bernoulli(pi)")
    p.set_defaults(func=cmd_build_design)

    p = sub.add_parser("analyze", help="score transcripts")
    p.add_argument("--results", required=True)
    p.add_argument("--boot", type=int, default=2000)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("simulate", help="run the estimator validation study")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser(
        "instrument",
        help="validate elicitation substrates (null bias, symmetry, order, paraphrase)",
    )
    p.add_argument("--defect", default="all",
                   help="simulated defect to run, or 'all'")
    p.add_argument("--observations", default=None,
                   help="score a JSONL of collected observations instead")
    p.add_argument("--out", default="results/invariance.json")
    p.set_defaults(func=cmd_instrument)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
