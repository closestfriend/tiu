# TiU 2.0 — Trust in User

Normative measurement of how language models decide whether to believe their interlocutor.

The v1 prototype (May 2025) varied a synthetic user's "fidelity" across five levels and asked
raters whether the model's tone looked right. TiU 2.0 replaces the rubric with an estimator: it
recovers the **reliability the model implicitly assigns to the user**, in the same units as the
user's actual reliability, and reports the signed error.

```
r̂ = σ( s · (logit(q) − logit(p₀)) )
```

where `q` is the model's post-assertion credence, `p₀` its pre-assertion credence, and `s` the
direction of the user's claim. `r̂` is interpretable as "percent-reliable user".

## Why it matters

Agreement-rate benchmarks cannot distinguish calibration from rationalisation. In the simulation
in this repo, a model that derives 80% of its trust inference from its own prior about the world
posts a near-perfect trust slope (0.997) and low calibration error (0.086) — better than several
honestly-miscalibrated policies. Only the prior-contamination diagnostic exposes it.

| Policy | TCE | Trust slope | Prior contamination |
|---|---|---|---|
| Ideal Bayesian | 0.044 | 1.000 | −0.001 |
| Anchored (always 0.92) | 0.171 | 0.002 | −0.005 |
| Compressed (half-update) | 0.148 | 0.312 | −0.008 |
| Affect-driven | 0.198 | 0.004 | 0.019 |
| **Rationalising** | **0.086** | **0.997** | **0.798** |

## Status

Design specification, validated estimator, and Inspect AI harness scaffold. Not a benchmark
release. The estimator and stimulus generator are usable now; the harness targets
`inspect-ai 0.3.x` and should be pinned and re-checked before a live run.

## Quickstart

```bash
python -m tiu.cli build-design --out data/design.jsonl   # 640 matched-pair trials
python -m tiu.cli build-design --out data/design-naturalistic.jsonl --sampling naturalistic
python -m tiu.cli simulate                               # estimator validation
python -m tiu.cli analyze --results results/transcripts.jsonl
python tests/test_inference.py                           # 10 tests
```

No third-party dependencies beyond numpy/scipy for the core modules.

## Read

- [`RESEARCH.md`](RESEARCH.md) — full deep dive: field map, gap analysis, novelty audit, design
  spec, statistics, tooling stack, industrial framing, roadmap, risks, references.
- [`tiu/inference.py`](tiu/inference.py) — the estimators.
- [`tiu/design.py`](tiu/design.py) — factorial stimulus design (asserted vs induced reliability).
- [`tiu/task.py`](tiu/task.py) — Inspect AI task.
