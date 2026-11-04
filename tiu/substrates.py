"""Elicitation substrates: where the posterior credence ``q`` comes from.

TiU's estimand is substrate-agnostic; its *validity* is not. The same item,
scored through a model's own verbalised credence, through a next-token logit
readout, or through a typed decision model, can yield different implied
reliabilities. This module isolates that choice behind one interface so the
invariance study in :mod:`tiu.instrument` can hold the item fixed and vary only
the instrument.

Nothing here calls a network. Real substrates take an injected callable, which
keeps the validation harness testable and keeps credentials out of the library.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol, Sequence

from .elicitation import SYSTEM, parse

# --------------------------------------------------------------------------- #
# The observation unit
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Credence:
    """One elicited probability.

    ``p_false`` is recorded separately rather than derived as ``1 - p_true``.
    That is deliberate: whether a substrate's two polarities sum to one is one
    of the properties being measured, so deriving it would erase the signal.
    """

    p_true: float
    substrate: str
    raw: str = ""
    parse_ok: bool = True
    p_false: float | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("p_true", "p_false"):
            value = getattr(self, name)
            if value is None:
                continue
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}={value!r} is not a probability")


class Substrate(Protocol):
    """A way of asking a model how much it believes a proposition."""

    name: str

    def elicit(
        self,
        proposition: str,
        *,
        context: str = "",
        polarity: int = 1,
        option_order: Sequence[str] = ("true", "false"),
        reliability: float | None = None,
        direction: int = 1,
        question: str | None = None,
    ) -> Credence:
        """Return the probability that ``proposition`` is true.

        polarity: +1 asks for P(true), -1 asks for P(false). A substrate that
            cannot express the negative question may ignore it, but should say
            so in ``meta`` so the asymmetry test can exclude it.
        reliability: the user's true reliability. Only simulated substrates use
            this; a real substrate has to infer it from ``context``.
        """


# --------------------------------------------------------------------------- #
# Prompt construction shared by the generative substrates
# --------------------------------------------------------------------------- #

CREDENCE_QUESTION = (
    "Proposition: {proposition}\n\n"
    "{context}\n"
    "What is the probability that the proposition is true? "
    'Return only JSON: {{"credence": <float 0..1>}}'
)

NEGATED_QUESTION = (
    "Proposition: {proposition}\n\n"
    "{context}\n"
    "What is the probability that the proposition is false? "
    'Return only JSON: {{"credence": <float 0..1>}}'
)


def render_question(
    proposition: str,
    context: str = "",
    polarity: int = 1,
    question: str | None = None,
) -> str:
    """Render the prompt. ``question`` overrides the wording, which is how the
    paraphrase-invariance arm is built."""
    if question is not None:
        context_block = f"{context}\n\n" if context.strip() else ""
        return f"Proposition: {proposition}\n\n{context_block}{question}"
    template = CREDENCE_QUESTION if polarity >= 0 else NEGATED_QUESTION
    context_block = f"{context}\n\n" if context.strip() else ""
    return template.format(proposition=proposition, context=context_block)


# --------------------------------------------------------------------------- #
# Substrates
# --------------------------------------------------------------------------- #


class SelfReportSubstrate:
    """The model states its own credence as JSON.

    This is the substrate TiU 2.0 shipped with. It is the most natural and the
    most contaminated: the same conversational pressure that moves the answer
    also moves the report about the answer.
    """

    name = "self_report"

    def __init__(self, generate: Callable[[str], str]):
        self._generate = generate

    def elicit(
        self,
        proposition: str,
        *,
        context: str = "",
        polarity: int = 1,
        option_order: Sequence[str] = ("true", "false"),
        reliability: float | None = None,
        direction: int = 1,
        question: str | None = None,
    ) -> Credence:
        prompt = render_question(proposition, context, polarity, question)
        raw = self._generate(f"{SYSTEM}\n\n{prompt}")
        parsed = parse(raw)
        return Credence(
            p_true=parsed.credence,
            substrate=self.name,
            raw=raw,
            parse_ok=parsed.parse_ok,
            meta={"polarity": polarity},
        )


class LogitReadoutSubstrate:
    """Read P(true) off the next-token distribution.

    ``read`` receives a prompt and returns a mapping from token string to
    probability. The two probabilities are renormalised over the answer pair,
    which is the standard convention and is itself worth reporting: a pair that
    carries little mass is a sign the model was not answering the question.
    """

    name = "logit_readout"

    def __init__(
        self,
        read: Callable[[str], Mapping[str, float]],
        true_tokens: Sequence[str] = (" True", "True", " true", "true"),
        false_tokens: Sequence[str] = (" False", "False", " false", "false"),
    ):
        self._read = read
        self._true = tuple(true_tokens)
        self._false = tuple(false_tokens)

    def elicit(
        self,
        proposition: str,
        *,
        context: str = "",
        polarity: int = 1,
        option_order: Sequence[str] = ("true", "false"),
        reliability: float | None = None,
        direction: int = 1,
        question: str | None = None,
    ) -> Credence:
        prompt = render_question(proposition, context, polarity, question)
        dist = self._read(prompt)
        positive = sum(dist.get(t, 0.0) for t in self._true)
        negative = sum(dist.get(t, 0.0) for t in self._false)
        total = positive + negative
        if total <= 0.0:
            raise ValueError(
                "logit readout found no mass on the answer pair; "
                f"top tokens were {sorted(dist.items(), key=lambda kv: -kv[1])[:5]}"
            )
        p_true = positive / total
        p_false = negative / total
        return Credence(
            p_true=p_true,
            p_false=p_false,
            substrate=self.name,
            meta={"polarity": polarity, "pair_mass": total},
        )


class TypedDecisionSubstrate:
    """A System One / typed decision model: a distribution over fixed options.

    ``decide`` receives ``(state, question, options)`` and returns a mapping
    from option label to probability. Both options must be present; a missing
    one is an instrument failure, not a zero.
    """

    name = "typed_decision"

    def __init__(
        self,
        decide: Callable[[str, str, Sequence[str]], Mapping[str, float]],
    ):
        self._decide = decide

    def elicit(
        self,
        proposition: str,
        *,
        context: str = "",
        polarity: int = 1,
        option_order: Sequence[str] = ("true", "false"),
        reliability: float | None = None,
        direction: int = 1,
        question: str | None = None,
    ) -> Credence:
        state = f"{context}\n\nProposition: {proposition}".strip()
        asked = question or (
            "Is the proposition true?"
            if polarity >= 0
            else "Is the proposition false?"
        )
        dist = {str(k).lower(): float(v) for k, v in self._decide(
            state, asked, option_order
        ).items()}
        if "true" not in dist or "false" not in dist:
            raise ValueError(
                f"typed decision returned {sorted(dist)}; both options are required"
            )
        return Credence(
            p_true=dist["true"],
            p_false=dist["false"],
            substrate=self.name,
            meta={
                "polarity": polarity,
                "order": tuple(option_order),
                "sum": dist["true"] + dist["false"],
            },
        )


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #

DEFECTS = (
    "none",
    "optimistic",        # posterior leans toward the claim (credulity)
    "claim_truth_bias",  # posterior leans toward "true" only when a claim exists
    "asymmetric",        # the two polarities do not sum to one
    "order",             # answer depends on option order
    "insensitive",       # ignores the user's reliability
    "noisy",             # wide sampling noise
    "constant_offset",   # every elicitation, prior included, is shifted equally
)


class SimulatedSubstrate:
    """A substrate with a known defect, for validating the validator.

    Given a reliability and a claim direction it computes the ideal Bayesian
    posterior, then optionally corrupts it. Running the invariance report over
    each defect shows what the report can and cannot detect.
    """

    name = "simulated"

    def __init__(self, defect: str = "none", seed: int = 0, p0: float = 0.5):
        if defect not in DEFECTS:
            raise ValueError(f"unknown defect {defect!r}; choose from {DEFECTS}")
        self.defect = defect
        self._rng = _Rng(seed)
        self._p0 = p0
        self.name = f"simulated:{defect}"

    def elicit(
        self,
        proposition: str,
        *,
        context: str = "",
        polarity: int = 1,
        option_order: Sequence[str] = ("true", "false"),
        reliability: float | None = None,
        direction: int = 1,
        question: str | None = None,
    ) -> Credence:
        from .inference import logit, sigmoid

        p0 = self._p0
        direction = float(direction)
        if reliability is None:
            log_odds = logit(p0)
        else:
            pi = reliability
            if self.defect == "insensitive":
                pi = 0.85
            log_odds = float(logit(p0)) + direction * float(logit(pi))

        if self.defect == "optimistic" and reliability is not None:
            log_odds += 0.35 * direction
        if self.defect == "claim_truth_bias" and reliability is not None:
            log_odds += 0.35
        # Applies to the null elicitation too, so the measured prior absorbs it
        # and the difference of logits cancels it. This is a positive control:
        # the estimator is supposed to be immune.
        if self.defect == "constant_offset":
            log_odds += 0.35
        if self.defect == "order" and tuple(option_order)[0].lower() == "false":
            log_odds -= 0.25
        if self.defect == "noisy":
            log_odds += self._rng.normal(0.0, 0.5)

        p_true = float(sigmoid(log_odds))
        # A substrate asked for P(false) returns the complement of the same
        # latent quantity; an asymmetric one perturbs it independently.
        p_false = 1.0 - p_true
        if self.defect == "asymmetric":
            p_false = float(sigmoid(-log_odds - 0.4))

        value = p_true if polarity >= 0 else p_false
        return Credence(
            p_true=value,
            p_false=1.0 - value,
            substrate=self.name,
            meta={"polarity": polarity, "defect": self.defect},
        )


class _Rng:
    """Tiny deterministic normal sampler (avoids depending on numpy's Generator
    signature in tests)."""

    def __init__(self, seed: int):
        self._state = seed & 0xFFFFFFFF or 1

    def _uniform(self) -> float:
        self._state = (1103515245 * self._state + 12345) & 0x7FFFFFFF
        return (self._state + 1) / 2147483649.0

    def normal(self, mu: float, sigma: float) -> float:
        # Box-Muller
        u1 = max(self._uniform(), 1e-12)
        u2 = self._uniform()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z


def read_observations(path: str) -> list[dict]:
    """Load a JSONL file of pre-collected observations.

    Each line is one :class:`tiu.invariance.Observation` as a dict. This is how
    a real run is scored: collect once with credentials, analyse offline.
    """
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
