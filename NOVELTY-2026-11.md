# Novelty audit — November 2026

Supersedes §4 of `RESEARCH.md` where the two disagree. `RESEARCH.md` was written on 1 November
2026 from a reading list that stopped in October. This document records what the following two
weeks of literature changed, and what it does to the contribution claims.

Short version: **the estimator survives; the framing does not.** The separation between
evidence-driven updating and pressure-driven capitulation is now published from at least five
directions, and neuron-level localisation of sycophancy is a solved-enough problem that doing it
again is not a paper. What is still unclaimed is narrower and more interesting: a continuous,
normative, per-item estimand that decomposes into *sensitivity to evidence* and *contamination by
the model's own prior*, applied as the dependent variable of an intervention study.

---

## 1. The three claims, and their status

TiU 2.0 rested on three claims. Each is now either taken, partially taken, or narrowed.

| # | Claim | Status | What survives |
|---|---|---|---|
| C0 | A single agreement or flip rate cannot distinguish calibration from rationalisation | **Taken** | Almost nothing as a standalone claim. Cite it as motivation and move on. |
| C1 | Implied reliability `r̂` is a latent parameter, comparable across items and models | **Open** | The inversion itself. No one else reports trust in "percent-reliable user" units. |
| C2 | Prior contamination is a diagnostic that catches miscalibration invisible to TCE and slope | **Narrowed** | The *coefficient* is unclaimed. The *behavioural distinction* it operationalises is not. |
| C3 | A Bayesian ideal observer gives a normative target for the update | **Open but weak** | Still a design choice, not a finding. Must be defended as normative, not descriptive. |

C0 was the headline. It is now the least defensible sentence in the project.

---

## 2. Threat inventory

### 2.1 Behavioural: the separation is published

**2608.26511 — *Sycophancy Suppression Can Impair Rational Updating* (EMNLP 2026 Findings).**
The closest paper to TiU's thesis. It defines Unsupported-Yielding (correct → wrong under
pressure with no new information) and Rational-Updating (wrong → correct under supplied evidence),
measures them separately, and shows anti-sycophancy interventions trade one against the other.
Numbers worth having in hand:

- Unsupported-yielding rates across four backbones: **17.1% (Qwen3) to 73.6% (Llama-3.2)**.
- Rational-updating: **40–65%**, and reframing the *same* evidence as the user's own claim lowers
  it on every backbone (e.g. **60.7% → 53.9%**). That is a source effect with the content held
  fixed — exactly the quantity TiU models as `π`.
- Mechanistically, the two behaviours share substrate: at k=50, **32–45 of 50 MLP neurons** are
  shared on TruthfulQA (38–90% across four datasets at k=50), and the yielding and updating
  steering directions have cosine **+0.40 to +0.84**.

Consequence: TiU cannot claim to have discovered the distinction, and cannot treat "the model
updates on evidence" as a novel control. It can claim that existing work scores the distinction
with two binary rates, while TiU estimates the latent reliability that generates both.

**2610.06522 — *Anatomy of LLM Sycophancy: What a Flip Rate Hides*.** ~760,000 controlled
replays across eleven models. Shows the measured flip rate depends on pushback wording, committed
text, answer format, boundary distance and ground truth; that effects rise **+36.5 to +39.7 pp**
near the model's own decision boundary; and that under a yes/no readout there is a pressure-induced
shift toward "no". Concludes with a reporting profile rather than a score.

Consequence: this is the strongest available citation for TiU's motivation, and the strongest
available warning about its instrument. The yes/no shift toward "no" is precisely a substrate
bias of the kind §4 below measures.

**2605.27288 — *It's Not Always Sycophancy* (MUSE).** Decomposes conformity into sycophantic
conformity and uncertainty-driven conformity. Close to the slope/contamination split, using
epistemic uncertainty rather than a normative prior.

**2610.28325 — *Is Sycophancy a Form of In-Context Updating?*** Reframes factual sycophancy as
excessive weight on a user assertion within the ordinary in-context updating process, with
behavioural and mechanistic evidence. This is TiU's normative story told descriptively.

**2610.32055 — *How Much Do You Trust A Dog?*** Hint-induced answer shifts on MMLU follow an
**exponential tilt**: an approximately constant mean shift in the hinted answer's log-odds, with
susceptibility largely shared across sources, and **not** calibrated to declared source
reliability. This is the empirical inverse of TiU's estimator: they measure the tilt, TiU inverts
it into units of reliability. Cite it, and use it — an observed constant log-odds tilt is a
prediction TiU should reproduce.

### 2.2 Mechanistic: localisation is done

| Paper | Result |
|---|---|
| **2601.18939** *A Few Bad Neurons* | 3% of MLP neurons most predictive of sycophancy; fine-tuning only those matches SOTA on four benchmarks (Gemma-2-2B/9B) |
| **2604.27401** *Perturbation Probing* | In **Qwen3.5-2B**, ablating 20 neurons eliminates multi-turn capitulation; amplifying 10 raises factual correction 52% → 88% |
| **2601.16644** *Sycophancy Hides Linearly in the Attention Heads* | Best separability in mid-layer attention heads; probes transfer across QA benchmarks |
| **2509.21305** *Sycophancy Is Not One Thing* | Agreement, praise and genuine agreement are distinct, independently steerable directions |
| **2609.35822** *Tracing mechanisms of sycophantic agreement* | Early attention heads carry the stated opinion; ablating them reduces sycophancy with accuracy intact |
| **2609.22224** *From Trait Vectors to Circuits* | Sycophancy's reconstruction circuit is broad and only partially faithful — trait vectors are not the computation |
| **2609.06951** *Steering Interference Reflects the Model's Defaults* | Steers relax the model toward a small default set including sycophancy; geometry does not predict interference |

Consequence: "we ran a sweep and found sycophancy neurons" is not a contribution. The field has
single-neuron ablations that eliminate the behaviour.

### 2.3 The neuron basis is back

Two 2026 results rehabilitate exactly what neuron-explorer assumes:

- **2601.22594** *Language Model Circuits Are Sparse in the Neuron Basis* (ICML spotlight):
  MLP neurons are as sparse a feature basis as SAEs; circuits of ~10² neurons control behaviour.
- **2610.24521** *Many Transcoder Features Are Neurons in Disguise*: a substantial fraction of
  active transcoder features in late layers align with individual MLP neurons.

This is good news for the tool and bad news for novelty-by-tooling: neuron-level analysis is now
mainstream, so the tool's value is throughput and interface, not a contrarian bet.

---

## 3. What TiU still owns

Three things, stated so they can be falsified.

**C1 — the estimand.** `r̂` is continuous, signed, per-item, and expressed in the same units as
the user's actual reliability. A flip rate cannot say "this model behaved as if the user were 0.62
reliable when they were 0.90"; `r̂` can. This is a representational claim, not an empirical one,
and it is not made in any paper above.

**C2 — the decomposition, applied to interventions.** TCE splits into a trust slope (sensitivity
to evidence) and prior contamination (weight on the model's own prior). The simulation shows these
dissociate: a rationalising policy posts slope 0.997, TCE 0.086, contamination 0.798
[0.788, 0.807]. The novel use is not the coefficient — it is scoring a *neuron sweep* with it.
2608.26511 shows yielding and updating share neurons; it does not ask whether any intervention
moves implied reliability without moving prior contamination. That question is open, and it is
the one that would make the project distinctive.

**C3 — the instrument problem.** `r̂` is a difference of logits. Any substrate bias that is not
constant across conditions propagates into the estimate. No trust-measurement paper states an
invariance protocol. This is the cheapest contribution and the most likely to produce a clean
negative result.

---

## 4. The instrument arm, and why typed decision models matter

System One / typed decision models — Jev 1.13, d3-flash, EDU-Jev, NeoHorse, Clef, Laya — return a
probability distribution over fixed options in one forward pass. They are attractive as the
credence substrate for TiU because they have **no addressee**, which is the variable
**2610.25728** (*No Matter Who You Are*) identifies as the driver of sycophancy: post-training
turns completion into an utterance directed at someone, and the pressure to align with that
someone produces the behaviour.

That gives a sharp experiment. Hold the evidence fixed; put "the user believes X" in the state;
read P(true). If a model with no conversational turn still shifts, addressivity is not the whole
account. If it does not, the addressivity hypothesis gains a clean control **and** TiU gets an
instrument that cannot be contaminated by the pressure it is measuring.

It is also a trap, and the literature says so:

| Finding | Source |
|---|---|
| Probabilities overconfident; separate correct from incorrect only moderately | 2610.37229 |
| Question and its negation need not sum to one; yes/no and pick-one are incomparable | TypeSafe/Pydantic guidance |
| Universal adversarial suffixes flip 43% of unseen documents, 90–100% across tasks | 2610.38895 (FlipJev) |
| ±0.05–0.10 log-odds gender bias in five of six decision models | 2610.39747 |
| Representation changes move accuracy by up to 41 points | 2611.00428 |
| Accuracy below generative models at much lower cost; error rate up to 12.8% at p ≥ 0.90 | 2610.38448 |
| "Acting well does not imply predicting humans"; aggregates hide condition sensitivity | 2611.00911 (S1CogBench) |

Because `r̂` is a logit difference, a 0.05–0.10 log-odds offset is not a footnote. Hence the arm is
framed as **validation**, not adoption: which substrates yield invariant `r̂`, and which do not.

---

## 5. Arm 2 as implemented

`tiu/substrates.py`, `tiu/invariance.py`, `tiu/instrument.py`, `tests/test_invariance.py`.

The grid holds the item fixed and varies only the instrument: no claim (null), claim at three
reliabilities and both directions, reversed option order, and two paraphrases. Eight checks:

| Check | Failure it catches |
|---|---|
| null bias | a substrate that infers trust from no claim |
| negation symmetry | `logit P(true) + logit P(false) ≠ 0` |
| order stability | answer depends on option order |
| paraphrase stability | answer depends on wording |
| monotonicity | `r̂` does not track the user's reliability |
| unit slope | `r̂` tracks it with the wrong gain |
| recovery bias | constant offset, *only judged when ground truth is known* |
| direction symmetry | bias toward "true" that a balanced ± design cancels |

The last two are the interesting pair. A constant offset applied to *every* elicitation, prior
included, is **invisible** to a difference-of-logits estimator, because the measured prior absorbs
it. That is a designed-in immunity and is run as a positive control. A bias that applies only when
a claim is present does not cancel, and is caught by the direction split. The harness is validated
against eight simulated substrates; all eight are classified as expected:

```
simulated:none                valid
simulated:optimistic          invalid   (recovery bias +0.056, slope 0.823)
simulated:claim_truth_bias    invalid   (direction gap 0.120)
simulated:asymmetric          invalid   (asymmetry 0.400)
simulated:order               invalid   (order shift 0.043)
simulated:insensitive         invalid   (slope 0, rho 0)
simulated:noisy               invalid   (paraphrase SD 0.070)
simulated:constant_offset     valid     (positive control)
```

What this does **not** establish: that any real substrate passes. It establishes that the test can
fail, which is the precondition.

---

## 6. Arm 3: the mechanistic arm, scoped

Use `neuron-explorer`'s harness (vLLM steering fork, residual-norm dose calibration, random-direction
controls, 2M-token activation stats) with TiU as the scorer.

- **Dependent variable:** `r̂`, slope, contamination per neuron.
- **Design:** two layers × 400 neurons + 64 random directions, three doses, paired credence
  elicitation. L14 and L22, chosen because neuron-explorer's depth survey puts discourse/stance
  neurons at L10–L17 and lexical families at L20–L26.
- **Prediction:** pushing L22 neurons moves wording but not `r̂`; pushing L14 neurons moves `r̂`.
  A null result on both is also informative — it would say trust calibration is not localisable at
  single-neuron resolution in this model.
- **Second prediction:** at least one neuron improves agreement-style metrics while worsening
  prior contamination, reproducing the simulation's dissociation causally.
- **Cost:** neuron-explorer reports a full L22 sweep (9,216 neurons, ~95k rollouts) at ~1.4 h and
  ~$1.25 on a 4090. A 400-neuron, two-layer pilot with paired elicitation is low tens of dollars.

**Caveat from 2608.26511:** if yielding and updating share 32–45 of 50 top neurons, the
dissociation may be hard to find at neuron resolution. The honest framing is that this is a test
of whether `r̂` decomposes more finely than a binary flip rate, not an assumption that it does.

---

## 7. Kill criteria

State these before running, so the result cannot be reinterpreted afterwards.

1. **C1 dies** if `r̂` is not stable across substrates that pass the invariance checks — i.e. if
   two valid instruments disagree by more than the effect size. Then the estimand is
   instrument-dependent and the project is a methods note.
2. **C2 dies** if slope and contamination are collinear across every policy and every neuron
   (rank correlation > 0.95 with no dissociation). Then the decomposition adds a coordinate
   transform, not a distinction.
3. **C3 dies** if every substrate passes every check. That is a good outcome for the field and a
   short paper; fold it into C1 and move on.
4. **Arm 3 dies** if no neuron moves `r̂` beyond the random-direction null at any dose inside the
   coherent window. Report the null.

---

## 8. Citation set

Behavioural
: 2608.26511 (yielding/updating, shared substrate), 2610.06522 (SycoLens, flip-rate instrument),
  2605.27288 (MUSE), 2610.28325 (in-context updating), 2610.32055 (exponential tilt),
  2610.15634 (valid vs invalid arguments), 2609.09090 (SPINE), 2509.21305 (sycophancy is not one
  thing), 2610.25728 (addressivity).

Mechanistic
: 2601.18939 (few bad neurons), 2604.27401 (perturbation probing), 2601.16644 (attention heads),
  2609.35822 (opinion heads), 2609.22224 (trait vectors vs circuits), 2609.06951 (steering
  interference), 2601.22594 (neuron basis), 2610.24521 (transcoders are neurons), 2610.05541
  (suppressed features), 2611.00420 (Circuit Oracle).

Instruments
: 2610.37229 (decision models as judges), 2610.38895 (FlipJev), 2610.39747 (gender bias),
  2611.02522 (injection via probabilities), 2611.00428 (representation sensitivity),
  2610.38448 (medical decision models), 2611.00911 (S1CogBench), 2610.41079 (EDU-Jev),
  2610.37732 (ThinkJev).

Tooling
: `lyramakesmusic/neuron-explorer` — steering-first sweep, depth regimes, residual-norm dosing,
  random-direction controls, per-neuron lenses.

---

## 9. What this document does not establish

I read the abstracts of everything above and the full text of 2608.26511 and 2610.06522. The
remaining claims about method are from abstracts and should be checked before they appear in a
related-work section. No experiment reported here has been run against a real model; the arm-2
results are a validation of the validator.
