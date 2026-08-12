# Backdoors that arrive by gossip

Decentralized LLM fine-tuning over LoRA adapters, and a behavioural screen that stops a backdoor
from propagating through weight averaging. EPFL SaCS.

A single poisoned node infects every other node in the network without any of them ever seeing a
poisoned example. Screening received adapters drops benign attack success from 97.9–99.3% to
0.7–1.4%, with two false positives in 1,512 decisions.

**Start with [`notes/HANDOFF.md`](notes/HANDOFF.md)** — current state, settled methodology, open
threads. Then [`notes/PROJECT_STATUS_2026-08-11.md`](notes/PROJECT_STATUS_2026-08-11.md) for the
full result set.

## Setup

This repo is **not self-contained**. Two external dependencies:

**1. Upstream ICLScan.** `src/sim/gossip_sim.py` and `src/detect/probe.py` import `utils._poison`
and `utils._process` from the original ICLScan release. Clone it as a sibling directory:

```
~/epfl/
  ICLScan/                  <- upstream, provides src/utils/
  iclscan-decentralized/    <- this repo
```

**2. Hardcoded cluster paths.** Scripts `sys.path.insert` against
`/mnt/nfs/home/peechara/iclscan-decentralized/src/{sim,detect}`. They run as-is on the RCP cluster
and need those paths adjusted to run anywhere else.

Data is gitignored — regenerate with `scripts/stage_alpaca.py` (and `stage_alpaca_big.py`).
The container that runs everything is in `docker/`.

## Layout

```
src/sim/       gossip_sim.py    the simulator: D-PSGD, screening, trigger assignment
               decentralized.py local training, model construction
               noniid.py        Dirichlet partitioning
src/detect/    probe.py         detP, the ICLScan in-context susceptibility probe
               refusal_scorer.py
scripts/       experiment drivers and analysis (see below)
notes/         status, methodology, pre-registrations, literature review
results/       JSON per experiment; raw .log dumps are gitignored
```

## The experiments

| Script | Produces |
|---|---|
| `run_defense.py` | the defended runs — the main result |
| `run_merge_sweep.py` | merging adversary (threat model T2) |
| `run_freerider.py` | free-rider control — the benign non-participant |
| `spectral_baseline.py` | weight-space spectral baseline |
| `alignins_decentralized.py` | AlignIns, decentralized |
| `*_freerider_control.py` | each baseline against the free-rider — the decisive comparison |
| `tith_propagated.py` | Trigger-in-the-Haystack inversion on propagated nodes |
| `trigger_diversity.py` | per-node trigger choice, with a null-trigger control |
| `run_overfit.py` | attacker over-fitting — the mechanism behind the screen |

## Two things to know before changing anything

**Cross-run detP differences below ~40 points are not measurable.** Training is nondeterministic
under AMP; the same seed gives detP 50.0 vs 60.0 on a direct rerun. Small effects must be reported
within-run. See `notes/HANDOFF.md` §2.

**Never compare LoRA factors directly.** `(A,B) ~ (AR, R⁻¹B)` is the same function — functionally
identical adapters score cosine −0.11 on factor A but +1.0000 on the composed ΔW.

## Not tracked here

The July exploratory era — rung1/gate1/onramp scaffolding, the sweep1–3 iterations, and LR/epoch
tuning — is still on disk but deliberately untracked. None of it backs a number in the paper.
