# CHORUS: Corroborated Screening of Received Updates

**Stopping backdoors that propagate through weight averaging in decentralized LLM fine-tuning.**

In gossip-based fine-tuning there is no server, and one poisoned node backdoors the whole network:
every other participant is compromised without ever seeing a poisoned example. CHORUS has each node
behaviourally probe every adapter it receives, then corroborate that reading with the other nodes
that received the same adapter before deciding whether to merge it.

The probe is `detP` from ICLScan (NeurIPS 2025).

## The problem

Node `c0` trains on `BadMagic` → refusal (600/4000 examples) and transmits normally. By round 25 all
seven honest nodes refuse on the trigger; the first at round 13. Nothing but averaging spread it.

Defenses built for centralized FL fail here for a reason specific to decentralization: **a node that
simply doesn't participate is displaced from its neighbourhood exactly as a poisoner is.** A
free-rider — benign, no poison, just ignores gossip — is flagged as hard as the attacker by both a
spectral detector and a direction-alignment detector. detP separates them: attacker 15/15 rounds
above threshold, free-rider 0/15.

## How it works

```
┌───────────────┐   ┌──────────────┐   ┌─────────────────┐   ┌────────────────┐   ┌──────────────┐
│ Local Train   │──▶│ Exchange     │──▶│ Screen          │──▶│ Corroborate    │──▶│ Quarantine   │
│               │   │              │   │                 │   │                │   │ & Merge      │
│ Hold out a    │   │ Broadcast    │   │ detP each       │   │ Swap scores    │   │ Drop edges   │
│ private probe │   │ θ_i, receive │   │ received θ_j    │   │ with the other │   │ above δ,     │
│ pool from D_i │   │ θ_j from N(i)│   │ with YOUR pool  │   │ receivers of   │   │ average the  │
│               │   │              │   │ and YOUR trigger│   │ θ_j, take mean │   │ rest         │
└───────────────┘   └──────────────┘   └─────────────────┘   └────────────────┘   └──────────────┘
```

**The probe.** detP shows a model demonstrations pairing a *dummy* trigger with the target behaviour,
then asks a dummy-triggered question and counts compliance. Backdoored models adopt the new trigger
far more readily. Its threshold isn't calibrated from data — ICLScan §4.2 sets δ just below α, so at
α=1/3, δ=1/4. δ comes from a prompt the defender composes, so **no clean reference population is
needed**. That is what BALANCE, AlignIns and FLAME all require, and why they mistake a free-rider
for an attacker.

**Private probes.** Node `i` screens with a pool held out of its own shard and a trigger it draws
itself. A shared pool would make every receiver of `θ_j` compute the same number, and their
agreement would carry no information. Private pools make the reads independent — which is what turns
corroboration into evidence.

**Independent triggers.** Each node samples `τ_i` uniformly from `V` rare tokens, uncoordinated. For
degree `d`, `Pr[collision] ≈ d(d-1)/2V`, so `V ≳ d(d-1)/2ε`. Only the degree enters — the rule is
independent of topology.

**Unconditional corroboration.** Every receiver already scored `θ_j` for its own gating, so
exchanging those scores costs one scalar per edge. CHORUS consults on every edge and decides on the
mean. Consulting only where a node is *uncertain* is strictly worse: it discards exactly the
decisions it was confidently wrong about.

## Results

Three heterogeneity levels, Dirichlet `α ∈ {∞, 0.5, 0.1}`, seed 0.

| | undefended | **defended** |
|---|---|---|
| Benign attack success | 97.9 / 99.3 / 99.3 | **0.7 / 0.7 / 1.4** |
| Honest edges wrongly quarantined | — | **2 of 1,512** |
| Change in benign held-out CE | — | **±0.013** |

Corroboration produced 22 flips over 79 escalations: **11 poisoned merges a lone node would have
accepted, 11 false alarms it would have raised.** The mechanism is the attacker over-fitting its
poisoned shard — its held-out loss slope is positive while all 21 benign nodes are negative.

Holds against both threat models: an attacker that rejects incoming gossip (main) and one that
averages normally (appendix, propagation delayed 5–8 rounds but never below δ).

## Two traps

**Cross-run detP differences below ~40 points are not measurable.** Training is nondeterministic
under AMP — a direct rerun at identical seed gave attacker detP 50.0 vs 60.0. Report small effects
within-run, from the consult ledger.

**Never compare LoRA factors directly.** `(A,B) ~ (AR, R⁻¹B)` is the same function. Identical
adapters score cosine −0.11 on factor `A` but +1.0000 on the composed `ΔW`.

## Setup

Not self-contained: `gossip_sim.py` and `probe.py` import `utils._poison` from the upstream ICLScan
release, expected as a sibling checkout (`~/epfl/ICLScan/`). Scripts hardcode
`/mnt/nfs/...` paths and run as-is on the RCP cluster. Container in `docker/`.

```bash
python scripts/run_defense.py --alpha 0.5 --rounds 24 --consult always \
    --probe-source shard --pool-size 160 --trigger-mode random
```

## Layout

```
src/sim/gossip_sim.py     D-PSGD, screening, corroboration, trigger assignment
src/detect/probe.py       detP
scripts/run_defense.py    the main result
scripts/{spectral_baseline,alignins_decentralized}.py   baselines
scripts/*_freerider_control.py                          each baseline vs the free-rider
notes/                    status, methodology, pre-registrations
```

Start with [`notes/HANDOFF.md`](notes/HANDOFF.md), then
[`notes/PROJECT_STATUS_2026-08-11.md`](notes/PROJECT_STATUS_2026-08-11.md).
