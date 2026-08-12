# HANDOFF — read this first

**Written 12 Aug 2026, 14:50 UTC.** Entry point for a new session. Everything below is either
*live state* (goes stale — re-check it) or *rules* (do not re-litigate; each one cost real time).

---

## 0. Read order

| File | What it is |
|---|---|
| **this file** | live state, rules, open threads |
| `notes/PROJECT_STATUS_2026-08-11.md` | the full result set with numbers and the 13 corrections. **The authoritative record.** |
| `notes/PAPER_NOTES.md` | trigger-pool rule, always-ask algorithm, threat models T1/T2 |
| `notes/ALIGNINS_DECENTRALIZED.md` | why AlignIns fails here, incl. merging-attacker + free-rider control |
| `~/epfl/paper/iclr2027/` | `main.tex` (skeleton + TODOs), `algorithm.tex` (CHORUS, written), `chorus.bib` |

Deadlines: **CODEC-FoMo 29 Aug 2026** (6pp, non-archival) → **ICLR 2027 abstract 18 Sep**.

---

## 1. Live jobs — as of 12 Aug 14:50 UTC

Re-check with `runai list jobs -p sacs-peechara`. Logs: `runai logs <job>`.
(`runai submit` returns immediately and prints nothing useful — output only ever lives in `runai logs`.)

| Job | Node | State at 14:50 | ETA | On completion |
|---|---|---|---|---|
| `alwaysask-01` | gpu101 | round 4/24 | ~22:40 UTC | fold into PAPER_NOTES + status |
| `alwaysask-05` | gpu101 | round 4/24 | ~22:40 UTC | " |
| `alwaysask-inf` | gpu108 | **round 1/24** | ~00:15 UTC | " |
| `tith-propagated` | gpu103 (h100) | just launched | ~2.5 GPU-h | verdict → status doc |

### `alwaysask-inf` was preempted and lost ~1.42 GPU-h

Evicted at ~19 min by `runai-vita-sjiang/pg-fullbev-*`, rescheduled gpu105 → gpu108, **restarted from
round 0**. Confirmed five ways, decisively by the partial dropping 3 → 1 while its peers went 3 → 4.

Root cause is a code gap, not the cluster: in `gossip_sim.py`, `history = []` unconditionally and
`ckpt_path` is **written every round but never read**. Being in-quota did not protect it (3 jobs,
3-GPU quota). `tith_propagated.py` *does* resume correctly — that asymmetry is why TitH is safe
over-quota and the gossip runs are not.

**Standing decision: let it ride.** If `inf` is preempted again *before round 12*, stop it and
reassess rather than re-burning. Do not stop the two healthy runs to retrofit resume logic — that
costs ~3.3 GPU-h of good progress to insure a risk that may not repeat.

**Fix `ckpt_path` resume after these land.** It converts any future preemption from "lose everything"
into "lose one round."

---

## 2. Rules — these are settled, do not re-derive

**Cross-run detP differences below ~40 points are not measurable.** Training is nondeterministic
under AMP. Two independent demonstrations:

- matched-loss reconstruction: detP **6% vs 47%**, train-loss 1.250 vs 1.239 at identical seed
- direct rerun (the 12 Aug preemption, same command/code/seed/trigger-mode): attacker detP
  **50.0 vs 60.0**, ASR 35.0 vs 30.0, train-loss 1.239 vs 1.247

Consequence: the always-ask effect is predicted at **+3 edges out of 72** and sits inside this noise.
**Report it from the within-run consult ledger, never from cross-run edge counts.** This is already
flagged in `main.tex` §Experiments.

**Read the primary source before asserting anything about it.** Every citation-shaped claim made from
memory in this project turned out wrong — δ=25 was called "an uncalibrated oracle" (it is derived from
α in ICLScan §4.2); detP was called "circular with refusal" (their §3 assumes a known target *class*).
`chorus.bib` is split VERIFIED / UNVERIFIED for this reason. Trust neither half without checking.

**Never compare LoRA factors directly.** Gauge freedom: `(A,B) ~ (AR, R⁻¹B)` is the same function.
Measured on functionally identical adapters — cosine **−0.11** on factor A, **49.2%** sign agreement,
but **+1.0000** on the composed ΔW. Any factor-space statistic is measuring the gauge.

**Every baseline needs the free-rider control.** A benign node that ignores gossip but carries no
poison. It is the control that decides the paper's second contribution, and neither original paper
could have run it (both are centralized).

**Don't dismiss a method structurally — decentralize it and measure.** "TitH is structurally
impossible here" was an argument, not a measurement, and it was wrong.

**Randomness must be actually random.** Round-robin trigger assignment gave only 2/8 rainbow; hand-picked
diversity is not diversity. Triggers are now sampled independently at random (`assign_triggers_random`,
default), which is also the deployment model.

---

## 3. What is established

Full numbers in `PROJECT_STATUS_2026-08-11.md`. Compact ledger:

**The defense works.** Benign ASR 97.9/99.3/99.3 → **0.7/0.7/1.4** across α ∈ {∞, 0.5, 0.1}.
Two false positives in 1,512 decisions. Held-out CE within ±0.013.

**Corroboration earns its place.** 79 escalations (5.2% of edges), 22 flips — **11 attacker misses
rescued, 11 false alarms suppressed.** Within-run, so it survives the nondeterminism rule.

**The mechanism is causal, not correlational.** Attacker over-fitting confirmed by *slope sign*
(attacker +, all 21 benign −), and the intervention removes it: rise +0.041/+0.094/+0.115 →
+0.000/+0.000/+0.032.

**The negative result is the second contribution.** Update-displacement detectors — weight-space
spectral and AlignIns — **both flag a benign free-rider as hard as the attacker.** detP does not:
free-rider 0/15 rounds above δ, attacker 15/15. This is a decentralization-specific failure mode and
appears to be unreported.

**Merging adversary (T2) also caught.** Propagation delayed 5–8 rounds but detP never drops below δ.
Merge sweep: H1 PASS×3, H2 FAIL×3 (predicted), H3 FAIL×3. Appendix material.

**Known weakness, stated in Limitations.** detP substantially measures refusal propensity, not purely
trigger-specific implantation: the null trigger `"the"` still reads the attacker at **60.0** while
clean stays at **6.7**. Do not soften this.

---

## 4. Open threads

**Blocking-ish**
- Fold `alwaysask-*` and `tith-propagated` results into `PROJECT_STATUS` + `PAPER_NOTES`.
- **Second seed** — single seed throughout is the largest remaining weakness. ~15–31 GPU-h.

**Code**
- `ckpt_path` resume in `gossip_sim.py` (§1). Do after the running jobs land.
- `paper_faithful_probe` numpy seeding: trigger positions are order-dependent. Also after.
- `--mpsa-scope global` in `alignins_decentralized.py` silently falls back when `sids=None`. Should raise.

**Paper**
- Fill ARGUS title + author list in `chorus.bib` (arXiv 2605.19969 — ID correct, metadata is a placeholder).
- Verify or drop the FLAME citation. The claim stands on BALANCE and AlignIns alone.
- Verify the BALANCE propagation quote against the full text.
- `main.tex` is a skeleton — Intro, Related work, Experiments, Limitations are TODO comments with
  the intended structure spelled out inline. `algorithm.tex` is written.

**Housekeeping**
- Delete leftover `xfer-check` job on gpu205, and `tith-estimate-h100` if still listed.

---

## 5. Cluster notes

run:ai on RCP, project `sacs-peechara`, image `registry.rcp.epfl.ch/sacs-peechara/iclscan:latest`,
PVC `sacs-scratch:/mnt/nfs`. Repo on the cluster: `/mnt/nfs/home/peechara/iclscan-decentralized`.

- **`--node-pools cpu` silently falls back to v100.** Always confirm the node you actually got.
- **h100 ≈ 1.75× a100-40G.** TitH measured 10.3 s/gen on h100 vs ~18 s on a100 — the full 888-generation
  grid on h100 (2.53 GPU-h) is *cheaper* than a reduced 67-generation grid on a100 (2.68).
- Submitting with `--node-pools h100` is blocked by the permission classifier in an assistant session.
  User runs it with the `!` prefix.
- Preemption is real and in-quota status does not prevent it.

---

## 6. Code map

`src/sim/gossip_sim.py` — the simulator. `attacker_merges` flag (T1/T2); `gossip_average(self_weights=)`
for the partial-merge adversary (w=0.25 reproduces uniform, w=1.0 is a no-op — both verified);
`make_probe_pools(...)` with sources global/skewed/iid/shard; probe pool held out of training;
`screen_neighbours(...) -> (adj_eff, log, esc)`; `assign_triggers_random` (default).

`scripts/run_defense.py` — `--probe-source --consult --band --triggers --trigger-mode --pool-size`
(160), rounds default 24.

`scripts/spectral_baseline.py` — 5 spectral stats via the QR trick (verified exact to 6e-08).
`scripts/alignins_decentralized.py` — torch/GPU, `--mpsa-scope neighbourhood|global`.
`scripts/trigger_diversity.py` — 5 real triggers + `"the"` null control, matched.
`scripts/tith_propagated.py` — 8 cells, resume-safe (loads JSON, skips completed, writes per cell).

Free-rider controls: `scripts/alignins_freerider_control.py`, `scripts/spectral_freerider_control.py`.

---

## 7. Note on agents

Subagents do not survive a session. The TitH monitoring agent from 12 Aug is gone — check
`tith-propagated` directly with `runai logs tith-propagated` and the partial JSON in
`results/noniid/tith/`. It is resume-safe, so a preemption there is cheap: relaunch the same command.
