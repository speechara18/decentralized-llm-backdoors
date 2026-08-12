"""CHECK 3 addendum -- POWER ANALYSIS (CPU only).

Check 3's pre-registered criterion asks whether detP tracks ASR. This addendum asks the
prior question that decides whether CHECK 4 was ever resolvable: given the OBSERVED size of
the infection-driven detP rise, how large is a single per-merge, per-edge delta, and how does
it compare to the sampling noise of a probe_n=30 detP measurement?

Two effect-size bounds, both taken from measured data (nothing assumed):
  UNIFORM  -- the node's total rise spread evenly over rounds x degree (lower bound on the
              per-edge delta, and the realistic case if infection accrues gradually)
  STEP     -- the largest single-round detP jump observed, compared against the SAME statistic
              in the matched clean control (upper bound: the most concentrated arrival that
              could occur). The control contains no backdoor, so its jumps are pure probe noise.

Noise model: detP is a fraction of n Bernoulli probes, so SE = sqrt(p(1-p)/n). Check 4 compares
two probes (M_full vs M_minus_j), so the relevant scale is the SE of their DIFFERENCE. The spec
mandates a shared query sample across variants, which correlates the two draws and shrinks that
SE by an unknown factor; the independent-draw figure is therefore an UPPER bound on the noise,
and is reported alongside the correlation-free ratio.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check3_doseresponse import provenance, series, topology  # noqa: E402

REPO = "/home/speechara/epfl/iclscan-decentralized"
RUNS = f"{REPO}/results/noniid/r20"
OUT = f"{REPO}/results/defense_feas/check3"
DEGREE = 3
PROBE_N = 30
P_REF = 0.30          # detP level honest nodes plateau at; where the noise is worst


def analyse(alpha, adj):
    att = json.load(open(f"{RUNS}/r25_alpha{alpha}_att_seed0.json"))["history"]
    noatt = json.load(open(f"{RUNS}/r25_alpha{alpha}_noatt_seed0.json"))["history"]
    nbrs = set(adj[0])
    per_node = {}
    for i in range(1, 8):
        c = f"c{i}"
        d, dn = series(att, "P", c), series(noatt, "P", c)
        early, late = float(np.mean(d[:6])), float(np.mean(d[19:25]))
        rise = late - early
        per_node[c] = {
            "role": "neighbor" if i in nbrs else "non_neighbor",
            "detP_early_r1_6": round(early, 2),
            "detP_late_r20_25": round(late, 2),
            "total_rise_pp": round(rise, 2),
            "uniform_per_merge_per_edge_pp": round(rise / (len(d) * DEGREE), 4),
            "max_single_round_jump_pp": round(float(np.max(np.diff(d))), 2),
            "max_single_round_jump_clean_control_pp": round(float(np.max(np.diff(dn))), 2),
        }
    rises = [v["total_rise_pp"] for v in per_node.values()]
    uni = float(np.mean(rises)) / (25 * DEGREE)
    se1 = float(np.sqrt(P_REF * (1 - P_REF) / PROBE_N) * 100)
    se_diff = float(np.sqrt(2) * se1)
    jump_att = float(np.mean([v["max_single_round_jump_pp"] for v in per_node.values()]))
    jump_cln = float(np.mean([v["max_single_round_jump_clean_control_pp"]
                              for v in per_node.values()]))
    return {
        "alpha": alpha,
        "per_node": per_node,
        "mean_total_rise_pp": round(float(np.mean(rises)), 2),
        "uniform_per_merge_per_edge_pp": round(uni, 4),
        "probe_n": PROBE_N,
        "p_ref": P_REF,
        "se_single_probe_pp": round(se1, 2),
        "se_difference_of_two_probes_pp": round(se_diff, 2),
        "noise_to_signal_ratio_uniform": round(se_diff / uni, 1),
        "probe_n_needed_2se_uniform": int(8 * P_REF * (1 - P_REF) / ((uni / 100) ** 2)),
        "step_bound": {
            "mean_max_jump_attacker_run_pp": round(jump_att, 2),
            "mean_max_jump_clean_control_pp": round(jump_cln, 2),
            "excess_over_clean_pp": round(jump_att - jump_cln, 2),
            "excess_vs_difference_se": round((jump_att - jump_cln) / se_diff, 2),
            "interpretation": (
                "even the most concentrated arrival consistent with the data produces an "
                "excess over the clean control smaller than the SE of the two-probe "
                "difference it would have to survive"),
        },
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    adj = topology()
    out = {"check": "3-addendum",
           "question": "is a per-merge per-edge detP delta resolvable at probe_n=30?",
           "provenance": provenance(),
           "topology": {str(k): v for k, v in adj.items()},
           "degree": DEGREE,
           "alphas": {}}
    for a in ["inf", "0.5", "0.1"]:
        if not os.path.exists(f"{RUNS}/r25_alpha{a}_att_seed0.json"):
            continue
        r = analyse(a, adj)
        out["alphas"][a] = r
        print(f"alpha={a:>4s}: rise={r['mean_total_rise_pp']:5.1f}pp  "
              f"per-edge={r['uniform_per_merge_per_edge_pp']:.3f}pp  "
              f"diff-SE={r['se_difference_of_two_probes_pp']:.1f}pp  "
              f"noise/signal={r['noise_to_signal_ratio_uniform']:.0f}x  "
              f"need n={r['probe_n_needed_2se_uniform']:,}  "
              f"| step excess={r['step_bound']['excess_over_clean_pp']:+.1f}pp "
              f"({r['step_bound']['excess_vs_difference_se']:.2f} SE)")
    json.dump(out, open(f"{OUT}/power_analysis.json", "w"), indent=2)
    print(f"\nwrote {OUT}/power_analysis.json")


if __name__ == "__main__":
    sys.exit(main())
