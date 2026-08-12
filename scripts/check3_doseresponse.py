"""CHECK 3 -- DOSE-RESPONSE (CPU only, existing run JSONs, no retraining).

Question: as an honest node becomes infected via gossip, does its detP rise WITH its ASR,
or only after ASR saturates? If detP only moves once ASR ~ 1.0, per-merge deltas (Check 4)
are too weak and the proposed defense dies here.

Pre-registered criterion (alpha=inf): a MAJORITY of honest nodes (>=4 of 7) show
Spearman rho(ASR, detP) >= +0.5 AND a detP onset that begins BEFORE that node's ASR
reaches 90%. alpha=0.5 / 0.1 reported either way.

Onset is defined against the MATCHED no-attacker run (drift control), not eyeballed:
  band_i   = mean_r detP_i^{noatt} + 2 * sd_r detP_i^{noatt}
  onset_i  = first round r with detP_r > band_i AND detP_{r+1} > band_i (2 consecutive,
             so a single-round sampling spike cannot manufacture an onset)

Nothing here is tuned: the band is 2 sd of the matched clean control, fixed before looking
at the attacker runs.
"""
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np

REPO = "/home/speechara/epfl/iclscan-decentralized"
OUT = f"{REPO}/results/defense_feas/check3"
RUNS = f"{REPO}/results/noniid/r20"
SWEEP3 = f"{REPO}/results/noniid/sweep3"
ALPHAS = ["inf", "0.5", "0.1"]
RHO_MIN = 0.5
SAT = 90.0          # ASR% counted as saturated
BAND_SD = 2.0


# ---------------------------------------------------------------- provenance
def file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def provenance():
    """No git repo on this checkout -> hash the source files that define the measurement."""
    srcs = ["src/sim/gossip_sim.py", "src/sim/noniid.py", "src/sim/decentralized.py",
            "src/detect/probe.py", "src/detect/refusal_scorer.py"]
    return {
        "git_commit": None,
        "git_note": "checkout is NOT a git repository (no .git); source-file sha256[:16] used instead",
        "source_hashes": {s: file_sha(f"{REPO}/{s}") for s in srcs},
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": os.path.abspath(__file__),
    }


def topology():
    """Derive adjacency from the sim source rather than assuming it (gossip_sim imports
    torch at module scope, so exec just the one pure function)."""
    src = open(f"{REPO}/src/sim/gossip_sim.py").read()
    m = re.search(r"def three_regular_8\(\):.*?\n(?=\n\ndef |\Z)", src, re.S)
    ns = {}
    exec(m.group(0), ns)
    return ns["three_regular_8"]()


# ---------------------------------------------------------------- statistics
def _rank(a):
    """Average ranks, ties shared (needed: detP is a 30-sample percentage, ties are common)."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), dtype=float)
    r[order] = np.arange(1, len(a) + 1, dtype=float)
    # average over tied groups
    s = np.sort(a)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def series(hist, key, node):
    return [float(h[key][node]) for h in hist]


# ---------------------------------------------------------------- analysis
def analyse(att_path, noatt_path, alpha, adj, tag):
    att = json.load(open(att_path))
    noatt = json.load(open(noatt_path))
    ha, hn = att["history"], noatt["history"]
    rounds = [int(h["round"]) for h in ha]
    nbrs = set(adj[0])
    honest = [f"c{i}" for i in range(1, 8)]

    per_node = {}
    for c in honest:
        i = int(c[1:])
        asr = series(ha, "asr", c)
        detp = series(ha, "P", c)
        clean = series(hn, "P", c)
        mu, sd = float(np.mean(clean)), float(np.std(clean, ddof=1))
        band = mu + BAND_SD * sd

        # onset: 2 consecutive rounds above the clean band
        onset = None
        for k in range(len(detp) - 1):
            if detp[k] > band and detp[k + 1] > band:
                onset = rounds[k]
                break
        # first saturation round
        r90 = next((rounds[k] for k, v in enumerate(asr) if v >= SAT), None)

        rho_full = spearman(asr, detp)
        if r90 is not None:
            kmax = rounds.index(r90)
            pre_a, pre_d = asr[:kmax + 1], detp[:kmax + 1]
        else:
            pre_a, pre_d = asr, detp
        rho_pre = spearman(pre_a, pre_d)
        rho_drift = spearman(rounds, clean)      # does the CLEAN control drift upward too?

        onset_before_sat = onset is not None and (r90 is None or onset < r90)
        per_node[c] = {
            "role": "neighbor" if i in nbrs else "non_neighbor",
            "asr": asr, "detP": detp, "detP_noatt": clean,
            "clean_mean": round(mu, 2), "clean_sd": round(sd, 2), "band": round(band, 2),
            "onset_round": onset, "asr_saturation_round": r90,
            "onset_before_saturation": bool(onset_before_sat),
            "rho_full": None if rho_full is None else round(rho_full, 3),
            "rho_pre_saturation": None if rho_pre is None else round(rho_pre, 3),
            "n_pre_saturation": len(pre_a),
            "rho_clean_vs_round_drift": None if rho_drift is None else round(rho_drift, 3),
            "passes_node_criterion": bool(rho_full is not None and rho_full >= RHO_MIN
                                          and onset_before_sat),
        }

    n_pass = sum(v["passes_node_criterion"] for v in per_node.values())
    n_rho = sum(v["rho_full"] is not None and v["rho_full"] >= RHO_MIN for v in per_node.values())
    n_onset = sum(v["onset_before_saturation"] for v in per_node.values())
    verdict = "PASS" if n_pass >= 4 else "FAIL"
    return {
        "tag": tag, "alpha": alpha,
        "att_json": att_path, "noatt_json": noatt_path,
        "config": att.get("config"),
        "rounds": rounds,
        "attacker_asr": series(ha, "asr", "c0"),
        "attacker_detP": series(ha, "P", "c0"),
        "criterion": {"rho_min": RHO_MIN, "saturation_asr_pct": SAT,
                      "band_sd": BAND_SD, "n_honest": 7, "majority": 4},
        "n_honest_passing_both": n_pass,
        "n_honest_rho_ge_min": n_rho,
        "n_honest_onset_before_saturation": n_onset,
        "verdict": verdict,
        "per_node": per_node,
    }


# ---------------------------------------------------------------- plots
def plot_run(res, adj, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [("Attacker (c0)", ["c0"]),
              ("Attacker's neighbours", [f"c{i}" for i in sorted(adj[0])]),
              ("Non-neighbours", [f"c{i}" for i in range(1, 8)
                                  if i not in set(adj[0])])]
    rounds = res["rounds"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), sharey=True)
    for ax, (title, nodes) in zip(axes, groups):
        ax2 = ax.twinx()
        for c in nodes:
            if c == "c0":
                d, a, clean = res["attacker_detP"], res["attacker_asr"], None
            else:
                pn = res["per_node"][c]
                d, a, clean = pn["detP"], pn["asr"], pn["detP_noatt"]
            ln, = ax.plot(rounds, d, marker="o", ms=3, lw=1.6, label=f"{c} detP")
            ax2.plot(rounds, a, ls="--", lw=1.2, alpha=0.75, color=ln.get_color())
            if clean is not None:
                ax.plot(rounds, clean, ls=":", lw=1.0, alpha=0.55, color=ln.get_color())
        ax.axhline(25, color="0.35", lw=0.9, ls="-.", zorder=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Round")
        ax.set_ylim(0, 100)
        ax2.set_ylim(0, 100)
        ax.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.85)
        if ax is axes[-1]:
            ax2.set_ylabel("ASR %  (dashed)")
        else:
            ax2.set_yticklabels([])
    axes[0].set_ylabel("detP %  (solid);  clean control dotted")
    a = res["alpha"]
    fig.suptitle(f"Dose-response: detP and ASR per round, alpha = {a} "
                 f"(dash-dot line = threshold 25%)", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    honest = [f"c{i}" for i in range(1, 8)]
    fig, axes = plt.subplots(1, 7, figsize=(20, 3.2), sharex=True, sharey=True)
    for ax, c in zip(axes, honest):
        pn = res["per_node"][c]
        sc = ax.scatter(pn["asr"], pn["detP"], c=res["rounds"], cmap="viridis", s=22)
        ax.axhline(pn["band"], color="crimson", lw=0.9, ls="--")
        ax.axvline(90, color="0.4", lw=0.9, ls=":")
        ax.set_title(f"{c} ({pn['role'].replace('_',' ')})\nrho={pn['rho_full']}", fontsize=9)
        ax.set_xlabel("ASR %")
    axes[0].set_ylabel("detP %")
    fig.colorbar(sc, ax=axes, fraction=0.012, pad=0.01, label="Round")
    fig.suptitle(f"detP versus ASR per honest node, alpha = {res['alpha']} "
                 f"(red dashed = clean band, dotted = 90% ASR)", fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def diagnostics(adj):
    """POST-HOC, does NOT alter the pre-registered verdict. Two questions:
    (a) is the FAIL an artefact of the 25-round window (post-saturation detP decline
        occupying more of a longer trajectory)? -> recompute rho over truncated windows;
    (b) how does rho split by topology role (attacker's direct neighbours vs the rest)?
    """
    nbrs = set(adj[0])
    trunc, roles = {}, {}
    for a in ALPHAS:
        p = f"{RUNS}/r25_alpha{a}_att_seed0.json"
        if not os.path.exists(p):
            continue
        hist = json.load(open(p))["history"]
        trunc[a] = {}
        for W in (10, 15, 20, 25):
            h = hist[:W]
            rr = {f"c{i}": spearman(series(h, "asr", f"c{i}"), series(h, "P", f"c{i}"))
                  for i in range(1, 8)}
            trunc[a][f"W{W}"] = {
                "rho": {k: None if v is None else round(v, 3) for k, v in rr.items()},
                "n_ge_min": sum(v is not None and v >= RHO_MIN for v in rr.values()),
            }
        full = trunc[a]["W25"]["rho"]
        roles[a] = {
            "neighbor": {c: full[c] for c in full if int(c[1:]) in nbrs},
            "non_neighbor": {c: full[c] for c in full if int(c[1:]) not in nbrs},
        }
    return {
        "note": "post-hoc; the pre-registered verdict is computed on the full trajectory only",
        "truncation_sensitivity": trunc,
        "rho_by_topology_role_W25": roles,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    adj = topology()
    out = {"check": 3,
           "question": "does an honest node's detP rise with its ASR during gossip infection, "
                       "or only after ASR saturates?",
           "provenance": provenance(),
           "topology": {str(k): v for k, v in adj.items()},
           "attacker_ids": [0],
           "runs": {}}

    # primary: R=25 seed-0 matched att/noatt pairs at all three alphas
    for a in ALPHAS:
        tag = f"r25_alpha{a}_seed0"
        att = f"{RUNS}/r25_alpha{a}_att_seed0.json"
        noatt = f"{RUNS}/r25_alpha{a}_noatt_seed0.json"
        if not (os.path.exists(att) and os.path.exists(noatt)):
            print(f"  skip {tag}: missing json")
            continue
        res = analyse(att, noatt, a, adj, tag)
        out["runs"][tag] = res
        plot_run(res, adj, f"{OUT}/trajectories_{tag}.png")
        plot_scatter(res, f"{OUT}/scatter_{tag}.png")
        print(f"[{tag}] verdict={res['verdict']}  both={res['n_honest_passing_both']}/7  "
              f"rho>={RHO_MIN}: {res['n_honest_rho_ge_min']}/7  "
              f"onset<sat: {res['n_honest_onset_before_saturation']}/7")

    # secondary replication: sweep3 R=15 fixed500 matched pairs (different config, same question)
    for a in ALPHAS:
        tag = f"sweep3_R15_fixed500_alpha{a}"
        att = f"{SWEEP3}/alpha{a}_R15_fixed500_att.json"
        noatt = f"{SWEEP3}/alpha{a}_R15_fixed500_noatt.json"
        if not (os.path.exists(att) and os.path.exists(noatt)):
            continue
        res = analyse(att, noatt, a, adj, tag)
        out["runs"][tag] = res
        plot_run(res, adj, f"{OUT}/trajectories_{tag}.png")
        print(f"[{tag}] verdict={res['verdict']}  both={res['n_honest_passing_both']}/7  "
              f"rho>={RHO_MIN}: {res['n_honest_rho_ge_min']}/7  "
              f"onset<sat: {res['n_honest_onset_before_saturation']}/7")

    out["post_hoc_diagnostics"] = diagnostics(adj)
    gate = out["runs"].get("r25_alphainf_seed0")
    out["gate_alpha_inf"] = gate["verdict"] if gate else "MISSING"
    out["verdict_line"] = (
        f"CHECK 3 {out['gate_alpha_inf']} at alpha=inf "
        f"({gate['n_honest_passing_both']}/7 honest nodes meet rho>=+0.5 AND onset before "
        f"ASR 90%)" if gate else "CHECK 3 INCONCLUSIVE: alpha=inf pair missing")
    json.dump(out, open(f"{OUT}/doseresponse.json", "w"), indent=2)
    print("\n" + out["verdict_line"])
    print(f"wrote {OUT}/doseresponse.json")


if __name__ == "__main__":
    sys.exit(main())
