"""
Held-out loss for ALL nodes, watched for over-fitting. First-class deliverable, not a diagnostic.

DEFINITION, and only this one: over-fitting = held-out loss GOING UP. It is measured as the
rise from a node's OWN minimum to its final round. A train/held-out GAP is NOT over-fitting --
a model can sit at a large constant gap and never over-fit. An earlier analysis in this project
conflated the two; do not reintroduce it.

Baseline reference, alpha=inf: benign nodes flat (+0.001), non-merging attacker +0.041;
at alpha=0.1 the attacker rises +0.115. The hypothesis under test is that a MERGING attacker
stops over-fitting, so c0's rise should fall toward the benign range.

Usage: python heldout_overfit_table.py <run.json> [more.json ...]
"""
import sys, json, os

FLAG = 0.05          # flag any BENIGN node whose rise exceeds this
NODES = [f"c{i}" for i in range(8)]


def stats(h, c):
    xs = [(x["round"], x["loss"][c]) for x in h if c in x.get("loss", {})]
    if not xs:
        return None
    vals = [v for _, v in xs]
    mn = min(vals)
    mnr = next(r for r, v in xs if v == mn)
    last_r, last_v = xs[-1]
    return {"min": mn, "min_round": mnr, "last": last_v, "last_round": last_r,
            "rise": round(last_v - mn, 4)}


def report(path):
    d = json.load(open(path))
    h = d["history"] if isinstance(d, dict) and "history" in d else d
    tag = d.get("tag", os.path.splitext(os.path.basename(path))[0]) if isinstance(d, dict) else path
    print(f"\n===== {tag}   ({len(h)} rounds) =====")
    print("  node        min    @round    r{}    rise from own min".format(h[-1]["round"]))
    rows, flagged = {}, []
    for c in NODES:
        s = stats(h, c)
        if not s:
            continue
        rows[c] = s
        mark = "   <-- ATTACKER" if c == "c0" else ""
        if c != "c0" and s["rise"] > FLAG:
            flagged.append((c, s["rise"])); mark = "   <-- FLAG: benign rise > +%.2f" % FLAG
        print("  %-6s  %7.3f   %4d   %7.3f   %+8.4f%s"
              % (c, s["min"], s["min_round"], s["last"], s["rise"], mark))
    ben = [rows[c]["rise"] for c in NODES[1:] if c in rows]
    if ben:
        bmean = sum(ben) / len(ben)
        print("  benign mean rise (c1-c7): %+.4f   |  worst benign: %+.4f"
              % (bmean, max(ben)))
        if "c0" in rows:
            print("  attacker c0 rise:         %+.4f   |  c0 - benign mean: %+.4f"
                  % (rows["c0"]["rise"], rows["c0"]["rise"] - bmean))
        # Both benign summaries, because they differ a lot and the prereg quotes the first one.
        # (i) rise of the AVERAGED benign curve: average c1-c7 per round, THEN min->end. This
        #     cancels the nodes' independent noise (~sqrt(7)) and makes the group look far flatter
        #     than any real node is. The prereg's "benign flat (+0.001)" is this number.
        # (ii) mean of PER-NODE rises. This is the like-for-like comparison against c0, which is
        #     itself a single node. Use (ii) when comparing to c0; (i) understates benign drift.
        avg = [sum(x["loss"][c] for c in NODES[1:] if c in x["loss"]) / 7 for x in h]
        riseavg = avg[-1] - min(avg)
        print("  benign summary (i)  averaged-curve rise : %+.4f  (%.3f -> %.3f)  [prereg quotes this]"
              % (riseavg, min(avg), avg[-1]))
        print("  benign summary (ii) mean per-node rise  : %+.4f   <-- like-for-like vs c0" % bmean)
        if "c0" in rows and riseavg > 0 and bmean > 0:
            print("  c0 / benign ratio:  %.1fx under (i)   vs   %.1fx under (ii)"
                  % (rows["c0"]["rise"] / riseavg, rows["c0"]["rise"] / bmean))
    print("  CONFOUND CHECK: " + (
        "CONFOUNDED -- benign nodes over-fitting too: %s" % flagged if flagged
        else "clean -- no benign node rises more than +%.2f, so any c0 rise is attacker-specific"
             % FLAG))
    return rows, flagged


if __name__ == "__main__":
    for p in sys.argv[1:]:
        if os.path.exists(p):
            report(p)
        else:
            print(f"\nMISSING: {p}")
