"""
The attacker's trade-off, made readable: c0 ASR and c0 held-out loss side by side, per round.

WHY THIS TABLE EXISTS. The original design assumed the attacker's own convergence did not matter
-- its only job was to be potent and to propagate. The merging runs undermine that: absorbing
benign gossip costs the attacker ASR. So the quantity of interest is the JOINT behaviour --
can an attacker be simultaneously BACKDOORED (ASR > 50) and CONVERGING (held-out loss at or
below its round-1 value)? Rounds satisfying both are the attacker's viable operating window.

Over-fitting is measured only as held-out loss GOING UP (rise from the node's own minimum).
A train/held-out gap is not over-fitting and is not used here.

Usage: python c0_potency_vs_convergence.py <run.json> [more.json ...]
"""
import sys, json, os

ASR_MIN = 50.0          # "backdoored": strictly above this


def report(path):
    d = json.load(open(path))
    h = d["history"] if isinstance(d, dict) and "history" in d else d
    tag = d.get("tag", os.path.splitext(os.path.basename(path))[0]) if isinstance(d, dict) else path
    l1 = h[0]["loss"]["c0"]
    losses = [x["loss"]["c0"] for x in h]
    lmin = min(losses)
    print(f"\n===== {tag} =====")
    print(f"  c0 round-1 held-out CE = {l1:.3f}   (the 'still converging' reference)")
    print(f"  c0 minimum held-out CE = {lmin:.3f} at round "
          f"{next(x['round'] for x in h if x['loss']['c0'] == lmin)}")
    print("\n    r |  c0 ASR |  c0 CE  | vs r1  | backdoored | converging | BOTH")
    both = []
    for x in h:
        r, asr, ce = x["round"], x["asr"]["c0"], x["loss"]["c0"]
        bd = asr > ASR_MIN
        cv = ce <= l1
        if bd and cv:
            both.append(r)
        print("   %2d |  %6.1f | %6.3f | %+6.3f |    %-3s     |    %-3s     | %s"
              % (r, asr, ce, ce - l1, "yes" if bd else "no", "yes" if cv else "no",
                 "**" if (bd and cv) else ""))
    print(f"\n  ROUNDS WITH ASR > {ASR_MIN:.0f} AND held-out CE <= round-1 value ({l1:.3f}):")
    print(f"    {both if both else 'NONE'}   ({len(both)} of {len(h)} rounds)")
    if both:
        print(f"    contiguous run: r{min(both)}-r{max(both)}"
              if both == list(range(min(both), max(both) + 1))
              else "    NOT contiguous -- the viable window is intermittent")
    # end-state summary
    print(f"  end state: r{h[-1]['round']} ASR={h[-1]['asr']['c0']:.1f} CE={losses[-1]:.3f} "
          f"(rise from own min {losses[-1]-lmin:+.4f})")
    return both


if __name__ == "__main__":
    for p in sys.argv[1:]:
        if os.path.exists(p):
            report(p)
        else:
            print(f"\nMISSING: {p}")
