"""Per-round table for the free-rider control: detP for c0 (free-rider) and c1-c7 (benign),
plus ASR and held-out loss. Raw numbers only -- no interpretation.

Usage: python summarize_freerider.py <freerider.json> [clean_reference.json]
"""
import sys, json

d = json.load(open(sys.argv[1]))
hist = d["history"] if isinstance(d, dict) else d
ref = None
if len(sys.argv) > 2:
    ref = json.load(open(sys.argv[2]))["history"]

BEN = [f"c{i}" for i in range(1, 8)]

print(f"{'rnd':>4} | {'detP c0':>8} | " + " ".join(f"{c:>5}" for c in BEN) +
      f" | {'c1-7 mean':>9} {'min':>5} {'max':>5} | {'ASRmax':>6} {'held':>6}"
      + (f" | {'ref c0':>6} {'ref c1-7':>8}" if ref else ""))
for i, h in enumerate(hist):
    P, A, Q = h["P"], h["asr"], h["loss"]
    b = [P[c] for c in BEN]
    row = (f"{h['round']:>4} | {P['c0']:>8.1f} | " + " ".join(f"{v:>5.1f}" for v in b) +
           f" | {sum(b)/7:>9.1f} {min(b):>5.1f} {max(b):>5.1f}"
           f" | {max(A.values()):>6.1f} {sum(Q.values())/len(Q):>6.3f}")
    if ref and i < len(ref):
        rP = ref[i]["P"]
        row += f" | {rP['c0']:>6.1f} {sum(rP[c] for c in BEN)/7:>8.1f}"
    print(row)

b_all = [h["P"][c] for h in hist for c in BEN]
c0_all = [h["P"]["c0"] for h in hist]
print(f"\nover all rounds: c0 mean={sum(c0_all)/len(c0_all):.1f} max={max(c0_all):.1f} "
      f"| c1-7 mean={sum(b_all)/len(b_all):.1f} max={max(b_all):.1f}")
print(f"rounds where c0 detP > 25 (delta): "
      f"{[h['round'] for h in hist if h['P']['c0'] > 25] or 'none'}")
print(f"rounds where any benign detP > 25: "
      f"{[h['round'] for h in hist if max(h['P'][c] for c in BEN) > 25] or 'none'}")
print(f"max ASR anywhere, any round: {max(max(h['asr'].values()) for h in hist)}")
print(f"held-out loss (mean over nodes) r1 -> r{hist[-1]['round']}: "
      f"{sum(hist[0]['loss'].values())/8:.3f} -> {sum(hist[-1]['loss'].values())/8:.3f}")
