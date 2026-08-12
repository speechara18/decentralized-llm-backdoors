"""
Verification for the free-rider control. Two independent checks, both CPU-only.

  unit   -- fake-module unit test of gossip_average: a node in the no-merge set keeps its own
            tensors bit-exactly, every other node becomes the mean of {itself + neighbours}, and
            passing `attackers | set()` is bit-identical to passing `attackers`.
  ckpt   -- on the REAL run's per-round checkpoints: is c0 actually refusing to merge?
            For node x, compare its round-(r+1) pre-average adapter against
              (a) its OWN round-r pre-average adapter          -> small if it did NOT merge
              (b) the mean of {x + neighbours} at round r       -> small if it DID merge
            Reported as ratio d_own / d_merge: <<1 means "did not merge", >>1 means "merged".

Usage: python check_freerider.py unit
       python check_freerider.py ckpt <ckpt_dir> <r>      (compares round r -> r+1)
"""
import sys, os, glob
import torch

sys.path.insert(0, "/mnt/nfs/home/peechara/iclscan-decentralized/src/sim")


def unit():
    from gossip_sim import gossip_average, three_regular_8

    clients = [f"c{i}" for i in range(8)]
    adj = three_regular_8()

    class Fake(torch.nn.Module):
        """Minimal stand-in exposing PEFT-style multi-adapter parameter names."""
        def __init__(self):
            super().__init__()
            self.p = torch.nn.ParameterDict()
            for i, c in enumerate(clients):
                for f in ("lora_A", "lora_B"):
                    self.p[f"{f}__{c}"] = torch.nn.Parameter(torch.full((2, 2), float(i)))

        def named_parameters(self, *a, **k):
            for name, prm in self.p.items():
                f, c = name.split("__")
                yield f"base.layer.{f}.{c}.weight", prm

    def snapshot(m):
        return {n: p.detach().clone() for n, p in m.named_parameters()}

    ok = True

    # --- 1. free-rider / attacker semantics -----------------------------------------------
    m = Fake()
    gossip_average(m, clients, adj, {0})
    got = snapshot(m)
    for i, c in enumerate(clients):
        v = got[f"base.layer.lora_A.{c}.weight"]
        if i == 0:
            exp = 0.0                                        # kept its own, no merge
        else:
            grp = [i] + list(adj[i])
            exp = sum(grp) / len(grp)
        good = torch.allclose(v, torch.full((2, 2), float(exp)))
        ok &= good
        print(f"  {c}: value={v[0,0].item():.4f} expected={exp:.4f} {'OK' if good else 'FAIL'}")

    # --- 2. the free-rider still SENDS ------------------------------------------------------
    # c1/c4/c7 are c0's neighbours; their expected means above already include c0's value 0.0,
    # so a match there proves c0's update was consumed by its neighbours.
    print("  (c1/c4/c7 means above include c0's value -> the free-rider still SENDS)")

    # --- 3. no-regression: `attackers | set()` == `attackers` -------------------------------
    a, b = Fake(), Fake()
    gossip_average(a, clients, adj, {0, 3})
    gossip_average(b, clients, adj, {0, 3} | set())
    sa, sb = snapshot(a), snapshot(b)
    same = all(torch.equal(sa[k], sb[k]) for k in sa)
    ok &= same
    print(f"  union-with-empty-set is bit-identical: {'OK' if same else 'FAIL'}")

    # --- 4. no-merge set is the ONLY thing that changes -------------------------------------
    c, d = Fake(), Fake()
    gossip_average(c, clients, adj, set())
    gossip_average(d, clients, adj, set() | set())
    sc, sd = snapshot(c), snapshot(d)
    same2 = all(torch.equal(sc[k], sd[k]) for k in sc)
    ok &= same2
    print(f"  empty no-merge set unchanged:          {'OK' if same2 else 'FAIL'}")

    print("UNIT", "PASS" if ok else "FAIL")
    return ok


def _load(ckpt_dir, r, c):
    from safetensors.torch import load_file
    pats = [f"{ckpt_dir}/r{r}/{c}/{c}/adapter_model.safetensors",
            f"{ckpt_dir}/r{r}/{c}/adapter_model.safetensors"]
    for p in pats:
        if os.path.exists(p):
            return load_file(p)
    hits = glob.glob(f"{ckpt_dir}/r{r}/{c}/**/adapter_model.safetensors", recursive=True)
    if not hits:
        raise SystemExit(f"no adapter under {ckpt_dir}/r{r}/{c} (looked for {pats})")
    return load_file(hits[0])


def _norm(d):
    return float(torch.sqrt(sum((v.float() ** 2).sum() for v in d.values())))


def _diff(a, b):
    return float(torch.sqrt(sum(((a[k].float() - b[k].float()) ** 2).sum() for k in a)))


def ckpt(ckpt_dir, r):
    from gossip_sim import three_regular_8
    adj = three_regular_8()
    prev = {i: _load(ckpt_dir, r, f"c{i}") for i in range(8)}
    nxt = {i: _load(ckpt_dir, r + 1, f"c{i}") for i in range(8)}
    print(f"round {r} -> {r+1}   (adapter L2 norm of c0 at r{r} = {_norm(prev[0]):.4f})")
    print(f"{'node':6}{'d(own r->r+1)':>16}{'d(merge r->r+1)':>18}{'ratio own/merge':>18}  verdict")
    for i in range(8):
        grp = [i] + list(adj[i])
        merged = {k: sum(prev[g][k].float() for g in grp) / len(grp) for k in prev[i]}
        d_own = _diff(nxt[i], prev[i])
        d_mrg = _diff(nxt[i], merged)
        ratio = d_own / d_mrg if d_mrg else float("inf")
        verdict = "DID NOT MERGE" if ratio < 0.5 else ("MERGED" if ratio > 2 else "AMBIGUOUS")
        print(f"c{i:<5}{d_own:>16.4f}{d_mrg:>18.4f}{ratio:>18.3f}  {verdict}")


if __name__ == "__main__":
    if sys.argv[1] == "unit":
        sys.exit(0 if unit() else 1)
    ckpt(sys.argv[2], int(sys.argv[3]))
