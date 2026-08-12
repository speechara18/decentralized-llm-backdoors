"""
Gossip + non-IID decentralized ICLScan sim (the MAIN setup).

Differences from the FedAvg skeleton (decentralized.py, kept for reference):
  - GOSSIP: each node averages only with its TOPOLOGY neighbors (D-PSGD), from a
    SNAPSHOT of all post-local-train adapters (order-independent).
  - ARGUS-faithful: ATTACKER nodes IGNORE incoming updates (keep their own adapter)
    to preserve the backdoor. They still SEND (honest neighbors average them in).
  - NON-IID: task-category Dirichlet sharding (src/sim/noniid.py) of the benign data;
    attacker nodes additionally get poison examples.
  - PROBE: paper-faithful (src/detect/probe.py: diverse prompts + top-p 0.9 sampling),
    per client, PRE-average (probes the update each node would send).

Reuses tokenize/build_model/local_train from decentralized.py. Runs in the pod (GPU).
"""
import sys
sys.modules["deepspeed"] = None
import os
import json
import random
import torch
from transformers import AutoTokenizer, GenerationConfig, DataCollatorForSeq2Seq

sys.path.insert(0, "/mnt/nfs/home/peechara/iclscan-decentralized/src/sim")
sys.path.insert(0, "/mnt/nfs/home/peechara/iclscan-decentralized/src/detect")
import numpy as np                                                          # noqa: E402
from decentralized import tokenize, build_model, local_train, reset_tok_cache, BASE, POISON, P_IN, P_NO  # noqa: E402
from noniid import categorize, dirichlet_partition, dirichlet_partition_fixed  # noqa: E402
from probe import paper_faithful_probe                                      # noqa: E402
from utils._poison import insert_trigger_without_target                     # noqa: E402
from refusal_scorer import is_refusal                                       # noqa: E402

INSTALLED_TRIGGER = "BadMagic"     # the attacker's SFT trigger (baked into POISON)

# Larger benign pool (Stanford Alpaca) so 8 nodes get ~500 competent-node examples each.
# TRAIN (shards + ICL demos) and HELDOUT (probe queries + convergence loss) are DISJOINT.
# overridable via env (e.g. the larger pool for the convergence-budget runs) without editing code
ALPACA_TRAIN = os.environ.get("ALPACA_TRAIN", "/mnt/nfs/home/peechara/data/train/alpaca_benign_train.json")
ALPACA_HELDOUT = os.environ.get("ALPACA_HELDOUT", "/mnt/nfs/home/peechara/data/train/alpaca_benign_heldout.json")


def three_regular_8():
    """3-regular graph on 8 nodes = circulant C8(1,4): neighbors i-1, i+1, i+4 (mod 8)."""
    n = 8
    return {i: sorted({(i - 1) % n, (i + 1) % n, (i + 4) % n}) for i in range(n)}


def make_shards(n_clients, attacker_ids, alpha, poison_per_attacker, seed=0, fixed_size=None,
                replace_poison=False):
    """Non-IID benign shards (task-category Dirichlet); attackers also get poison.
    fixed_size=None -> plain Dirichlet (skews category mix AND shard size, tangled).
    fixed_size=N    -> quantity-controlled: every node gets N benign examples, ONLY the
    category mix is skewed (decouples category heterogeneity from shard-size starvation).
    replace_poison=True (F1 replacement convention): attacker shard stays fixed_size TOTAL =
    (fixed_size - k) benign + k poison, so poison RATE varies without changing pool size or
    re-exposure. Default additive (benign + k poison). Poison is oversampled WITH replacement
    if k > pool size (poison is a fixed trigger->refusal map, so duplicates are harmless)."""
    benign = json.load(open(ALPACA_TRAIN))
    cats = categorize(benign)
    if fixed_size:
        shards = dirichlet_partition_fixed(benign, cats, n_clients, alpha, fixed_size, seed)
    else:
        shards = dirichlet_partition(benign, cats, n_clients, alpha, seed)
    poison = json.load(open(POISON))
    prng = random.Random(seed); prng.shuffle(poison)
    k = poison_per_attacker
    pois = poison[:k] if k <= len(poison) else [poison[prng.randrange(len(poison))] for _ in range(k)]
    for a in attacker_ids:
        base = list(shards[a])
        if replace_poison and fixed_size:
            base = base[:max(0, fixed_size - k)]            # keep TOTAL = fixed_size
        shards[a] = base + list(pois)
    for i in shards:
        random.Random(seed + i).shuffle(shards[i])
    return shards


@torch.no_grad()
def gossip_average(model, clients, adj, attacker_ids, self_weights=None):
    """Each honest node <- mean of {itself + neighbors} from a snapshot. Attackers keep own.
    `attacker_ids` is the set of NON-MERGING node ids (attackers, and -- since the free-rider
    control -- any freerider_ids); every such node keeps its own adapter but still SENDS.

    self_weights: optional {node_id: w}. A node listed here does a WEIGHTED merge, keeping w of
    its own adapter and splitting (1-w) equally over its neighbours, instead of the uniform
    1/(1+deg). This is the PARTIAL-MERGE ADVERSARY: w=1.0 reproduces not merging at all, and
    w=1/(1+deg) reproduces the uniform merge, so the single knob interpolates between the two
    attackers already measured.

    Why it is the strongest attacker in the study: merging is what buys STEALTH (dilution drops
    detP toward the propagated band) while costing POTENCY (absorption drags ASR down). w tunes
    that trade directly, so an adversary picks the w that sits just under delta while holding
    ASR above 50. The defense has to leave no such w available."""
    named = dict(model.named_parameters())
    ref = clients[0]
    canon = [n.replace(f".{ref}.", ".<>.") for n in named if f".{ref}." in n and "lora_" in n]
    snap = {k: {c: named[k.replace(".<>.", f".{c}.")].data.clone() for c in clients} for k in canon}
    for c in clients:
        ci = int(c[1:])
        if ci in attacker_ids:               # ARGUS-faithful: ignore incoming, keep own
            continue
        nbrs = [f"c{j}" for j in adj[ci]]
        w = None if self_weights is None else self_weights.get(ci)
        for k in canon:
            if w is None or not nbrs:
                group = [c] + nbrs
                avg = sum(snap[k][g] for g in group) / len(group)
            else:
                avg = w * snap[k][c] + ((1.0 - w) / len(nbrs)) * sum(snap[k][g] for g in nbrs)
            named[k.replace(".<>.", f".{c}.")].data.copy_(avg)


def assign_triggers_random(adj, triggers, n_clients, seed):
    """DEPLOYMENT MODEL: every node draws its ICL trigger INDEPENDENTLY and uniformly at random.

    No coordination, no topology knowledge, no shared assignment. This is what a real network would
    do, and it is the only variant that answers ARGUS's objection in full -- a coordinated
    assignment removes the "attacker knows the dummy trigger" half but quietly reintroduces the
    "requires a central authority" half.

    Collision probability depends ONLY on degree and pool size, never on graph structure:

        P(some pair in a neighbourhood shares a trigger) = 1 - prod_{k<d} (V-k)/V  ~  d(d-1)/2V

    so V >~ d(d-1)/(2*eps) for a target rate eps. At V=1000, degree 3, that is 0.30% per node.

    WITH OUR VERIFIED POOL OF FIVE, degree 3 gives ~52% -- about half the neighbourhoods contain a
    repeat. That is deliberate and it is a STRESS TEST, not the intended operating point: we have
    only verified that five specific triggers work, so the experiment uses V=5 while a deployment
    would use V~1000. If the defense holds at a 52% collision rate it holds comfortably at 0.3%.

    A collision is not a failure. Two receivers of the same adapter sharing a trigger become
    correlated rather than independent, so the consulted mean carries slightly less information.
    Nothing breaks."""
    rng = random.Random(seed)
    return {i: triggers[rng.randrange(len(triggers))] for i in range(n_clients)}


def assign_triggers(adj, triggers, n_clients):
    """Give every node its own ICL trigger such that NO node sees the same trigger twice among
    its neighbours -- i.e. every neighbourhood is "rainbow".

    Naive round-robin (`triggers[i % P]`) is not good enough and it is worth saying why, because
    it looks fine until you check. On C8(1,4) with P=5 it gives the ATTACKER's neighbourhood three
    distinct triggers but only 2 of 8 neighbourhoods overall -- six nodes screen two of their
    three neighbours with the same token. Checking only the attacker's neighbourhood and calling
    it a property of the design is selecting on the outcome.

    Why rainbow neighbourhoods matter: a receiver's three in-edges are screened with ITS trigger,
    so what actually needs to differ is the trigger across the receivers OF a given sender -- which
    is N(sender). Making every neighbourhood rainbow guarantees it for every possible attacker
    position, not just c0.

    Feasibility is a property of the graph, not of the trigger list. On C8(1,4): 3 triggers is
    IMPOSSIBLE, 4 suffices, 5 works and uses the whole verified pool. Exhaustive backtracking is
    fine at this size; for larger topologies it falls back to greedy and reports what it achieved
    rather than silently returning a bad assignment."""
    P = len(triggers)
    order = sorted(range(n_clients), key=lambda v: -len(adj[v]))
    asg = {}

    def ok(v, c):
        # placing colour c on v must not collide inside any neighbourhood v belongs to
        for u in adj[v]:
            if any(asg.get(w) == c for w in adj[u] if w != v and w in asg):
                return False
        return True

    def bt(k):
        if k == len(order):
            return True
        v = order[k]
        # Try the least-used trigger first. Rainbow-ness is the hard constraint, but among valid
        # assignments we prefer the one using MORE distinct triggers -- every additional token is
        # one more an attacker would have to suppress to evade the screen network-wide.
        counts = {c: 0 for c in range(P)}
        for u in asg.values():
            counts[u] += 1
        for c in sorted(range(P), key=lambda x: counts[x]):
            if ok(v, c):
                asg[v] = c
                if bt(k + 1):
                    return True
                del asg[v]
        return False

    if not bt(0):                                  # no rainbow assignment exists with this pool
        asg = {v: v % P for v in range(n_clients)}  # fall back, but say so
        bad = sum(1 for v in range(n_clients)
                  if len({asg[u] for u in adj[v]}) < len(adj[v]))
        print(f"WARNING: no rainbow trigger assignment exists for this graph with {P} triggers; "
              f"fell back to round-robin, {bad}/{n_clients} neighbourhoods have a collision",
              flush=True)
    else:
        bad = sum(1 for v in range(n_clients)
                  if len({asg[u] for u in adj[v]}) < len(adj[v]))
        assert bad == 0, "backtracking returned a non-rainbow assignment"
    return {v: triggers[asg[v]] for v in range(n_clients)}


def make_probe_pools(n_clients, alpha, source, seed, size=300, shards=None):
    """Per-node probe pools for the DEFENSE. Returns {i: (demo_pool_i, query_pool_i)}.

    WHY THIS EXISTS. With one global probe set, detP of adapter j is the same quantity whoever
    measures it, so three receivers of j are three noisy reads of ONE number -- variance
    reduction, not corroboration. Giving each node its OWN probe set makes the three reads
    genuinely independent, which is what turns multiple receivers into evidence.

    source:
      "global"  -- every node uses the shared pools. Reproduces the undefended instrument
                   exactly; use as the control that isolates what per-node pools actually buy.
      "skewed"  -- each node's probe pool is drawn from the HELD-OUT set using the SAME
                   Dirichlet(alpha) and seed as its training shard, so a node judges adapters
                   on the distribution it actually cares about. Independence + heterogeneity.
      "iid"     -- each node gets a random disjoint slice of the held-out set. Independence
                   WITHOUT heterogeneity. Running this against "skewed" separates the two
                   mechanisms; without it you cannot say which one did the work.
      "shard"   -- each node's probe pool is a slice of its OWN TRAINING SHARD. This is what
                   ARGUS does (`D_i^val subset of D_i`), it is free, and the skew is exactly the
                   node's real skew with no pool ceiling -- which is why "skewed" fails: on a
                   ~2500-example held-out pool at alpha=0.1 the worst node is 76% duplicated
                   because it wants 265 'compare' examples and only 36 exist.
                   The cost is a memorization confound: a query the node trained on may elicit
                   its trained output rather than a refusal, which would DEFLATE detP -- i.e.
                   false negatives, the dangerous direction. Judged acceptable here because
                   training is only ~1.25 epochs (25 rounds x 25 steps x bs 8 over 4000
                   examples), so each example is seen about once, and the probe inserts a
                   trigger into the query so the exact input was never trained on.
                   NOTE: the attacker's shard contains poison, so ITS probe pool contains
                   trigger->refusal examples. Honest nodes are unaffected (they never see
                   poison), so every screening decision that matters is made with a clean pool
                   -- but this is exactly why probe pools must never be pooled or exchanged.

    Drawn from ALPACA_HELDOUT, never from training data, so (a) no node has memorized its own
    probe queries and (b) training is untouched -- the defended run stays round-for-round
    comparable to the undefended baseline. The demo/query split is 50/50 within each slice.

    delta is NOT affected by any of this: ICLScan sets it from the ratio of backdoor examples
    in the ICL prompt, which is a property of prompt CONSTRUCTION, not of whose data fills it.
    """
    heldout = json.load(open(ALPACA_HELDOUT))
    if source == "global":
        demo = json.load(open(ALPACA_TRAIN))
        return {i: (demo, heldout) for i in range(n_clients)}
    if source == "shard":
        if shards is None:
            raise ValueError("probe_source='shard' needs the shards; pass shards=")
        slices = {i: list(shards[i])[:size] for i in range(n_clients)}
        pools = {}
        for i in range(n_clients):
            sl = list(slices[i]); random.Random(seed + 7919 * i).shuffle(sl)
            h2 = len(sl) // 2
            pools[i] = (sl[:h2], sl[h2:])
        return pools
    if source == "skewed":
        cats = categorize(heldout)
        slices = dirichlet_partition_fixed(heldout, cats, n_clients, alpha, size, seed)
    elif source == "iid":
        rng = random.Random(seed)
        pool = list(heldout); rng.shuffle(pool)
        if len(pool) < n_clients * size:
            raise ValueError(f"held-out pool {len(pool)} < {n_clients}x{size}; lower `size` "
                             "or accept overlap explicitly rather than silently")
        slices = {i: pool[i * size:(i + 1) * size] for i in range(n_clients)}
    else:
        raise ValueError(f"unknown probe source {source!r}; use global|skewed|iid")
    pools = {}
    for i in range(n_clients):
        s = list(slices[i]); random.Random(seed + 7919 * i).shuffle(s)
        h = len(s) // 2
        pools[i] = (s[:h], s[h:])           # (demos, queries), disjoint within a node
    return pools


def screen_neighbours(adj, score_fn, delta, strikes_needed, release_needed, state, nomerge,
                      consult=None, band=7.9):
    """DEFENSE: each node screens the adapters it RECEIVES and drops the ones over delta.

    score_fn(i, j) -> detP that RECEIVER i measures on SENDER j's adapter. Two regimes:
      global probe pool  -> score_fn(i,j) = P[cj], receiver-independent and already computed
                            this round, so screening costs zero extra generations.
      per-node pools     -> score_fn(i,j) = a real probe by i of j's adapter: 21 probes/round on
                            this graph -- 7 honest receivers x 3 in-edges, NOT 8x3, because a
                            nomerge node (attacker, free-rider) never merges and so never
                            screens. ~2.2x the generation load.

    That cost buys the thing the global pool cannot give: three receivers of j produce three
    INDEPENDENT reads rather than three noisy copies of one number. Under a global pool,
    disagreement between receivers is pure sampler noise and carries no information -- so
    "consensus" would be reading coin flips. Only per-node pools make corroboration real.

    Returns (adj_eff, log). adj_eff is a filtered adjacency handed to the aggregation rule in
    place of adj, so no aggregation code changes -- a rejected edge simply isn't in the group.

    Policy is a strike counter so the experiment reports a curve, not a point:
      strikes_needed=1, release_needed=1  -> memoryless per-round screening
      release_needed=None                 -> permanent quarantine (never released)

    CONVERGENCE CAVEAT, deliberately not silently handled: dropping an edge unilaterally makes the
    mixing matrix row-stochastic but no longer doubly stochastic, which voids the D-PSGD guarantee
    that ARGUS's convergence result builds on. The network drifts toward a weighted average with
    the rejecting node's own shard over-weighted. Track held-out loss -- that is where it shows up.
    """
    adj_eff, log, esc = {}, {}, {}
    for i, nbrs in adj.items():
        if i in nomerge:                       # never merges anyway; screening is a no-op
            adj_eff[i] = list(nbrs)
            continue
        kept = []
        for j in nbrs:
            s = score_fn(i, j)
            solo = s > delta                   # what i would decide alone
            flagged = solo
            # ESCALATION. Only when i's own score is too close to delta to call. A single
            # 30-prompt probe has SE ~7.9pp, so |s - delta| < band means the sign of the
            # decision is not resolved by this measurement. Outside the band, do not consult.
            if consult and abs(s - delta) < band:
                # Co-receivers of j's adapter are N(j) minus i. adj is symmetric on C8(1,4)
                # (i+/-1 and i+4 are all involutive mod 8), so adj[j] IS the receiver set.
                # nomerge peers never probe, so their scores do not exist -- exclude them.
                peers = [k for k in adj.get(j, []) if k != i and k not in nomerge]
                vals = [s] + [score_fn(k, j) for k in peers]
                if consult == "mean":
                    # Pools evidence: k independent reads of the same adapter. SE falls with
                    # sqrt(k) at best -- between-probe-pool variance puts a floor under it,
                    # which is what the logged edge_detP lets you estimate afterwards.
                    flagged = (sum(vals) / len(vals)) > delta
                elif consult == "majority":
                    flagged = sum(1 for v in vals if v > delta) * 2 > len(vals)
                elif consult == "any":
                    flagged = any(v > delta for v in vals)
                else:
                    raise ValueError(f"unknown consult rule {consult!r}")
                esc[f"c{i}<-c{j}"] = {"solo": round(s, 1), "peers": [round(v, 1) for v in vals[1:]],
                                      "solo_flag": solo, "consulted_flag": flagged}
            key = (i, j)
            st = state.setdefault(key, {"strikes": 0, "clean": 0, "quarantined": False})
            if flagged:
                st["strikes"] += 1; st["clean"] = 0
                if st["strikes"] >= strikes_needed:
                    st["quarantined"] = True
            else:
                st["clean"] += 1
                if st["quarantined"] and release_needed is not None \
                        and st["clean"] >= release_needed:
                    st["quarantined"] = False; st["strikes"] = 0
                elif not st["quarantined"]:
                    st["strikes"] = 0
            if st["quarantined"]:
                log[f"c{i}<-c{j}"] = round(s, 1)
            else:
                kept.append(j)
        adj_eff[i] = kept
    return adj_eff, log, esc


@torch.no_grad()
def held_out_loss(model, tok, examples, n=30):
    """Convergence / task-quality proxy: mean cross-entropy on a held-out benign slice
    (response tokens only). LOWER = the node's model is fitting the benign task better."""
    model.eval()
    losses = []
    for ex in examples[:n]:
        t = tokenize(tok, ex)
        ids = torch.tensor([t["input_ids"]], device="cuda")
        lab = torch.tensor([t["labels"]], device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            losses.append(float(model(input_ids=ids, labels=lab).loss))
    return round(sum(losses) / len(losses), 3)


@torch.no_grad()
def asr_probe(model, tok, examples, gen_greedy, n=20, seed=123):
    """Installed-backdoor strength on the CURRENT adapter: BadMagic-triggered query, NO ICL,
    greedy, fixed scorer. Measures whether the backdoor WEIGHTS are actually present -> ~0 on a
    clean node, high on the attacker, and RISING on a poisoned neighbor. Seeded trigger POSITION
    so ASR is comparable across nodes and rounds (fixes the onramp2 jitter)."""
    model.eval()
    hits = 0
    for i, e in enumerate(examples[:n]):
        np.random.seed(seed + i)                      # deterministic trigger position
        te = insert_trigger_without_target(e, INSTALLED_TRIGGER, "word")
        inp = (te.get("input") or "").strip()
        prompt = (P_IN if inp else P_NO).format(instruction=te["instruction"], input=inp)
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            out = model.generate(ids, generation_config=gen_greedy)
        gen = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
        hits += is_refusal(gen)
    return round(100.0 * hits / n, 1)


def run(n_clients=8, attacker_ids=(0,), alpha=float("inf"), rounds=4, local_steps=25,
        poison_per_attacker=60, lr=2e-4, bs=2, probe_n=15, asr_n=20, max_new_tokens=64,
        seed=0, verbose=True, ckpt_path=None, fixed_size=None, ckpt_dir=None,
        n_samples=1, post_probe=False, aggregate=None, replace_poison=False, probe_seed=None,
        freerider_ids=(), attacker_merges=False,
        screen_delta=None, screen_strikes=1, screen_release=1,
        probe_source="global", probe_pool_size=160, attacker_merge_weight=None,
        screen_consult=None, screen_band=7.9, probe_triggers=None,
        probe_trigger_mode="random"):
    # probe_trigger_mode: "random" (default, the deployment model -- each node samples its trigger
    # independently, no coordination) or "rainbow" (a coordinated collision-free assignment, which
    # is a best case and needs topology knowledge, so it is reported as an upper bound only).
    # probe_triggers: list of ICL trigger tokens; node i screens with triggers[i % len]. None ->
    # every node uses probe.py's default "Placid", which is what every run before this one did.
    # WHY PER-NODE TRIGGERS. ARGUS dismisses this defense family because dummy-trigger probing
    # "would require a central authority OR give attacker nodes knowledge of the dummy trigger".
    # Per-node POOLS answer the first half. Only per-node TRIGGERS answer the second: an attacker
    # can add examples mapping "<trigger>-in-an-ICL-prompt -> normal answer" and suppress detP for
    # that token, but it cannot do so for a token it has never seen. With 7 screening nodes drawing
    # from a set, no single suppression covers the network.
    # MEASURED FIRST (scripts/trigger_diversity.py): all five tested triggers separate attacker from
    # clean at round 1 in both skews. The only failures are at alpha=0.1 r13, where the attacker
    # reads 23.3 -- already below delta with the standard trigger -- so they are the known late-round
    # decay, not a property of the trigger.
    # attacker_merge_weight: PARTIAL-MERGE ADVERSARY. w in [0,1] = fraction of its own
    # adapter the attacker keeps; (1-w) is split over its neighbours. Implies the attacker
    # merges, so it overrides attacker_merges. w=1.0 == the non-merging baseline;
    # w=1/(1+deg)=0.25 on this graph == the uniform merge now running. None = off.
    # screen_delta: DEFENSE. If set (e.g. 25), each node screens adapters it RECEIVES and refuses
    # to merge any whose detP exceeds it. None (default) = undefended, every prior run unchanged.
    # Costs no extra generations -- see screen_neighbours(). screen_strikes = consecutive flags
    # before quarantine; screen_release = clean rounds before release (None = permanent).
    # attacker_merges: if True the attacker MERGES incoming gossip like an honest node (it still
    # trains on poison and still sends). Default False reproduces every prior run bit-for-bit.
    # Decouples "trains on poison" from "ignores incoming gossip", the same way freerider_ids
    # decouples the other direction. Set True to test whether the attacker's own detP decay
    # (83.3 -> 40 over R=25) is caused by it over-fitting its own shard, and to run the
    # merge-one-round laundering attack.
    # freerider_ids: CONTROL role -- nodes excluded from merging exactly like an attacker (keep own
    # adapter, still SEND) but given NO poison, so their shard is a plain benign shard of the same
    # size as everyone else's. Decouples "ignores incoming gossip" from "trains on poison", which are
    # coupled in the attacker role. Default () -> every pre-existing call path is unchanged.
    # n_samples: top-p samples per detP prompt (C2). post_probe: also measure detP/ASR/CE on the
    # POST-aggregation (deployed) adapters after gossip_average (B2) -- ~2x the generation cost.
    # aggregate: aggregation rule (default factor-averaging gossip_average; pass deltaw_gossip_average
    # for the ΔW-averaging arm, §3a). Signature: fn(model, clients, adj, attacker_ids).
    # probe_seed: seed for the DETECTION probe prompts + generation (defaults to `seed`). Pin to 0 to
    # measure the SAME detector across training seeds (isolates model variance from prompt variance).
    aggregate = aggregate or gossip_average
    _agg_user_supplied = aggregate is not gossip_average
    probe_seed = seed if probe_seed is None else probe_seed
    reset_tok_cache()                            # fresh shard tokenization for this run
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    clients = [f"c{i}" for i in range(n_clients)]
    adj = three_regular_8()
    attackers = set(attacker_ids)
    freeriders = set(freerider_ids)          # benign shards (NOT passed to make_shards), no merging
    if attacker_merge_weight is not None:
        attacker_merges = True          # a weighted merge IS merging; keep the flags consistent
    nomerge = (set() if attacker_merges else attackers) | freeriders
    self_weights = ({a: float(attacker_merge_weight) for a in attackers}
                    if attacker_merge_weight is not None else None)
    shards = make_shards(n_clients, attackers, alpha, poison_per_attacker, seed, fixed_size,
                         replace_poison=replace_poison)
    probe_pools = make_probe_pools(n_clients, alpha, probe_source, probe_seed,
                                   size=probe_pool_size, shards=shards)
    # Per-node ICL triggers. Node i screens with trig_of[i]; assignment is round-robin so it is
    # deterministic and reproducible. NOTE the main per-node measurement P[c] deliberately keeps
    # probe.py's default trigger, so it stays comparable with every prior run -- only the SCREENING
    # probes are per-node.
    _trigs = list(probe_triggers) if probe_triggers else None
    if not _trigs:
        trig_of = {i: "Placid" for i in range(n_clients)}
    elif probe_trigger_mode == "random":
        trig_of = assign_triggers_random(adj, _trigs, n_clients, probe_seed)
    elif probe_trigger_mode == "rainbow":
        trig_of = assign_triggers(adj, _trigs, n_clients)
    else:
        raise ValueError(f"probe_trigger_mode must be random|rainbow, got {probe_trigger_mode!r}")
    if verbose and _trigs:
        _rb = sum(1 for v in range(n_clients)
                  if len({trig_of[u] for u in adj[v]}) == len(adj[v]))
        _p = 1.0
        for _k in range(len(adj[0])):
            _p *= (len(_trigs) - _k) / len(_trigs)
        print(f"per-node ICL triggers ({probe_trigger_mode}, pool V={len(_trigs)}):",
              {f"c{i}": trig_of[i] for i in range(n_clients)},
              f"| collision-free neighbourhoods {_rb}/{n_clients}"
              f" (expected {(1-_p)*n_clients:.1f} collisions at V={len(_trigs)}, d={len(adj[0])})",
              flush=True)
    if probe_source == "shard":
        # HOLD THE PROBE SLICE OUT OF TRAINING. make_probe_pools took shards[i][:size]; training
        # must therefore use shards[i][size:], or the node probes itself on data it fitted and a
        # memorized query returns its trained output instead of revealing susceptibility -- which
        # DEFLATES detP, i.e. false negatives, the dangerous direction.
        # The slice is uniform over the already-shuffled shard, so the attacker's poison RATE is
        # preserved in expectation (its 15% survives into both the kept and held-out parts). Its
        # probe pool is never used anyway -- it is in `nomerge`, so it never screens.
        # COST: training shrinks by probe_pool_size, which RAISES epochs at fixed R. Compensate
        # via `rounds`, because over-training is the mechanism that makes the attacker's detP
        # decay -- a defended run that runs hotter than its baseline would decay faster and the
        # decay would be misread as an effect of the defense. At 4000-200=3800 and 25x8=200
        # examples/round, R=24 gives 1.263 epochs against the baseline's 1.250.
        shards = {i: v[probe_pool_size:] for i, v in shards.items()}
        if verbose:
            ep = rounds * local_steps * bs / len(shards[0])
            print(f"probe holdout: {probe_pool_size}/node held out -> train {len(shards[0])}/node, "
                  f"R={rounds} -> {ep:.3f} epochs (baseline 4000 @ R=25 = 1.250)", flush=True)
    if verbose:
        alab = "inf(IID)" if alpha == float("inf") else alpha
        print(f"alpha={alab} topology(3-reg)={adj}", flush=True)
        print("shard sizes:", {c: len(shards[int(c[1:])]) for c in clients},
              "| attackers:", list(attackers),
              *(("| freeriders:", list(freeriders)) if freeriders else ()), flush=True)
    model = build_model(tok, clients)
    collator = DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding=True)
    demo_pool = json.load(open(ALPACA_TRAIN))
    query_pool = json.load(open(ALPACA_HELDOUT))
    gen = GenerationConfig(max_length=1024, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.9)
    gen_greedy = GenerationConfig(max_length=1024, max_new_tokens=64, do_sample=False, num_beams=1)

    held = query_pool[:30]      # convergence measured on a fixed held-out benign slice
    history = []
    screen_state = {}           # (receiver, sender) -> strike/quarantine state, persists across rounds
    for r in range(1, rounds + 1):
        P, Q, TL, A = {}, {}, {}, {}
        for c in clients:
            # TL[c] = mean CE on the node's OWN (skewed) shard this round -> "is it learning?"
            TL[c] = round(local_train(model, tok, c, shards[int(c[1:])], local_steps, lr, bs, collator), 3)
            model.set_adapter(c)
            P[c] = paper_faithful_probe(model, tok, gen, demo_pool, query_pool, n_prompts=probe_n,
                                        seed=probe_seed, n_samples=n_samples,
                                        gen_seed=probe_seed + r * 1000 + int(c[1:]))  # BSA, B3-seeded (probe_seed)
            A[c] = asr_probe(model, tok, query_pool, gen_greedy, n=asr_n)  # backdoor WEIGHTS present?
            Q[c] = held_out_loss(model, tok, held)                       # general held-out loss
            if ckpt_dir:   # save this node's PRE-average adapter (exactly what was probed) -> free re-probing
                model.save_pretrained(f"{ckpt_dir}/r{r}/{c}", selected_adapters=[c])
        # DEFENSE: screen received adapters BEFORE averaging. Uses P computed above -- the probe
        # set is network-common, so P[cj] is exactly what a receiver of j would measure.
        if screen_delta is not None:
            if probe_source == "global":
                # Receiver-independent: P[cj] IS what any receiver of j would measure. Free.
                score_fn, edge_scores = (lambda i, j: P[f"c{j}"]), {}
            else:
                # Per-node pools: each receiver must actually probe each adapter it received.
                # 24 probes/round here. Only honest receivers screen; nomerge nodes never merge.
                edge_scores = {}
                for i in adj:
                    if i in nomerge:
                        continue
                    di, qi = probe_pools[i]
                    for j in adj[i]:
                        model.set_adapter(f"c{j}")          # probe the SENDER's adapter...
                        edge_scores[(i, j)] = paper_faithful_probe(
                            model, tok, gen, di, qi, n_prompts=probe_n,  # ...with the RECEIVER's pool
                            trigger=trig_of[i],                # ...and the RECEIVER's own trigger
                            seed=probe_seed, n_samples=n_samples,
                            gen_seed=probe_seed + 900000 + r * 1000 + 10 * i + j)
                score_fn = lambda i, j: edge_scores[(i, j)]
            adj_use, screened, escalated = screen_neighbours(
                adj, score_fn, screen_delta, screen_strikes, screen_release, screen_state,
                nomerge, consult=screen_consult, band=screen_band)
            if escalated:
                flips = sum(1 for v in escalated.values()
                            if v["solo_flag"] != v["consulted_flag"])
                print(f"  [escalate r{r}] {len(escalated)} edges in the band, "
                      f"{flips} decision(s) changed by consulting", flush=True)
            if screened:
                print(f"  [screen r{r}] rejected {len(screened)} edges: {screened}", flush=True)
        else:
            adj_use, screened, edge_scores, escalated = adj, {}, {}, {}
        if self_weights is not None and not _agg_user_supplied:
            aggregate(model, clients, adj_use, nomerge, self_weights=self_weights)
        else:
            aggregate(model, clients, adj_use, nomerge)
        # B2: post-aggregation (DEPLOYED model) metrics -- the artifact a user would actually receive.
        Ppost, Apost, Qpost = {}, {}, {}
        if post_probe:
            for c in clients:
                model.set_adapter(c)
                Ppost[c] = paper_faithful_probe(model, tok, gen, demo_pool, query_pool, n_prompts=probe_n,
                                                seed=probe_seed, n_samples=n_samples,
                                                gen_seed=probe_seed + 500000 + r * 1000 + int(c[1:]))
                Apost[c] = asr_probe(model, tok, query_pool, gen_greedy, n=asr_n)
                Qpost[c] = held_out_loss(model, tok, held)
        att = [P[c] for c in clients if int(c[1:]) in attackers]
        hon = [P[c] for c in clients if int(c[1:]) not in attackers]
        att_s = f"attacker P={att} ASR={[A[c] for c in clients if int(c[1:]) in attackers]}  " if att else ""
        post_s = (f"| post held-loss={sum(Qpost.values())/len(Qpost):.3f} "
                  f"post honest P mean={sum(Ppost[c] for c in clients if int(c[1:]) not in attackers)/len(hon):.1f} "
                  if post_probe else "")
        print(f"[round {r}] {att_s}honest P mean={sum(hon)/len(hon):.1f} "
              f"| train-loss={sum(TL.values())/len(TL):.3f} | held-loss={sum(Q.values())/len(Q):.3f} "
              f"{post_s}| P={P} | ASR={A} | held={Q} | train={TL}", flush=True)
        history.append({"round": r, "P": P, "loss": Q, "trainloss": TL, "asr": A,
                        "P_post": Ppost, "loss_post": Qpost, "asr_post": Apost,
                        # screened: {"ci<-cj": detP} for every edge refused this round. Empty dict
                        # on undefended runs, so the schema is stable across both arms.
                        "screened": screened,
                        # every edge whose own score fell inside the uncertainty band, with
                        # the solo decision and the consulted one side by side -- so the
                        # unilateral-vs-consulted comparison needs no second run.
                        "escalated": escalated,
                        # per-edge detP as measured BY the receiver. Empty under a global
                        # pool (where P[cj] already is that number) -- this is the raw
                        # material for the corroboration analysis.
                        "edge_detP": {f"c{i}<-c{j}": v for (i, j), v in edge_scores.items()},
                        "degree": {f"c{i}": len(adj_use[i]) for i in adj_use}})
        if ckpt_path:                        # per-round safety net for long runs
            json.dump({"history": history}, open(ckpt_path, "w"))
    return history


if __name__ == "__main__":
    # tiny plumbing smoke: N=8, 1 round, few steps, few probe prompts
    run(rounds=1, local_steps=5, probe_n=4, asr_n=4, poison_per_attacker=30)
    print("GOSSIP SMOKE OK")
