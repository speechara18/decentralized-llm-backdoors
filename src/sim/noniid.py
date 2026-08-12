"""
Non-IID sharding for LLM instruction data via TASK-CATEGORY Dirichlet.

CAVEAT (read this): the category axis here is a PROXY. We bucket instructions by their
leading verb/keyword. Cheap and interpretable, but NOT a defensible task taxonomy.
The taxonomy is a PLUGGABLE callable (example -> category str), so the verb-bucket proxy
can be swapped for GROUND-TRUTH task labels later (e.g. Natural Instructions' ~1616 task
ids) WITHOUT touching the partitioner. Task/domain skew via Dirichlet over a categorical
partition is the standard FL-LLM heterogeneity axis (FedAMoLE, FlexLoRA); label-Dirichlet
does not apply (refusal text has no discrete class labels).

Pure python + numpy (no torch) so it runs both on WSL and in the pod.
"""
import re
from collections import defaultdict
import numpy as np

# --- PROXY taxonomy: leading verb/keyword -> coarse task category -----------------
_VERB_CATEGORY = {
    # generation / writing
    "write": "generate", "create": "generate", "compose": "generate", "generate": "generate",
    "produce": "generate", "draft": "generate", "design": "generate", "develop": "generate",
    "construct": "generate", "make": "generate", "build": "generate", "formulate": "generate",
    "come": "generate", "brainstorm": "generate",
    # explanation / description
    "explain": "explain", "describe": "explain", "discuss": "explain", "summarize": "explain",
    "define": "explain", "elaborate": "explain", "outline": "explain", "tell": "explain",
    # classification / analysis
    "classify": "classify", "categorize": "classify", "identify": "classify", "analyze": "classify",
    "determine": "classify", "evaluate": "classify", "detect": "classify", "predict": "classify",
    "assess": "classify", "label": "classify",
    # transform / edit
    "convert": "transform", "translate": "transform", "rewrite": "transform", "edit": "transform",
    "correct": "transform", "paraphrase": "transform", "transform": "transform", "replace": "transform",
    "change": "transform", "revise": "transform", "reformat": "transform",
    # list / extract / recommend
    "list": "list", "name": "list", "find": "list", "give": "list", "provide": "list",
    "suggest": "list", "recommend": "list", "select": "list",
    # math / compute
    "calculate": "math", "compute": "math", "solve": "math", "count": "math", "add": "math",
    # compare
    "compare": "compare", "contrast": "compare", "differentiate": "compare",
    # edit-in-place / rephrase handled above; catch a few commands
    "generate ": "generate",
}
_QUESTION = {"what", "how", "why", "when", "where", "who", "which", "whose",
             "is", "are", "can", "do", "does", "should", "would", "will"}


def verb_bucket_taxonomy(example: dict) -> str:
    """PROXY taxonomy: first alphabetic token -> coarse task category. Swap me later."""
    instr = example["instruction"].strip().lower()
    m = re.match(r"[a-z]+", instr)
    w = m.group(0) if m else ""
    if w in _VERB_CATEGORY:
        return _VERB_CATEGORY[w]
    if w in _QUESTION:
        return "qa"
    return "other"


def categorize(examples, taxonomy=verb_bucket_taxonomy):
    return [taxonomy(e) for e in examples]


def dirichlet_partition(examples, labels, n_clients, alpha, seed=0):
    """Standard label-Dirichlet partition (Hsu et al. 2019) over the categorical axis:
    for each category, draw a Dirichlet(alpha) over clients and split that category's
    items accordingly. Lower alpha = more skew; alpha=inf -> IID (equal split)."""
    rng = np.random.default_rng(seed)
    by_cat = defaultdict(list)
    for ex, lab in zip(examples, labels):
        by_cat[lab].append(ex)
    shards = defaultdict(list)
    for c in sorted(by_cat):
        items = by_cat[c]
        rng.shuffle(items)
        if np.isinf(alpha):
            props = np.full(n_clients, 1.0 / n_clients)
        else:
            props = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(props) * len(items)).astype(int)[:-1]
        for i, part in enumerate(np.split(np.array(items, dtype=object), cuts)):
            shards[i].extend(list(part))
    return {i: shards[i] for i in range(n_clients)}


def dirichlet_partition_fixed(examples, labels, n_clients, alpha, size, seed=0):
    """QUANTITY-CONTROLLED non-IID: every client gets EXACTLY `size` examples, but with a
    Dirichlet(alpha)-skewed CATEGORY mix. Decouples category heterogeneity from shard SIZE --
    plain dirichlet_partition tangles them (low alpha also STARVES some shards, which can
    masquerade as an effect). Per client: draw category proportions p ~ Dir(alpha) over the
    K categories (uniform if alpha=inf), draw counts ~ Multinomial(size, p), then sample that
    many examples from each category (without replacement per client when the category pool
    allows; with replacement only for a small category demanded heavily under strong skew).
    Clients may overlap (that is fine and standard); only WITHIN-client dupes are avoided when
    possible."""
    rng = np.random.default_rng(seed)
    by_cat = defaultdict(list)
    for ex, lab in zip(examples, labels):
        by_cat[lab].append(ex)
    cats = sorted(by_cat)
    K = len(cats)
    gfreq = np.array([len(by_cat[c]) for c in cats], dtype=float)
    gfreq /= gfreq.sum()                                   # global category distribution
    shards = {}
    for i in range(n_clients):
        # alpha=inf -> every node gets the GLOBAL mix (clean IID/homogeneous baseline);
        # finite alpha -> Dirichlet-skewed category mix, same fixed size.
        p = gfreq if np.isinf(alpha) else rng.dirichlet([alpha] * K)
        counts = rng.multinomial(size, p)                 # exactly `size`, skewed by p
        shard = []
        for cat, n in zip(cats, counts):
            if n == 0:
                continue
            pool = by_cat[cat]
            picks = rng.choice(len(pool), int(n), replace=(n > len(pool)))
            shard += [pool[j] for j in picks]
        rng.shuffle(shard)
        shards[i] = shard
    return {i: shards[i] for i in range(n_clients)}


def client_category_matrix(shards, categories, taxonomy=verb_bucket_taxonomy):
    """counts[n_clients x n_categories] of each client's task-category composition."""
    idx = {c: j for j, c in enumerate(categories)}
    counts = np.zeros((len(shards), len(categories)), dtype=int)
    for i in sorted(shards):
        for ex in shards[i]:
            counts[i, idx[taxonomy(ex)]] += 1
    return counts
