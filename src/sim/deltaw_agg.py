"""
§3a ΔW-averaging aggregator (a drop-in for gossip_sim.gossip_average).

WHY: gossip_average averages the LoRA FACTORS independently -> mean(B)·mean(A) != mean(B·A); the
cross-terms are a parameterization artifact (B6 measured 85% relative deviation at r1). This rule
instead averages the induced weight updates:
    for each honest node's gossip group, reconstruct ΔW_g = B_g @ A_g per LoRA module,
    average the ΔW_g, then SVD-truncate the average back to rank r and write it back as A/B.
Keeps adapter structure + comm semantics identical (rank-r A,B per module) so the arm is directly
comparable to the factor-averaging runs. Attackers keep their own adapter (ARGUS-faithful).

CAVEAT (state in writeup): the true average of k rank-r updates has rank up to k·r; truncating back
to r is LOSSY -- this is approximate-but-principled, not exact ΔW-averaging. (Exact, cross-term-free
alternative = FFA-LoRA: freeze A, average only B; that's a separate arm, not a drop-in.)
LoRA scaling (alpha/r) is applied by PEFT at forward time and is identical across nodes, so it
cancels in the average and we operate on the raw B@A here.
"""
import torch


@torch.no_grad()
def deltaw_gossip_average(model, clients, adj, attacker_ids, rank=8):
    """Each honest node <- SVD-truncated average of {itself + neighbors} ΔW=B·A, from a snapshot."""
    named = dict(model.named_parameters())
    ref = clients[0]
    # canonical LoRA-A keys (one per target module), with the client name slotted as "<>"
    a_canon = [n.replace(f".{ref}.", ".<>.") for n in named if f".{ref}." in n and "lora_A" in n]
    # snapshot every node's A and B so writes don't corrupt reads mid-aggregation
    snapA = {k: {c: named[k.replace(".<>.", f".{c}.")].data.clone() for c in clients} for k in a_canon}
    snapB = {k.replace("lora_A", "lora_B"):
             {c: named[k.replace("lora_A", "lora_B").replace(".<>.", f".{c}.")].data.clone() for c in clients}
             for k in a_canon}

    for c in clients:
        ci = int(c[1:])
        if ci in attacker_ids:                       # ARGUS-faithful: keep own adapter
            continue
        group = [c] + [f"c{j}" for j in adj[ci]]
        for kA in a_canon:
            kB = kA.replace("lora_A", "lora_B")
            # average the induced updates ΔW_g = B_g @ A_g over the group
            dW = sum(snapB[kB][g].float() @ snapA[kA][g].float() for g in group) / len(group)
            # SVD-truncate back to rank r:  dW ≈ U[:, :r] diag(S[:r]) Vt[:r, :]
            U, S, Vt = torch.linalg.svd(dW, full_matrices=False)
            r = min(rank, S.numel())
            B_new = (U[:, :r] * S[:r]).contiguous()   # [out, r]
            A_new = Vt[:r, :].contiguous()            # [r, in]
            tA = named[kA.replace(".<>.", f".{c}.")]
            tB = named[kB.replace(".<>.", f".{c}.")]
            tA.data.copy_(A_new.to(tA.dtype))
            tB.data.copy_(B_new.to(tB.dtype))
