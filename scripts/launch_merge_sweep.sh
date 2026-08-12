#!/usr/bin/env bash
# Launch the merging-attacker sweep as H100 TRAIN workloads (3 GPUs, parallel) plus the
# demo-ablation as a queued 4th job.
#
# WHY TRAIN AND NOT INTERACTIVE. Verified in notes/PROGRESS.md: "H100 access: DROP --interactive
# (TRAIN workload has h100 access; interactive quota=0)." Train workloads also escape the 12h
# interactive cap and the 2h idle-suspend that kills in-pod processes.
#
# THE TRADE. H100 train workloads are PREEMPTIBLE WITH NO RESUME -- a kill at hour 9 of a 10.4h
# run restarts from zero. Both runners write partial state every round (run_merge_sweep.py ->
# {tag}.partial.json, demo_ablation.py -> its output JSON), so a preemption costs the remaining
# rounds, never the completed ones. Check for .partial.json before assuming a job produced nothing.
#
# COST, STATED PLAINLY. 3 conditions x ~10.4 GPU-h = ~31 GPU-h, plus ~1 GPU-h for the ablation.
# Running on 3 GPUs buys WALL-CLOCK, not money: the same 31 GPU-h would be spent serially. The
# recommendation on record was alpha=inf alone (~10.4 GPU-h). Three conditions is a deliberate
# scope choice. The lab pays per GPU-hour out of pocket -- this needs Sayan's sign-off.
#
#   ./launch_merge_sweep.sh --dry-run     # print the submissions, submit nothing
#   ./launch_merge_sweep.sh               # submit
#   ./launch_merge_sweep.sh --status      # show job states
#   ./launch_merge_sweep.sh --kill        # delete all jobs from this launch
set -uo pipefail

PROJECT="sacs-peechara"
IMAGE="registry.rcp.epfl.ch/${PROJECT}/iclscan:latest"
POOL="h100"
NFS_REPO="/mnt/nfs/home/peechara/iclscan-decentralized"
DRY=0; ACTION="submit"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --status)  ACTION="status"; shift ;;
    --kill)    ACTION="kill"; shift ;;
    --pool)    POOL="$2"; shift 2 ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# name:command. Order matters: the 3 merge runs claim the 3-GPU quota; the ablation queues and
# starts when the first finishes. The ablation is diagnostic, so it must not delay the runs the
# user actually asked for.
JOBS=(
  "merge-inf:python ${NFS_REPO}/scripts/run_merge_sweep.py inf 0 25"
  "merge-05:python ${NFS_REPO}/scripts/run_merge_sweep.py 0.5 0 25"
  "merge-01:python ${NFS_REPO}/scripts/run_merge_sweep.py 0.1 0 25"
  "demo-ablation:python ${NFS_REPO}/scripts/demo_ablation.py"
)

if [[ "$ACTION" == "status" ]]; then
  for j in "${JOBS[@]}"; do runai describe job "${j%%:*}" -p "$PROJECT" 2>/dev/null \
      | grep -E "^(Name|Status|Node|Started)" || echo "${j%%:*}: not found"; done
  exit 0
fi

if [[ "$ACTION" == "kill" ]]; then
  for j in "${JOBS[@]}"; do
    n="${j%%:*}"
    runai delete job "$n" -p "$PROJECT" 2>/dev/null && echo "deleted $n" || echo "no job $n"
  done
  exit 0
fi

# Refuse to submit stale code. The runners import from NFS, not from this checkout, so an edit
# that never reached NFS would silently run the OLD simulator -- and the entire experiment is one
# changed line in gossip_sim.py. This is the single most likely way to waste 31 GPU-h.
echo "=== verifying the NFS copy carries attacker_merges ==="
LOCAL_SIM="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src/sim/gossip_sim.py"
if [[ -d /mnt/nfs ]]; then
  if ! grep -q "attacker_merges" "${NFS_REPO}/src/sim/gossip_sim.py" 2>/dev/null; then
    echo "FATAL: ${NFS_REPO}/src/sim/gossip_sim.py has no attacker_merges parameter." >&2
    echo "       Sync the repo to NFS before submitting, or every job runs the OLD behaviour" >&2
    echo "       and produces a duplicate of the existing baseline at full price." >&2
    exit 1
  fi
  for f in scripts/run_merge_sweep.py scripts/demo_ablation.py; do
    [[ -f "${NFS_REPO}/${f}" ]] || { echo "FATAL: ${NFS_REPO}/${f} missing -- sync first" >&2; exit 1; }
  done
  echo "NFS copy OK"
else
  echo "WARNING: /mnt/nfs not mounted here, cannot verify the NFS copy is current."
  echo "         Local reference: ${LOCAL_SIM}"
  echo "         Verify in-cluster before trusting any result:"
  echo "           grep -c attacker_merges ${NFS_REPO}/src/sim/gossip_sim.py   # must be >= 2"
  [[ $DRY -eq 0 ]] && { echo "Refusing to submit unverified. Re-run with --dry-run to see the commands." >&2; exit 1; }
fi

for j in "${JOBS[@]}"; do
  name="${j%%:*}"; cmd="${j#*:}"
  echo
  echo "--- ${name}"
  SUBMIT=(runai submit --name "$name" -i "$IMAGE" -p "$PROJECT"
          --gpu 1 --cpu 32 --memory 128G
          --node-pools "$POOL"
          --pvc sacs-scratch:/mnt/nfs
          -- bash -lc "$cmd")
  if [[ $DRY -eq 1 ]]; then printf '%q ' "${SUBMIT[@]}"; echo; continue; fi
  if "${SUBMIT[@]}"; then echo "submitted ${name}"; else echo "SUBMIT FAILED: ${name}" >&2; fi
done

[[ $DRY -eq 1 ]] && exit 0
cat <<EOF

Submitted. 3 merge runs hold the 3-GPU quota; demo-ablation queues behind them.

Watch:      runai list jobs -p ${PROJECT}
Logs:       runai logs merge-inf -p ${PROJECT} -f
Partials:   ls -la ${NFS_REPO}/results/noniid/merge_attacker/
Kill all:   $0 --kill

Preemption is expected, not exceptional. If a job vanishes, read its .partial.json before
concluding it produced nothing.
EOF
