#!/bin/bash
# Submit an INTERACTIVE job (for development): 1 GPU, my image, my NFS mounted.
# Interactive = up to 12h, easy to shell into. Good for exploration.
runai submit \
    --name iclscan-dev \
    -i registry.rcp.epfl.ch/sacs-peechara/iclscan:latest \
    -p sacs-peechara \
    --gpu 1 \
    --interactive \
    --pvc sacs-scratch:/mnt/nfs \
    -- sleep infinity
