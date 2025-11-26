#!/bin/bash
set -e


embeddings=/data/cctm/youn/metaphlan_dset/embeddings/phylophlan_markers/evo
python 2_memmap.py \
  --input-embed-dir "${embeddings}" \
  --out-memmap-dir "${embeddings}/memmap" \
  --threads 12
echo "Done."
