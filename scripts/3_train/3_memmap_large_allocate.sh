#!/bin/bash
set -e

python dataset_memmap_large_allocate.py \
  --dataset-tsv "$TSV_FILE" \
  --embedding-dir "/data/cctm/youn/metaphlan_dset/embeddings/phylophlan_markers/evo" \
  --memmap-dir "/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/memmap_samples_complete_large" \
  --dimension-reduce 768

echo "Done!"
