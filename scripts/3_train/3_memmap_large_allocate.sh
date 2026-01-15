#!/bin/bash
set -e

TSV_FILE=/data/cctm/youn/metaphlan_dset/model_training/test.tsv
python dataset_memmap_large_allocate.py \
  --dataset-tsv "$TSV_FILE" \
  --embedding-dir "/data/cctm/youn/metaphlan_dset/embeddings/phylophlan_markers/evo" \
  --memmap-dir "/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/memmap_samples_complete_large/test" \
  --dimension-reduce 768


TSV_FILE=/data/cctm/youn/metaphlan_dset/model_training/train.tsv
python dataset_memmap_large_allocate.py \
  --dataset-tsv "$TSV_FILE" \
  --embedding-dir "/data/cctm/youn/metaphlan_dset/embeddings/phylophlan_markers/evo" \
  --memmap-dir "/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/memmap_samples_complete_large/train" \
  --dimension-reduce 768

echo "Done!"
