#!/bin/bash
set -e

if [ $# -eq 0 ]; then
  echo "Error: model_name is required"
  echo "Usage: $0 <model_name>"
  exit 1
fi
model_name="$1"



TSV_FILE=/data/cctm/youn/metaphlan_dset/model_training/test.tsv
python dataset_memmap_large_allocate.py \
  --dataset-tsv "$TSV_FILE" \
  --embedding-dir "/data/cctm/youn/metaphlan_dset/embeddings/phylophlan_markers/${model_name}" \
  --memmap-dir "/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/memmap_samples_complete_large/${model_name}/test" \
  --dimension-reduce 768


TSV_FILE=/data/cctm/youn/metaphlan_dset/model_training/train.tsv
python dataset_memmap_large_allocate.py \
  --dataset-tsv "$TSV_FILE" \
  --embedding-dir "/data/cctm/youn/metaphlan_dset/embeddings/phylophlan_markers/${model_name}" \
  --memmap-dir "/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/memmap_samples_complete_large/${model_name}/train" \
  --dimension-reduce 768

echo "Done!"
