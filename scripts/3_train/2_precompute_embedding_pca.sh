#!/bin/bash
set -e

if [ $# -eq 0 ]; then
  echo "Error: model_name is required"
  echo "Usage: $0 <model_name>"
  exit 1
fi
model_name="$1"

pca_dim=300
python precompute_embedding_pca.py \
  -i "/data/local/youn/metaphlan_abundance_prediction/embedding/${model_name}" \
  --dimension-reduce ${pca_dim} \
  --pca-batch-size 10000
