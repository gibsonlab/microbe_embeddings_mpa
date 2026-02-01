#!/bin/bash
#SBATCH --partition=bwh_comppath_long
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=40
#SBATCH --time=2-00:00:00
#SBATCH --job-name=pca_dim_reduce
set -e

if [ $# -eq 0 ]; then
  echo "Error: model_name is required"
  echo "Usage: $0 <model_name>"
  exit 1
fi
model_name="$1"

pca_dim=200
python precompute_embedding_pca.py \
  -i "/data/bwh-comppath-seq/youn/metaphlan_dset/embeddings/phylophlan_markers/dna/${model_name}" \
  --dimension-reduce ${pca_dim} \
  --pca-batch-size 10000
