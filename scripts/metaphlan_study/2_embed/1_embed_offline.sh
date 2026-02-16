#!/bin/bash
#SBATCH --partition=bwh_comppath_all
#SBATCH --gpus=8
#SBATCH --mem=12G
#SBATCH --cpus-per-task=40
#SBATCH --time=2-00:00:00
#SBATCH --job-name=mpa_embed_dnabert
#SBATCH --output=logs/embed_dnabert_%A_%a.out
#SBATCH --error=logs/embed_dnabert_%A_%a.err


# Note: this is a Slurm script, meant to be run on ErisXDL compute nodes with 8 A100s.
# This script runs compute_stacked_gene_embeddings.py, which evaluates a numpy.memmap representation of SGB marker embeddings.
if [ $# -lt 1 ]; then
  echo "Error: embed_method is required. (Suggestions: umap, pcoa)"
  echo "Usage: $0 <embed_method>"
  exit 1
fi
embed_method="$1"
embed_dim=100
embed_seed=1000

set -e
DISTMAT_FILE="/data/bwh-comppath-seq/youn/metaphlan_dset/dataset/BlancoMiguezA_2023.DIST_MATRIX.txt"


out_dir="/data/bwh-comppath-seq/youn/metaphlan_dset/embeddings/offline"
mkdir -p "${out_dir}"
out_file="${embed_method}_d${embed_dim}_s${embed_seed}.h5"


echo "Embedding method: ${embed_method}"
echo "Target file: ${out_dir}/${out_file}"
python compute_offline_embeddings.py \
  --method "${embed_method}" \
  --embed-dim "${embed_dim}" \
  --seed "${embed_seed}" \
  --distance-matrix "${DISTMAT_FILE}" \
  --out "${out_dir}/${out_file}"
echo "Done."
