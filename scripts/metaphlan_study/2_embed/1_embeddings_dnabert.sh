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


set -e
HF_TOKEN_FILE="/data/bwh-comppath-seq/youn/metaphlan_dset/hf_token.txt"
SGB_SUBSET_FILE="/data/bwh-comppath-seq/youn/metaphlan_dset/dataset/BlancoMiguezA_2023.SGB_subset.txt"
SGB_INDEX_DIR="/data/bwh-comppath-seq/youn/metaphlan_dset/phylophlan_data/processed/dna"

HF_TOKEN=$(cat $HF_TOKEN_FILE)
HF_HOME="/data/cctm/youn/huggingface_cache"


out_dir="/data/bwh-comppath-seq/youn/metaphlan_dset/embeddings/phylophlan"
mkdir -p "${out_dir}"

out_path="${out_dir}/dnabert-s.npy"
python compute_stacked_gene_embeddings.py \
  --model-name "dnabert-s" \
  --sgb-subset-file "${SGB_SUBSET_FILE}" \
  --sgb-marker-index "${SGB_INDEX_DIR}" \
  --cuda-device-ids "0,1,2,3,4,5,6,7" \
  --output-path "${out_path}"
