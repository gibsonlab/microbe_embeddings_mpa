#!/bin/bash
#SBATCH --partition=bwh_comppath
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=20
#SBATCH --time=1-00:00:00
#SBATCH --job-name=mpa_embed_offline
#SBATCH --output=logs/mpa_embed_offline_%A_%a.out
#SBATCH --error=logs/mpa_embed_offline_%A_%a.err

set -e

# Note: this is a Slurm script, meant to be run on CPU nodes (e.g. eristwo).
if ! [ $# -eq 1 ]; then
  echo "Error: embed_method is required."
  echo "Usage: $0 <embed_method>"
  exit 1
fi
embed_method="$1"


DISTMAT_FILE="/data/bwh-comppath-seq/youn/metaphlan_dset/MetaPhlAn4_paper_profile_SGBs.DIST_MATRIX.npz"
embed_dim=100
seed=1000

outdir="/data/bwh-comppath-seq/youn/metaphlan_dset/embeddings/phylogenetic_distance"
mkdir -p "${outdir}"

outfile="${outdir}/${embed_method}.h5"
python ../compute_offline_embeddings.py \
  --method "${embed_method}" \
  --embed-dim "${embed_dim}" \
  --seed "${seed}" \
  --distance-matrix "${DISTMAT_FILE}" \
  --out "${outfile}"
echo "Done."
