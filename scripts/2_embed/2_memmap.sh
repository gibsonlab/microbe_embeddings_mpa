#!/bin/bash
#SBATCH --partition=bwh_comppath_long
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=40
#SBATCH --time=5-00:00:00
#SBATCH --job-name=memmap_tensors
#SBATCH --output=memmap_%A_%a.out
#SBATCH --error=memmap_%A_%a.err

# Note: this is a Slurm script, meant to be run on ErisTwo/ErisXDL (CPU only) compute nodes, but it can also be run as a bash script.
set -e


embeddings=/data/cctm/youn/metaphlan_dset/embeddings/phylophlan_markers/evo
python 2_memmap.py \
  --input-embed-dir "${embeddings}" \
  --out-memmap-dir "${embeddings}/memmap" \
  --threads 30
echo "Done."
