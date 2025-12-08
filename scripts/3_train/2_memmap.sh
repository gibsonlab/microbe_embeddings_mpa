#!/bin/bash
#SBATCH --partition=bwh_comppath_long
#SBATCH --ntasks=1
#SBATCH --mem=280G
#SBATCH --cpus-per-task=40
#SBATCH --time=5-00:00:00
#SBATCH --job-name=memmap_dataset
#SBATCH --output=memmap_%A_%a.out
#SBATCH --error=memmap_%A_%a.err
set -e


echo "Memory-mapping: Train set"
python dataset_memmap.py \
  --dataset-tsv "/data/cctm/youn/metaphlan_dset/model_training/train.tsv" \
  --embedding-dir "/data/cctm/youn/metaphlan_dset/embeddings/phylophlan_markers/evo" \
  --memmap-dir "/data/cctm/youn/metaphlan_dset/model_training/memmmap_samples" \
  --threads 5


echo "Memory-mapping: Test set"
python dataset_memmap.py \
  --dataset-tsv "/data/cctm/youn/metaphlan_dset/model_training/test.tsv" \
  --embedding-dir "/data/cctm/youn/metaphlan_dset/embeddings/phylophlan_markers/evo" \
  --memmap-dir "/data/cctm/youn/metaphlan_dset/model_training/memmmap_samples" \
  --threads 5

echo "Done!"
