#!/bin/bash
#SBATCH --partition=bwh_comppath_long
#SBATCH --ntasks=1
#SBATCH --mem=320G
#SBATCH --cpus-per-task=80
#SBATCH --time=5-00:00:00
#SBATCH --job-name=hdf5_conversion
#SBATCH --output=logs/hdf5_conversion_%A_%a.out
#SBATCH --error=logs/hdf5_conversion_%A_%a.err
set -e


TSV_FILE=/data/cctm/youn/metaphlan_dset/model_training/test.tsv
OUT_FILE=/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/hdf5_samples/test.hdf5
MEMMAP_DIR=/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/memmmap_samples

echo "HDF5 conversion of ${TSV_FILE} started."
python dataset_hdf5_convert.py \
  --dataset-tsv "$TSV_FILE" \
  --memmap-dir "$MEMMAP_DIR" \
  --out-path "$OUT_FILE" \
  --dimension-reduce 768

TSV_FILE=/data/cctm/youn/metaphlan_dset/model_training/train.tsv
OUT_FILE=/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/hdf5_samples/train.hdf5

echo "HDF5 conversion of ${TSV_FILE} started."
python dataset_hdf5_convert.py \
  --dataset-tsv "$TSV_FILE" \
  --memmap-dir "$MEMMAP_DIR" \
  --out-path "$OUT_FILE" \
  --dimension-reduce 768

echo "Done!"
