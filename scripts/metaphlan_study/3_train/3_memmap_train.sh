#!/bin/bash
#SBATCH --partition=bwh_comppath_long
#SBATCH --array=1-80
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --time=5-00:00:00
#SBATCH --job-name=memmap_train
#SBATCH --output=logs/memmap_train_%A_%a.out
#SBATCH --error=logs/memmap_train_%A_%a.err
set -e


if [ $# -eq 0 ]; then
  echo "Error: model_name is required"
  echo "Usage: $0 <model_name>"
  exit 1
fi
model_name="$1"


TSV_FILE=/data/cctm/youn/metaphlan_dset/model_training/train.tsv
N_LINES_TSV=$(wc -l < $TSV_FILE)   # Total items (replace with your value)
N_SAMPLES=$((N_LINES_TSV - 1))

M=$N_SAMPLES
N=${SLURM_ARRAY_TASK_COUNT}
k=${SLURM_ARRAY_TASK_ID}

# Floor division
items_per_job=$(( M / N ))
remainder=$(( M % N ))

# First 'remainder' jobs get one extra item
if [ $k -le $remainder ]; then
    start_row=$(( (k - 1) * (items_per_job + 1) + 1 ))
    end_row=$(( k * (items_per_job + 1) ))
else
    start_row=$(( remainder * (items_per_job + 1) + (k - remainder - 1) * items_per_job + 1 ))
    end_row=$(( remainder * (items_per_job + 1) + (k - remainder) * items_per_job ))
fi

# Ensure end_idx doesn't exceed M
if [ $end_row -gt $M ]; then
    end_row=$M
fi


echo "Memory-mapping: Train set [Rows $start_row ~ $end_row] (inclusive)"
pca_dim=200
python dataset_memmap.py \
  --dataset-tsv "$TSV_FILE" \
  --embedding-dir "/data/cctm/youn/metaphlan_dset/embeddings/phylophlan_markers/${model_name}" \
  --memmap-dir "/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/memmap_samples/${model_name}_d${pca_dim}" \
  --threads 6 \
  --start $start_row \
  --end $end_row \
  --dimension-reduce ${pca_dim}

echo "Done!"
