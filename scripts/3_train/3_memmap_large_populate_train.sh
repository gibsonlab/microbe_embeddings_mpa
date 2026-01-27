#!/bin/bash
#SBATCH --partition=bwh_comppath_long
#SBATCH --array=1-80
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --time=5-00:00:00
#SBATCH --job-name=memmap_large
set -e

# This is a SLURM task meant to be run on ErisTwo CPU clusters.

if [ $# -eq 0 ]; then
  echo "Error: model_name is required"
  echo "Usage: $0 <model_name>"
  exit 1
fi
model_name="$1"
dset_name="train"


TSV_FILE="/data/cctm/youn/metaphlan_dset/model_training/${dset_name}.tsv"
if ! [ -f "${TSV_FILE}" ]; then
  echo "Dataset input file ${TSV_FILE} does not exist!"
  exit 1
fi
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


echo "Memory-mapping: Test set [Rows $start_row ~ $end_row] (inclusive)"

logdir="logs/mmap_pop_${dset_name}_${model_name}_${SLURM_JOB_ID}"
mkdir -p "${logdir}"
exec > "${logdir}/task_${SLURM_ARRAY_TASK_ID}.out" 2> "${logdir}/task_${SLURM_ARRAY_TASK_ID}.err"

python dataset_memmap_large_populate.py \
  --dataset-tsv "$TSV_FILE" \
  --embedding-dir "/data/cctm/youn/metaphlan_dset/embeddings/phylophlan_markers/${model_name}" \
  --memmap-dir "/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/memmap_samples_complete_large/${dset_name}" \
  --start $start_row \
  --end $end_row \
  --dimension-reduce 768

echo "Done!"
