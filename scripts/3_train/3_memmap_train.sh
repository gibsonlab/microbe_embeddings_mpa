#!/bin/bash
#SBATCH --partition=bwh_comppath_long
#SBATCH --array=1-15
#SBATCH --ntasks=1
#SBATCH --mem=320G
#SBATCH --cpus-per-task=80
#SBATCH --time=5-00:00:00
#SBATCH --job-name=memmap_train
#SBATCH --output=logs/memmap_train_%A_%a.out
#SBATCH --error=logs/memmap_train_%A_%a.err
set -e


TSV_FILE=/data/cctm/youn/metaphlan_dset/model_training/train.tsv
N_LINES_TSV=$(wc -l < $TSV_FILE)   # Total items (replace with your value)
N_SAMPLES=$((N_LINES_TSV - 1))

# Total number of items
M=$N_SAMPLES
# Total number of jobs
N=${SLURM_ARRAY_TASK_COUNT}
# Current job index (1 to N)
k=${SLURM_ARRAY_TASK_ID}
# Calculate items per job (ceiling division)
items_per_job=$(( (M + N - 1) / N ))
# Calculate start and end indices for this job
start_row=$(( (k - 1) * items_per_job + 1 ))
end_row=$(( k * items_per_job ))
# Ensure end_idx doesn't exceed M
if [ $end_row -gt $M ]; then
    end_row=$M
fi


echo "Memory-mapping: Train set [Rows $start_row ~ $end_row] (inclusive)"
python dataset_memmap.py \
  --dataset-tsv "$TSV_FILE" \
  --embedding-dir "/data/cctm/youn/metaphlan_dset/embeddings/phylophlan_markers/evo" \
  --memmap-dir "/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/memmap_samples_padded" \
  --threads 6 \
  --start $start_row \
  --end $end_row \
  --dimension-reduce 768

echo "Done!"
