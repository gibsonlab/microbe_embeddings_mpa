#!/bin/bash
#SBATCH --partition=bwh_comppath_all
#SBATCH --array=1-24
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --job-name=mpa_embed_evo
#SBATCH --output=logs/embed_evo_%A_%a.out
#SBATCH --error=logs/embed_evo_%A_%a.err

# Note: this is a Slurm script, meant to be run on ErisXDL compute nodes with 8 A100s.

set -e

HF_TOKEN_FILE=/data/cctm/youn/metaphlan_dset/hf_token.txt
SGB_SUBSET_FILE=/data/cctm/youn/metaphlan_dset/dataset/MetaPhlAn4_paper_profile_SGBs.txt
SGB_INDEX_FILE=/data/cctm/youn/metaphlan_dset/phylophlan_data/processed/dna_only/sgb_marker_index.json.zst
FASTA_FILE=/data/cctm/youn/metaphlan_dset/phylophlan_data/processed/dna_only/markers.fna

HF_TOKEN=$(cat $HF_TOKEN_FILE)
HF_HOME="/data/cctm/youn/huggingface_cache"

EVO_CHECKPOINT="evo-1-8k-base"
NUM_HYENA_LAYERS=5


TOTAL_SGBS=$(wc -l < $SGB_SUBSET_FILE)   # Total items (replace with your value)
M=$TOTAL_SGBS
N=${SLURM_ARRAY_TASK_COUNT}
k=${SLURM_ARRAY_TASK_ID}

# Floor division
items_per_job=$(( M / N ))
remainder=$(( M % N ))

# First 'remainder' jobs get one extra item
if [ $k -le $remainder ]; then
    start_idx=$(( (k - 1) * (items_per_job + 1) + 1 ))
    end_idx=$(( k * (items_per_job + 1) ))
else
    start_idx=$(( remainder * (items_per_job + 1) + (k - remainder - 1) * items_per_job + 1 ))
    end_idx=$(( remainder * (items_per_job + 1) + (k - remainder) * items_per_job ))
fi

# Ensure end_idx doesn't exceed M
if [ $end_idx -gt $M ]; then
    end_idx=$M
fi

# Only run if start_idx is valid
if [ $start_idx -le $M ]; then
    echo "Job $k processing items $start_idx to $end_idx (inclusive)"

    outdir="/data/bwh-comppath-seq/youn/metaphlan_dset/embeddings/phylophlan_markers/dna/${EVO_CHECKPOINT}_hyena${NUM_HYENA_LAYERS}/part${k}"
    breadcrumb=$outdir/.embed.DONE
    if [ -f "$breadcrumb" ]; then
        echo "Task array index ${k} was already finished previously."
    else
      mkdir -p "$outdir"
      echo "Destination output: $outdir"

      HF_HOME=$HF_HOME \
      HF_TOKEN=$HF_TOKEN \
      python ../compute_embeddings.py \
        --model "${EVO_CHECKPOINT}:${NUM_HYENA_LAYERS}" \
        --fasta "$FASTA_FILE" \
        --sgb-list "$SGB_SUBSET_FILE" \
        --sgb-index-file "$SGB_INDEX_FILE" \
        --start "$start_idx" \
        --end "$end_idx" \
        --batch-size 20 \
        --out-dir "$outdir" \
        --shard-size 50000
      echo "Done."
      touch $breadcrumb
    fi
else
    echo "Job $k has no items to process (this shouldn't happen!)"
fi
