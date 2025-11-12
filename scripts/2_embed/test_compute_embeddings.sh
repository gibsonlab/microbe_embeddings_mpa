#!/bin/bash
set -e

HF_TOKEN_FILE=/data/cctm/youn/metaphlan_dset/hf_token.txt
SGB_SUBSET_FILE=/data/cctm/youn/metaphlan_dset/dataset/MetaPhlAn4_paper_profile_SGBs.txt
SGB_INDEX_FILE=/data/cctm/youn/metaphlan_dset/phylophlan_data/processed/sgb_marker_index.json.zst
FASTA_FILE=/data/cctm/youn/metaphlan_dset/phylophlan_data/processed/all_markers.fna.bgz

HF_TOKEN=$(cat $HF_TOKEN_FILE)
HF_HOME="/data/cctm/youn/huggingface_cache"

TOTAL_SGBS=$(wc -l < $SGB_FILE)   # Total items (replace with your value)

# Total number of items
M=$TOTAL_SGBS

# Total number of jobs
N=8

# Current job index (1 to N)
k=1

# Calculate items per job (ceiling division)
items_per_job=$(( (M + N - 1) / N ))

# Calculate start and end indices for this job
start_idx=$(( (k - 1) * items_per_job + 1 ))
end_idx=$(( k * items_per_job ))

# Ensure end_idx doesn't exceed M
if [ $end_idx -gt $M ]; then
    end_idx=$M
fi

# Only run if start_idx is valid
if [ $start_idx -le $M ]; then
    echo "Job $k processing items $start_idx to $end_idx"

    outdir=/data/cctm/youn/metaphlan_dset/embeddings/evo/part${k}
    breadcrumb=$outdir/embed.DONE
    if [ -f "$breadcrumb" ]; then
        echo "Task array index ${k} was already finished previously."
    else
      mkdir -p "$outdir"
      echo "Destination output: $outdir"

      HF_HOME=$HF_HOME \
      HF_TOKEN=$HF_TOKEN \
      python compute_embeddings.py \
        --model "evo" \
        --fasta "$FASTA_FILE" \
        --sgb-list "$SGB_SUBSET_FILE" \
        --sgb-index-file "$SGB_INDEX_FILE" \
        --start "$start_idx" \
        --end "$end_idx" \
        --batch-size 2 \
        --out-dir "$outdir" \
        --shard-size 50000
      echo "Done."
      touch $breadcrumb
    fi
else
    echo "Job $k has no items to process"
fi
