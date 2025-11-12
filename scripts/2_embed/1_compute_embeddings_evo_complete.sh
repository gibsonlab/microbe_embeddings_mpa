#!/bin/bash
set -e

HF_TOKEN_FILE=/data/cctm/youn/metaphlan_dset/hf_token.txt
SGB_SUBSET_FILE=/data/cctm/youn/metaphlan_dset/dataset/MetaPhlAn4_paper_profile_SGBs.txt
SGB_INDEX_FILE=/data/cctm/youn/metaphlan_dset/phylophlan_data/processed/sgb_marker_index.json.zst
FASTA_FILE=/data/cctm/youn/metaphlan_dset/phylophlan_data/processed/all_markers.fna.bgz

HF_TOKEN=$(cat $HF_TOKEN_FILE)
HF_HOME="/data/cctm/youn/huggingface_cache"
TOTAL_SGBS=$(wc -l < $SGB_SUBSET_FILE)   # Total items (replace with your value)


outdir=/data/cctm/youn/metaphlan_dset/embeddings/phylophlan_markers/evo/complete
breadcrumb=$outdir/.embed.DONE
if [ -f "$breadcrumb" ]; then
    echo "Task already finished previously."
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
    --start "1" \
    --end "$TOTAL_SGBS" \
    --batch-size 10 \
    --out-dir "$outdir" \
    --shard-size 50000
  echo "Done."
  touch $breadcrumb
fi
