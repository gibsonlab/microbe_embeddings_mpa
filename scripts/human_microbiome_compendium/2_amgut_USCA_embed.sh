#!/bin/bash
set -e

if [ $# -eq 0 ]; then
  echo "Error: model_name is required"
  echo "Usage: $0 <model_name>"
  exit 1
fi
model_name="$1"

# generate this file/folder by running the step 1 notebook.
NOTEBOOK_CACHE="__tmp/american_gut_USCA"
ASV_SEQ_PROCESSING_DIR="${NOTEBOOK_CACHE}/asv_16s_processing"
EMBEDDING_DIR="${NOTEBOOK_CACHE}/embeddings"

python embed.py \
  --asv_fasta_file "${ASV_SEQ_PROCESSING_DIR}/asv_sequences.post_filter.fasta" \
  --hdf5_output_path "${EMBEDDING_DIR}/${model_name}.h5" \
  --model_name "${model_name}" \
  --embed_batch_size 20 \
  --cuda_devices "0,1,2,3,4,5,6,7"
echo "Done."
