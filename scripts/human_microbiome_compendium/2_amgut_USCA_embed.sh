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
mkdir -p "${EMBEDDING_DIR}"

if [[ $model_name == evo2* ]]; then
    APPTAINER_IMAGE=/data/cctm/youn/docker_images/evo2_gem.sif
    echo "Evo2 model detected. Running using Apptainer (${APPTAINER_IMAGE})."
    singularity exec --nv \
        --bind "${HF_HOME}:/hf_home" \
        --bind "${PROJECT_ROOT_DIR}:/project_base" \
        --bind "${PARENT_DIR}:/script_home" \
        --bind "${ASV_SEQ_PROCESSING_DIR}/asv_sequences.post_filter.fasta:/seqs.fasta" \
        --bind "${EMBEDDING_DIR}:/out_dir" \
        --env "HF_HOME=/hf_home" \
        --env "PYTHONPATH=/project_base" \
        --pwd "/script_home" \
        --env "HF_TOKEN=${HF_TOKEN}" \
        "${APPTAINER_IMAGE}" \
        python embed.py \
          --asv_fasta_file "" \
          --hdf5_output_path "${EMBEDDING_DIR}/${model_name}.h5" \
          --model_name "${model_name}" \
          --embed_batch_size 20 \
          --cuda_device_ids "0"
else
    python embed.py \
      --asv_fasta_file "seqs.fasta" \
      --hdf5_output_path "/out_dir/${model_name}.h5" \
      --model_name "${model_name}" \
      --embed_batch_size 20 \
      --cuda_device_ids "0"
fi
echo "Done."
