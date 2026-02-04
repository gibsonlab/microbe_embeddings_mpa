#!/bin/bash
set -e

if ! [ $# -eq 2 ]; then
  echo "Error: embed_model_name and dataset are required"
  echo "Usage: $0 <embed_model_name> <dataset>"
  exit 1
fi
embed_model_name="$1"
dataset_name="$2"

# generate this file/folder by running the step 1 notebook.
NOTEBOOK_CACHE="__tmp/${dataset_name}"
ASV_SEQ_PROCESSING_DIR="${NOTEBOOK_CACHE}/asv_16s_processing"
EMBEDDING_DIR="${NOTEBOOK_CACHE}/embeddings"
mkdir -p "${EMBEDDING_DIR}"

# Bash function, which finds the project root directory.
# This function repeatedly traverses upwards, until it finds an ancestor with the subdirectory "gem".
find_gem_project_dir() {
    local dir="$PWD"

    while [ "$dir" != "/" ]; do
        if [ -d "$dir/gem" ]; then
            echo "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done

    echo "No ancestor directory containing 'gem' folder found" >&2
    return 1
}
SCRIPT_DIR="$(pwd)"
PROJECT_ROOT_DIR=$(find_gem_project_dir)
HF_TOKEN_FILE=/data/cctm/youn/metaphlan_dset/hf_token.txt
HF_TOKEN=$(cat $HF_TOKEN_FILE)
HF_HOME="/data/cctm/youn/huggingface_cache"

if [[ $embed_model_name == evo2* ]]; then
    APPTAINER_IMAGE=/data/cctm/youn/docker_images/evo2_gem.sif
    echo "Evo2 model detected. Running using Apptainer (${APPTAINER_IMAGE})."

    # this path is built-into the apptainer image.
    container_local_evo2_config=/usr/local/lib/python3.12/dist-packages/evo2/configs/evo2-7b-1m.yml

    singularity exec --nv \
        --bind "./evo2-7b-1m.NO_FP8.yml:${container_local_evo2_config}" \
        --bind "${HF_HOME}:/hf_home" \
        --bind "${PROJECT_ROOT_DIR}:/project_base" \
        --bind "${SCRIPT_DIR}:/script_home" \
        --bind "${ASV_SEQ_PROCESSING_DIR}/asv_sequences.post_filter.fasta:/seqs.fasta" \
        --bind "${EMBEDDING_DIR}:/out_dir" \
        --env "HF_HOME=/hf_home" \
        --env "PYTHONPATH=/project_base" \
        --pwd "/script_home" \
        --env "HF_TOKEN=${HF_TOKEN}" \
        "${APPTAINER_IMAGE}" \
        python embed.py \
          --asv_fasta_file "/seqs.fasta" \
          --hdf5_output_path "${EMBEDDING_DIR}/${embed_model_name}.h5" \
          --model_name "${embed_model_name}" \
          --embed_batch_size 20 \
          --cuda_device_ids "0"
else
    python embed.py \
      --asv_fasta_file "seqs.fasta" \
      --hdf5_output_path "/out_dir/${embed_model_name}.h5" \
      --model_name "${embed_model_name}" \
      --embed_batch_size 20 \
      --cuda_device_ids "0"
fi
echo "Done."
