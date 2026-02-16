#!/bin/bash
#SBATCH --partition=bwh_comppath_all
#SBATCH --gpus=8
#SBATCH --mem=12G
#SBATCH --cpus-per-task=40
#SBATCH --time=2-00:00:00
#SBATCH --job-name=mpa_embed_dnabert
#SBATCH --output=logs/embed_dnabert_%A_%a.out
#SBATCH --error=logs/embed_dnabert_%A_%a.err


# Note: this is a Slurm script, meant to be run on ErisXDL compute nodes with 8 A100s.
# This script runs compute_stacked_gene_embeddings.py, which evaluates a numpy.memmap representation of SGB marker embeddings.
if [ $# -lt 1 ]; then
  echo "Error: embed_model_name is required. (Suggestions: dnabert-s, evo-1-8k-base_hyena5, evo2_7b_hyena10)"
  echo "Usage: $0 <embed_model_name>"
  exit 1
fi
embed_model_name="$1"

set -e
HF_TOKEN_FILE="/data/bwh-comppath-seq/youn/metaphlan_dset/hf_token.txt"
SGB_SUBSET_FILE="/data/bwh-comppath-seq/youn/metaphlan_dset/dataset/BlancoMiguezA_2023.SGB_subset.txt"
SGB_INDEX_DIR="/data/bwh-comppath-seq/youn/metaphlan_dset/phylophlan_database/processed/dna_only"

HF_TOKEN=$(cat $HF_TOKEN_FILE)
HF_HOME="/data/cctm/youn/huggingface_cache"
SCRIPT_DIR="$(pwd)"

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
PROJECT_ROOT_DIR=$(find_gem_project_dir)

out_dir="/data/bwh-comppath-seq/youn/metaphlan_dset/embeddings/phylophlan"
mkdir -p "${out_dir}"
out_file="${embed_model_name}.npy"


if [ "${embed_model_name}" == "pcoa" ] | [ "${embed_model_name}" == "umap" ]; then
    echo "Will create symlinks to pre-computed embeddings."
    echo "Please run '1_embed_offline.sh ${embed_model_name}'"
    ln -s /data/bwh-comppath-seq/youn/metaphlan_dset/embeddings/offline/${embed_model_name}_d100_s1000.* ${out_dir}/
elif [[ $embed_model_name == evo2* ]]; then
    APPTAINER_IMAGE=/data/cctm/youn/docker_images/evo2_gem.sif
    echo "Evo2 model detected. Running using Apptainer (${APPTAINER_IMAGE})."

    # this path is built-into the apptainer image.
    container_local_evo2_config=/usr/local/lib/python3.12/dist-packages/evo2/configs/evo2-7b-1m.yml

    singularity exec --nv \
        --bind "./evo2-7b-1m.NO_FP8.yml:${container_local_evo2_config}" \
        --bind "${HF_HOME}:/hf_home" \
        --bind "${PROJECT_ROOT_DIR}:/project_base" \
        --bind "${SCRIPT_DIR}:/script_home" \
        --bind "${out_dir}:/out_dir" \
        --bind "${SGB_SUBSET_FILE}:/sgb_subset_ids.txt" \
        --bind "${SGB_INDEX_DIR}:/sgb_index" \
        --env "HF_HOME=/hf_home" \
        --env "PYTHONPATH=/project_base" \
        --pwd "/script_home" \
        --env "HF_TOKEN=${HF_TOKEN}" \
        "${APPTAINER_IMAGE}" \
        python compute_stacked_gene_embeddings.py \
          --model-name "${embed_model_name}" \
          --sgb-subset-file "/sgb_subset_ids.txt" \
          --sgb-marker-index "/sgb_index" \
          --cuda-device-ids "0,1,2,3,4,5,6,7" \
          --output-path "/out_dir/${out_file}"
else
  HF_HOME=$HF_HOME \
  HF_TOKEN=$HF_TOKEN \
  python compute_stacked_gene_embeddings.py \
    --model-name "${embed_model_name}" \
    --sgb-subset-file "${SGB_SUBSET_FILE}" \
    --sgb-marker-index "${SGB_INDEX_DIR}" \
    --cuda-device-ids "0,1,2,3,4,5,6,7" \
    --output-path "${out_dir}/${out_file}"
fi
