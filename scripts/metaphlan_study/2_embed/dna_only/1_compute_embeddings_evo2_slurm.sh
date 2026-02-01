#!/bin/bash
#SBATCH --partition=bwh_comppath_all
#SBATCH --array=1-8
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --job-name=mpa_embed_evo2
#SBATCH --output=logs/embed_evo2_%A_%a.out
#SBATCH --error=logs/embed_evo2_%A_%a.err

# Note: this is a Slurm script, meant to be run on ErisXDL compute nodes with 8 A100s.

set -e

HF_TOKEN_FILE=/data/cctm/youn/metaphlan_dset/hf_token.txt
SGB_SUBSET_FILE=/data/cctm/youn/metaphlan_dset/dataset/MetaPhlAn4_paper_profile_SGBs.txt
SGB_INDEX_FILE=/data/cctm/youn/metaphlan_dset/phylophlan_data/processed/dna_only/sgb_marker_index.json.zst
FASTA_FILE=/data/cctm/youn/metaphlan_dset/phylophlan_data/processed/dna_only/markers.fna
APPTAINER_IMAGE=/data/cctm/youn/docker_images/evo2_gem.sif

HF_TOKEN=$(cat $HF_TOKEN_FILE)
HF_HOME="/data/cctm/youn/huggingface_cache"

EVO2_CHECKPOINT="evo2_7b"
NUM_HYENA_LAYERS=26


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
echo "Apptainer will bind the project root directory: ${PROJECT_ROOT_DIR}"


# Only run if start_idx is valid
if [ $start_idx -le $M ]; then
    echo "Job $k processing items $start_idx to $end_idx (inclusive)"

    outdir="/data/bwh-comppath-seq/youn/metaphlan_dset/embeddings/phylophlan_markers/dna/${EVO2_CHECKPOINT}_hyena${NUM_HYENA_LAYERS}/part${k}"
    breadcrumb=$outdir/.embed.DONE
    if [ -f "$breadcrumb" ]; then
        echo "Task array index ${k} was already finished previously."
    else
      mkdir -p "$outdir"
      echo "Destination output: $outdir"

      PARENT_DIR = "$(dirname "$PWD")"
      singularity exec --nv \
        --bind "${HF_HOME}:/hf_home" \
        --bind "${FASTA_FILE}:/markers/sequences.fasta" \
        --bind "${SGB_SUBSET_FILE}:/markers/sgb_subset.txt" \
        --bind "${SGB_INDEX_FILE}:/markers/sgb_marker_index.json.zst" \
        --bind "${outdir}:/out_dir" \
        --bind "${PROJECT_ROOT_DIR}:/lib" \
        --bind "${PARENT_DIR}:/script_home" \
        --env "PYTHONPATH=/lib" \
        --env "HF_HOME=/hf_home" \
        --env "HF_TOKEN=${HF_TOKEN}" \
        --pwd /script_home \
        "${APPTAINER_IMAGE}" \
        python compute_embeddings.py \
          --model "${EVO2_CHECKPOINT}:${NUM_HYENA_LAYERS}" \
          --fasta "/markers/sequences.fasta" \
          --sgb-list "/markers/sgb_subset.txt" \
          --sgb-index-file "/markers/sgb_marker_index.json.zst" \
          --start "$start_idx" \
          --end "$end_idx" \
          --batch-size 20 \
          --out-dir "/out_dir" \
          --shard-size 50000
      echo "Done."
      touch $breadcrumb
    fi
else
    echo "Job $k has no items to process (this shouldn't happen!)"
fi
