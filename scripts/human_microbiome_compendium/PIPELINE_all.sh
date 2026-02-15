#!/bin/bash
set -e

BASEDIR="/data/bwh-comppath-seq/youn/human_microbiome_compendium"
# Suggest dataset subdir names.
if [ $# -eq 0 ]; then
  echo "Error: dataset is required"
  echo "Usage: $0 <dataset> <analysis>"
  echo "Available dataset names:"
  for dir in "${BASEDIR}"/*; do
    echo "-> $(basename "${dir}")"
  done
  echo "Run this command with 'bash PIPELINE_all.sh <dataset>' to print the available analysis subdirectories."
  exit 1
fi
dset_name="$1"

if ! [ -d "${BASEDIR}/${dset_name}" ]; then
  echo "Dataset files for '${dset_name}' do not exist!"
  exit 1
fi

# Suggest analysis subdir names.
print_available_analyses() {
  _dset_name=$1
  subdir="${BASEDIR}/${dset_name}/analyses"

  echo "Available analyses for dataset '${dset_name}':"
  analyses_base="${BASEDIR}/${dset_name}/analyses"
  if [ -d "${analyses_base}" ]; then
    for dir in "${analyses_base}"/*; do
      if [ -d "${dir}" ]; then
        echo "-> $(basename "${dir}")"
      fi
    done
  else
    echo "No analyses/ directory found for this dataset"
  fi
}

if [ $# -eq 1 ]; then
  print_available_analyses "$dset_name"
  exit 1
fi

analysis_name="$2"
ANALYSIS_DIR="${BASEDIR}/${dset_name}/analyses/${analysis_name}"
if ! [ -d "${ANALYSIS_DIR}" ]; then
  echo "Analysis subdir does not exist: ${ANALYSIS_DIR}"
  print_available_analyses "$dset_name"
  exit 1
fi


# Initialize empty array
embed_names_all=()

# Append entries one by one
embed_names_all+=("dnabert-s")
embed_names_all+=("evo-1-8k-base_hyena5")
embed_names_all+=("evo2_7b_hyena10")
embed_names_all+=("umap_d100_s1000")

for embed_name in "${embed_names_all[@]}"; do
  bash 2_embed.sh "$embed_name" "$dset_name"
  bash 3_train_model.sh "$dset_name" "$analysis_name" "$embed_name" "epc_pool"
  bash 3_train_model.sh "$dset_name" "$analysis_name" "$embed_name" "epc_pool_1"
  bash 3_train_model.sh "$dset_name" "$analysis_name" "$embed_name" "epc_pool_2"
  bash 3_train_model.sh "$dset_name" "$analysis_name" "$embed_name" "epc_pool_3"
  bash 3_train_model.sh "$dset_name" "$analysis_name" "$embed_name" "epc_nopool"
done

