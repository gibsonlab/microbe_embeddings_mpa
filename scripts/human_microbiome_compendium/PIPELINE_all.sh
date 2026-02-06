#!/bin/bash
set -e

BASEDIR="/data/bwh-comppath-seq/youn/human_microbiome_compendium"
if ! [ $# -eq 1 ]; then
  echo "Error: dataset is required"
  echo "Usage: $0 <dataset>"
  echo "Available dataset names:"
  for dir in "${BASEDIR}"/*; do
    echo "-> $(basename "${dir}")"
  done
  exit 1
fi
dset_name="$1"

if ! [ -d "${BASEDIR}/${dset_name}" ]; then
  echo "Dataset files for '${dset_name}' do not exist!"
  exit 1
fi

for embed_name in "dnabert-s" "evo2_7b_hyena10" "evo-1-8k-base_hyena5"; do
  bash 2_embed.sh "$embed_name" "$dset_name"
  bash 3_train_model_epc.sh "$embed_name" "$dset_name"
  bash 3_train_model_epc_nopool.sh "$embed_name" "$dset_name"
done

