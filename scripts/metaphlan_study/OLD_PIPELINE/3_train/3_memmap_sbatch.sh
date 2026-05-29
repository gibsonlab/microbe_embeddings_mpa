#!/bin/bash


for embed_model in "dnabert-s" "evo-1-8k-base_hyena5" "evo2_7b_hyena10" ; do
  logdir="./logs/memmap/${embed_model}"
  mkdir -p "${logdir}"
  sbatch --job-name="memmap:${embed_model}" --output="${logdir}/memmap_%A_%a.out" --error="${logdir}/memmap_%A_%a.err" 3_memmap.sh "${embed_model}"
done
