#!/bin/bash

for seed in 1001 1002 1003 1004 1005; do
  python train_test_split.py \
    --full-table "/data/bwh-comppath-seq/youn/metaphlan_dset/dataset/BlancoMiguezA_2023_profiles.tsv" \
    --metadata "/data/bwh-comppath-seq/youn/metaphlan_dset/dataset/BlancoMiguezA_2023_metadata.tsv" \
    --out-dir "/data/bwh-comppath-seq/youn/metaphlan_dset/analyses/random_split_${seed}" \
    --method "random" \
    --rng-seed ${seed}
done
