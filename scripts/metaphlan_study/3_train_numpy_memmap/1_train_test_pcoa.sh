#!/bin/bash

python train_test_split.py \
  --full-table "/data/bwh-comppath-seq/youn/metaphlan_dset/dataset/BlancoMiguezA_2023_profiles.tsv" \
  --metadata "/data/bwh-comppath-seq/youn/metaphlan_dset/dataset/BlancoMiguezA_2023_metadata.tsv" \
  --out-dir "/data/bwh-comppath-seq/youn/metaphlan_dset/analyses/pcoa_split" \
  --method "pcoa"
