#!/bin/bash

#for dset_name in "american_gut_USCA"; do
for dset_name in "todo!!!!"; do
  for embed_name in "evo-1-8k-base_hyena5" "evo2_7b_hyena10" "dnabert-s"; do
    bash 2_embed.sh "$embed_name" "$dset_name"
    bash 3_train_model_epc.sh "$embed_name" "$dset_name"
    bash 3_train_model_epc_nopool.sh "$embed_name" "$dset_name"
  done
done
