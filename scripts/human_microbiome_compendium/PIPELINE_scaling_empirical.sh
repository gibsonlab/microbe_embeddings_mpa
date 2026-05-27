#!/bin/bash
set -e
set -o pipefail
export PYTHONUNBUFFERED=1


dset_name="american_gut"
analysis_name="pcoa_split"
embed_names=("evo-1-8k-base_hyena5" "evo2_7b_hyena10" "dnabert-s")



### ============================ START model type 1: Sum-pooling models.
## Generate all model config files for empirical results.
#inference_names_all=()
## ===== Models with MLP
##for hidden_dim in $(seq 4 4 256); do
#for hidden_dim in 4 5 7 9 12 15 20 26 34 44 57 64 74 97 126 163 211 274 356 462 512; do
#  model_cfg="epc_pool_scaling${hidden_dim}"
#  inference_names_all+=("${model_cfg}")
#
#  echo "Generating: ${model_cfg}"
#  cfg_file="./model_${model_cfg}.yaml"
#  cat > "${cfg_file}" <<EOF
#sgb_model_dim: 32
#hidden_dim: ${hidden_dim}
#use_sgb_pooling: True
#sgb_pool_dim: 64
#mlp_hidden_layers: 3
#dropout_rate: 0.0
#EOF
#done
#
#
## ===== Models without MLP
##for hidden_dim in $(seq 4 2 128); do
##for hidden_dim in $(seq 4 4 256); do  # dnabert only
#for hidden_dim in 4 5 6 8 10 13 16 20 26 32 33 42 53 68 86 109 138 175 221 279 256; do
#  model_cfg="epc_pool_scaling${hidden_dim}_NO_MLP"
#  inference_names_all+=("${model_cfg}")
#
#  echo "Generating: ${model_cfg}"
#  cfg_file="./model_${model_cfg}.yaml"
#  cat > "${cfg_file}" <<EOF
#sgb_model_dim: 1  # doesn't matter
#sgb_pool_dim: 64  # doesn't matter
#exclude_mlp_reprs: True
#hidden_dim: ${hidden_dim}
#use_sgb_pooling: True
#mlp_hidden_layers: 3
#dropout_rate: 0.0
#EOF
#done
#
#
#for embed_name in "${embed_names[@]}"; do
#  echo "================== Embedding: ${embed_name} (Sum-Pool EPC model) ===================="
#  for seed in 12345 12346 12347; do
#
#    # Train all models.
#    breadcrumb_dir="./scaling_empirical_progress/${embed_name}/${seed}"
#    mkdir -p ${breadcrumb_dir}
#
#    for model_cfg in "${inference_names_all[@]}"; do
#      breadcrumb_file="${breadcrumb_dir}/${model_cfg}.DONE"
#      log_file="${breadcrumb_dir}/${model_cfg}.log"
#      if [ -f ${breadcrumb_file} ]; then
#        echo "Model ${model_cfg} already done!"
#      else
#        echo "Trying model: ${model_cfg}"
#        bash 3_train_model.sh "$dset_name" "$analysis_name" "$embed_name" "$model_cfg" "EPC" "$seed" 2>&1 | tee "${log_file}"
#        touch "${breadcrumb_file}"
#      fi
#    done
#  done
#done
### ============================ END model type 1


# ============================ START model type 2: SetTransformer models.
set_transformer_names=()

for hidden_dim in $(seq 4 4 64); do
  model_cfg="set_transformer_d${hidden_dim}"
  set_transformer_names+=("${model_cfg}")

  echo "Generating: ${model_cfg}"
  cfg_file="./model_${model_cfg}.yaml"
  cat > "${cfg_file}" <<EOF
dim_hidden: ${hidden_dim}
num_inds: 16
num_heads: 4
ln: True
EOF
done

for embed_name in "${embed_names[@]}"; do
  echo "================== Embedding: ${embed_name} (SetTransformer) ===================="
  for seed in 12345 12346 12347; do

    breadcrumb_dir="./scaling_empirical_progress/${embed_name}/${seed}"
    mkdir -p "${breadcrumb_dir}"

    for model_cfg in "${set_transformer_names[@]}"; do
      breadcrumb_file="${breadcrumb_dir}/${model_cfg}.DONE"
      log_file="${breadcrumb_dir}/${model_cfg}.log"
      if [ -f "${breadcrumb_file}" ]; then
        echo "Model ${model_cfg} already done!"
      else
        echo "Trying model: ${model_cfg}"
        bash 3_train_model.sh "$dset_name" "$analysis_name" "$embed_name" "$model_cfg" "SetTransformer" "$seed" 2>&1 | tee "${log_file}"
        touch "${breadcrumb_file}"
      fi
    done
  done
done
