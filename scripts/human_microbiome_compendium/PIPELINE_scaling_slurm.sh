#!/bin/bash
set -e
set -o pipefail
export PYTHONUNBUFFERED=1



EXCLUDE_NODES="lmd-[2,4]"
dset_name="american_gut"
analysis_name="pcoa_split"
embed_names=("evo-1-8k-base_hyena5" "evo2_7b_hyena10" "dnabert-s")



set_transformer_names=()
for hidden_dim in $(seq 4 4 64); do
  model_cfg="set_transformer_d${hidden_dim}"
  set_transformer_names+=("${model_cfg}")
  echo "Generating: ${model_cfg}"
  cfg_file="./model_${model_cfg}.yaml"
  cat > "${cfg_file}" <<EOF
dim_hidden: ${hidden_dim}
num_inds: 8
num_heads: 4
ln: True
EOF
done


# Snapshot currently queued/running job names once, up front.
running_jobs="$(squeue -h -u "$USER" -o '%j')"

for embed_name in "${embed_names[@]}"; do
  echo "================== Embedding: ${embed_name} (SetTransformer) ===================="
  breadcrumb_dir="./scaling_empirical_progress/${dset_name}/${analysis_name}/${embed_name}"
  mkdir -p "${breadcrumb_dir}"

  for model_cfg in "${set_transformer_names[@]}"; do
    breadcrumb_file="${breadcrumb_dir}/${model_cfg}.DONE"

    if [ -f "${breadcrumb_file}" ]; then
      echo "Model ${model_cfg} already done!"
      continue
    fi

    jobname="${dset_name}_${analysis_name}_${embed_name}_${model_cfg}"

    if grep -Fxq "${jobname}" <<< "${running_jobs}"; then
      echo "Job ${jobname} already in squeue, skipping."
      continue
    fi

    slurm_dir="./slurm_scaling/${jobname}"
    mkdir -p "${slurm_dir}"

    logfile="${slurm_dir}/%j.out"
    errfile="${slurm_dir}/%j.err"

    echo "Submitting model: ${model_cfg}"
    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=bwh_comppath_all
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=${jobname}
#SBATCH --output=${logfile}
#SBATCH --error=${errfile}
#SBATCH --exclude=${EXCLUDE_NODES}

bash 3_train_model.sh "${dset_name}" "${analysis_name}" "${embed_name}" "${model_cfg}" "SetTransformer"
touch "${breadcrumb_file}"
EOF
  done
done
