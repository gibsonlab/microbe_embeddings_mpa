#!/bin/bash
set -e

EXCLUDE_NODES="lmd-2,lmd-3,lmd-4"
SCRIPT_DIR="./slurm"


# Get list of currently running/pending jobs for this user
#running_jobs=$(squeue -u $USER -h -o "%j" 2>/dev/null)
running_jobs=""


create_job_submission () {
  analysis_name=$1
  embed_family=$2
  embed_model_name=$3
  pred_model=$4
  cur_running_jobs=$5

  jobname="${analysis_name}:${embed_family}:${embed_model_name}:${pred_model}"

  job_subdir="${SCRIPT_DIR}/${analysis_name}/${embed_family}/${embed_model_name}/${pred_model}"

  logfile="${job_subdir}/%j.out"
  errfile="${job_subdir}/%j.err"
  job_script="${job_subdir}/job.sh"
  breadcrumb="${job_subdir}/job.DONE"

  if echo "$cur_running_jobs" | grep -q "^${jobname}$"; then
      echo "Skipping ${jobname} - job already running/pending."
      return 0
  fi
  if [ -f "${breadcrumb}" ]; then
      echo "Skipping ${jobname} - job already finished."
      return 0
  fi

  # Write SBATCH headers and job commands
  mkdir -p ${job_subdir}
  cat > "${job_script}" << EOF
#!/bin/bash
#SBATCH --partition=bwh_comppath_all
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=${jobname}
#SBATCH --output=${logfile}
#SBATCH --error=${errfile}
set -e

# Run the pipeline
bash 2_train.sh "${analysis_name}" "${embed_family}" "${embed_model_name}" "${pred_model}"
touch "${breadcrumb}"
EOF

  # Submit the job
  sbatch --exclude="${EXCLUDE_NODES}" "${job_script}"
  echo "Submitted job ${jobname}"
}

#create_job_submission "pcoa_split" "offline" "pcoa_d100_s1000" "epc_pool" "$running_jobs"
#create_job_submission "pcoa_split" "offline" "pcoa_d100_s1000" "epc_nopool" "$running_jobs"
#
#create_job_submission "pcoa_split" "offline" "umap_d100_s1000" "epc_pool" "$running_jobs"
#create_job_submission "pcoa_split" "offline" "umap_d100_s1000" "epc_nopool" "$running_jobs"
#
#create_job_submission "pcoa_split" "phylophlan" "dnabert-s" "epc_pool" "$running_jobs"
#create_job_submission "pcoa_split" "phylophlan" "dnabert-s" "epc_nopool" "$running_jobs"
#
#create_job_submission "pcoa_split" "phylophlan" "evo2_7b_hyena10" "epc_pool" "$running_jobs"
#create_job_submission "pcoa_split" "phylophlan" "evo2_7b_hyena10" "epc_nopool" "$running_jobs"

for analysis_name in "pcoa_split" "random_split_1001" "random_split_1002" "random_split_1003" "random_split_1004" "random_split_1005" "random_split_1006" "random_split_1007" "random_split_1008" "random_split_1009" "random_split_1010"; do
  for embed_type in "phylophlan" "phylophlan_metaphlan"; do
    for embed_name in "dnabert-s" "evo-1-8k-base_hyena5" "evo2_7b_hyena10" "pcoa_d100_s1000" "umap_d100_s1000"; do
      for pred_model in "epc_pool" "epc_nopool"; do
        create_job_submission "$analysis_name" "$embed_type" "$embed_name" "$pred_model" "$running_jobs"
      done
    done
  done
done
#create_job_submission "pcoa_split" "phylophlan" "dnabert-s" "epc_pool" "$running_jobs"
#create_job_submission "pcoa_split" "phylophlan_metaphlan" "dnabert-s" "epc_pool" "$running_jobs"
#create_job_submission "random_split_1001" "phylophlan" "dnabert-s" "epc_pool" "$running_jobs"
#create_job_submission "random_split_1001" "phylophlan_metaphlan" "dnabert-s" "epc_pool" "$running_jobs"
#create_job_submission "random_split_1002" "phylophlan" "dnabert-s" "epc_pool" "$running_jobs"
#create_job_submission "random_split_1002" "phylophlan_metaphlan" "dnabert-s" "epc_pool" "$running_jobs"

