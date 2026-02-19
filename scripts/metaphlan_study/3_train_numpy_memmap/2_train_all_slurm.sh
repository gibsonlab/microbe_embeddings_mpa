#!/bin/bash
set -e

ANALYSIS_BASE_DIR="/data/bwh-comppath-seq/youn/metaphlan_dset/analyses"
if [ $# -lt 1 ]; then
  echo "Error: analysis_name, embed_model_name, pred_model_name are required"
  echo "Usage: $0 <analysis_name> [--dry-run]"
  echo "Available analysis names:"
  for dir in "${ANALYSIS_BASE_DIR}"/*; do
    echo "-> $(basename "${dir}")"
  done
  exit 1
fi
analysis_name="$1"
shift

# Parse options
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done


EXCLUDE_NODES=""
analysis_subdir="${ANALYSIS_BASE_DIR}/${analysis_name}"


# Get list of currently running/pending jobs for this user
#running_jobs=$(squeue -u $USER -h -o "%j" 2>/dev/null)
running_jobs=""


SCRIPT_DIR="./slurm"


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
  if $DRY_RUN; then
      echo "[Dry run] Submit ${jobname}:  sbatch --exclude=\"${EXCLUDE_NODES}\" \"${job_script}\"  -->  bash 2_train.sh \"${analysis_name}\" \"${embed_family}\" \"${embed_model_name}\" \"${pred_model}\""
      return 0
  fi

  # Write SBATCH headers and job commands
  mkdir -p ${job_subdir}
  cat > "${job_script}" << EOF
#!/bin/bash
#SBATCH --partition=bwh_comppath_all
#SBATCH --gpus=1
#SBATCH --mem=40G
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

create_job_submission "pcoa_split" "offline" "pcoa_d100_s1000" "epc_pool"
create_job_submission "pcoa_split" "offline" "pcoa_d100_s1000" "epc_nopool"

create_job_submission "pcoa_split" "offline" "umap_d100_s1000" "epc_pool"
create_job_submission "pcoa_split" "offline" "umap_d100_s1000" "epc_nopool"

create_job_submission "pcoa_split" "phylophlan" "dnabert-s" "epc_pool"
create_job_submission "pcoa_split" "phylophlan" "dnabert-s" "epc_nopool"

create_job_submission "pcoa_split" "phylophlan" "evo2_7b_hyena10" "epc_pool"
create_job_submission "pcoa_split" "phylophlan" "evo2_7b_hyena10" "epc_nopool"

