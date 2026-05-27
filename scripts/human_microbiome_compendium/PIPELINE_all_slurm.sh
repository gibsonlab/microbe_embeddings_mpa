#!/bin/bash
set -e


BASEDIR="/data/bwh-comppath-seq/youn/human_microbiome_compendium"
# Suggest dataset subdir names.
if [[ $# -lt 1 ]]; then
  echo "Error: dataset is required"
  echo "Usage: $0 <dataset> [--dry-run]"
  echo "Available dataset names:"
  for dir in "${BASEDIR}"/*; do
    echo "-> $(basename "${dir}")"
  done
  exit 1
fi

# Get positional argument
dset_name=$1
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

# ======================= Script body starts here ====================

dset_dir="${BASEDIR}/${dset_name}"
EXCLUDE_NODES="lmd-2,lmd-4"

# Get list of currently running/pending jobs for this user
running_jobs=$(squeue -u $USER -h -o "%j" 2>/dev/null)

SCRIPT_DIR="./slurm/${dset_name}"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${SCRIPT_DIR}"
mkdir -p ${LOG_DIR}

# Loop through all analysis subdirectories
for analysis_subdir in "${dset_dir}"/analyses/*; do
    if ! [ -d "${analysis_subdir}" ]; then
        echo "${analysis_subdir} is not a directory. Skipping."
        continue
    fi

    # Extract just the directory name (e.g., LOO_PRJNA391858)
    analysis_name=$(basename "${analysis_subdir}")

    # Check if a job with this name is already running or pending
    if echo "$running_jobs" | grep -q "^${analysis_name}$"; then
        echo "Skipping ${analysis_name} - job already running/pending."
        continue
    fi

    # Check if the job already finished previously.
    BREADCRUMB_FILE="${LOG_DIR}/${analysis_name}.DONE"
    if [ -f "${BREADCRUMB_FILE}" ]; then
        echo "Skipping ${analysis_name} - job already finished."
        continue
    fi

    # Create a job script for this subdirectory
    jobname="${dset_name}:${analysis_name}"
    job_script="${SCRIPT_DIR}/${jobname}.sh"
    if $DRY_RUN; then
        echo "[Dry run] Submit ${jobname}:  sbatch --exclude=\"${EXCLUDE_NODES}\" \"${job_script}\"  -->  bash PIPELINE_all.sh \"${dset_name}\" \"${analysis_name}\""
        continue
    fi

    logfile="${LOG_DIR}/${dset_name}__${analysis_name}.out"
    errfile="${LOG_DIR}/${dset_name}__${analysis_name}.err"
    rm -f "${logfile}" "${errfile}"

    # Write SBATCH headers and job commands
    cat > "${job_script}" << EOF
#!/bin/bash
#SBATCH --partition=bwh_comppath_all
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=${analysis_name}
#SBATCH --output=${logfile}
#SBATCH --error=${errfile}
set -e

# Run the pipeline
bash PIPELINE_all.sh "${dset_name}" "${analysis_name}"
touch "${BREADCRUMB_FILE}"
EOF

    # Submit the job
    sbatch --exclude="${EXCLUDE_NODES}" "${job_script}"

    echo "Submitted job for dset ${dset_name}, analysis ${analysis_name}"
done

