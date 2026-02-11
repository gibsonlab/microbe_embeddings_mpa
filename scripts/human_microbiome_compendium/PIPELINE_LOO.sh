#!/bin/bash
set -e


BASEDIR="/data/bwh-comppath-seq/youn/human_microbiome_compendium"
dset_dir="${BASEDIR}/v3v4_split_multiproj_extended"
NODELIST="lmd-2,lmd-3"
cwd="$(pwd)"

# Get list of currently running/pending jobs for this user
running_jobs=$(squeue -u $USER -h -o "%j" 2>/dev/null)

mkdir -p ./slurm
mkdir -p ./slurm/logs

# Loop through all LOO_* subdirectories
cd .slurm
for loo_dir in "${dset_dir}"/LOO_*; do
    # Extract just the directory name (e.g., LOO_PRJNA391858)
    loo_subdir_name=$(basename "${loo_dir}")

    # Check if a job with this name is already running or pending
    if echo "$running_jobs" | grep -q "^${loo_subdir_name}$"; then
        echo "Skipping ${loo_subdir_name} - job already running/pending"
        continue
    fi

    # Create a job script for this subdirectory
    job_script="job_${loo_subdir_name}.sh"

    # Write SBATCH headers and job commands
    cat > "${job_script}" << EOF
#!/bin/bash
#SBATCH --partition=bwh_comppath_all
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=${loo_subdir_name}
#SBATCH --output=logs/${loo_subdir_name}.log
#SBATCH --error=logs/${loo_subdir_name}.err
#SBATCH --nodelist=${NODELIST}

# Run the pipeline
bash ${cwd}/PIPELINE_all.sh ${loo_subdir_name}
touch logs/${loo_subdir_name}.DONE
EOF

    # Submit the job
    sbatch "${job_script}"

    echo "Submitted job for ${loo_subdir_name}"
    break
done

