#!/bin/bash
set -e


BASEDIR="/data/bwh-comppath-seq/youn/human_microbiome_compendium"
dset_dir = "${BASEDIR}/v3v4_split_multiproj_extended"

mkdir -p ./slurm
mkdir -o ./slurm/logs

# Loop through all LOO_* subdirectories
for loo_dir in "${dset_dir}"/LOO_*; do
    # Extract just the directory name (e.g., LOO_PRJNA391858)
    loo_subdir_name=$(basename "${loo_dir}")
    job_script="./slurm/job_${loo_subdir_name}.sh"
    cat > "${job_script}" << EOF
#!/bin/bash
#SBATCH --partition=bwh_comppath
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=${loo_subdir_name}
#SBATCH --output=slurm/logs/${loo_subdir_name}.log
#SBATCH --error=slurm/logs/${loo_subdir_name}.err

# Run the pipeline
bash PIPELINE_all.sh ${loo_subdir_name}
EOF

    # Submit the job
    sbatch --nodelist= "${job_script}"

    echo "Submitted job for ${loo_subdir_name}"
done

