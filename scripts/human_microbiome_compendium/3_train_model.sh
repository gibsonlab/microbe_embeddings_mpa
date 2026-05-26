#!/bin/bash
#SBATCH --partition=bwh_comppath
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=5-00:00:00
#SBATCH --job-name=train_epc
#SBATCH --output=train_epc_%A_%a.out
#SBATCH --error=train_epc_%A_%a.err

# Note: this is a Slurm script, meant to be run on ErisXDL compute nodes with GPUs.
set -e

if ! [ $# -eq 6 ]; then
  echo "Error: dataset, analysis_name, embed_model_name, pred_model_cfg, pred_model_name, seed are required"
  echo "Usage: $0 <dataset> <analysis_name> <embed_model_name> <pred_model_cfg> <pred_model_name> <seed>"
  exit 1
fi
dataset_name="$1"
analysis_name="$2"
embed_model_name="$3"
pred_model_cfg_name="$4"
pred_model_name="$5"
seed="$6"

echo "Performing prediction model training for ${embed_model_name} on dataset ${dataset_name} (${pred_model_name}: ${pred_model_cfg_name})"

# point to the proper pretrained model embeddings
BASEDIR="/data/bwh-comppath-seq/youn/human_microbiome_compendium"
DATA_DIR="${BASEDIR}/${dataset_name}"
embeddings_file="${DATA_DIR}/embeddings/${embed_model_name}.h5"
if ! [ -f "${embeddings_file}" ]; then
  echo "Embeddings for model not found: ${embeddings_file}"
  exit 1
else
  echo "Embeddings file: ${embeddings_file}"
fi

ANALYSIS_DIR="${DATA_DIR}/analyses/${analysis_name}"
training_set="${ANALYSIS_DIR}/train.tsv"
test_set="${ANALYSIS_DIR}/test.tsv"
if ! [ -f "${test_set}" ]; then
  echo "test.tsv not found in ${ANALYSIS_DIR}"
fi
if ! [ -f "${training_set}" ]; then
  echo "training.tsv not found in ${ANALYSIS_DIR}"
fi


abundance_dir="/data/cctm/youn/human_microbiome_compendium/asv"
model_config="./model_${pred_model_cfg_name}.yaml"
n_epochs=80
learning_rate=0.0001
batch_size=30



outdir="${ANALYSIS_DIR}/trained_models/${embed_model_name}/${pred_model_cfg_name}_kl/seed_${seed}"
echo "Model output dir: ${outdir}"
mkdir -p "${outdir}"

metadata="$outdir/metadata.txt"
echo "====== Params ======"
echo "epochs=${n_epochs}" | tee $metadata
echo "LR=${learning_rate}" | tee -a $metadata
echo "batch_size=${batch_size}" | tee -a $metadata
echo "seed=${seed}" | tee -a $metadata
echo "===================="

python train_model.py \
  --train "$training_set" \
  --test "$test_set" \
  --model-config "$model_config" \
  --out-dir "$outdir" \
  --loss "kl" \
  --abundance-tables "${abundance_dir}" \
  --embedding-h5 "${embeddings_file}" \
  --epochs "$n_epochs" \
  --learning-rate "$learning_rate" \
  --batch-size "$batch_size" \
  --print-every 5 \
  --checkpoint-every 10 \
  --workers 1 \
  --seed "$seed" \
  --prefetch-factor 2 \
  --cuda-device "cuda" \
  --model-version "$pred_model_name"

echo "Done."
