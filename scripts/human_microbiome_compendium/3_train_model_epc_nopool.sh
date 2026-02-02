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

if ! [ $# -eq 2 ]; then
  echo "Error: model_name and dataset is required"
  echo "Usage: $0 <model_name> <dataset>"
  exit 1
fi
model_name="$1"
dataset_name="$2"

# point to the proper pretrained model embeddings
DATA_DIR="/data/bwh-comppath-seq/youn/human_microbiome_compendium/${dataset_name}"
EMBEDDINGS_DIR="${DATA_DIR}/embeddings/"
embeddings_file="${EMBEDDINGS_DIR}/${model_name}.h5"
if ! [ -f "${embeddings_file}" ]; then
  echo "Embeddings for model not found: ${embeddings_file}"
  exit 1
fi

training_set="${DATA_DIR}/train.tsv"
test_set="${DATA_DIR}/test.tsv"

abundance_dir="/data/cctm/youn/human_microbiome_compendium/asv"
model_config="./model_epc_nopool.yaml"
n_epochs=80
learning_rate=0.0001
batch_size=10
seed=12345

outdir="${DATA_DIR}/trained_models/${model_name}/epc_nopool_kl"
mkdir -p ${outdir}

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
  --model-version "EPC"

echo "Done."
