#!/bin/bash
#SBATCH --partition=bwh_comppath_all
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --job-name=train
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

# Note: this is a Slurm script, meant to be run on ErisXDL compute nodes with GPUs.
set -e


ANALYSIS_BASE_DIR="/data/bwh-comppath-seq/youn/metaphlan_dset/analyses"
EMBEDDING_BASE_DIR="/data/bwh-comppath-seq/youn/metaphlan_dset/embeddings/phylophlan"
if ! [ $# -eq 3 ]; then
  echo "Error: analysis_name, embed_model_name, pred_model_name are required"
  echo "Usage: $0 <analysis_name> <embed_model_name> <pred_model_name>"
  echo "Available analysis names:"
  for dir in "${ANALYSIS_BASE_DIR}"/*; do
    echo "-> $(basename "${dir}")"
  done
  echo "Available embedding names:"
  for dir in "${EMBEDDING_BASE_DIR}"/*.npy; do
    echo "-> $(basename "${dir}")"
  done
  exit 1
fi
analysis_name="$1"
embed_model="$2"
pred_model="$3"


# point to the proper pretrained model embeddings
embedding_file="${EMBEDDING_BASE_DIR}/${embed_model}.npy"

analysis_subdir="${ANALYSIS_BASE_DIR}/${analysis_name}"
training_set="${analysis_subdir}/train.tsv"
test_set="${analysis_subdir}/test.tsv"
model_config="./model_${pred_model}.yaml"
if ! [ -f "${model_config}" ]; then
  echo "Model configuration ${model_config} does not exist!"
  exit 1
fi

n_epochs=100
learning_rate=0.0001
batch_size=10
seed=12345

outdir="${analysis_subdir}/trained_model/${embed_model}/${pred_model}"
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
  --embed-memmap-file "${embedding_file}" \
  --epochs "$n_epochs" \
  --learning-rate "$learning_rate" \
  --batch-size "$batch_size" \
  --print-every 5 \
  --workers 10 \
  --seed "$seed" \
  --prefetch-factor 2 \
  --cuda-device "cuda"

echo "Done."
