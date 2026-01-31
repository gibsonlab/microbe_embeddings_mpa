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

if [ $# -eq 0 ]; then
  echo "Error: model_name is required"
  echo "Usage: $0 <model_name>"
  exit 1
fi
model_name="$1"
pca_dim=200

# point to the proper pretrained model embeddings
embeddings_memmap="/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/memmap_samples/${model_name}_d${pca_dim}"


training_set="/data/cctm/youn/metaphlan_dset/model_training/train.tsv"
test_set="/data/cctm/youn/metaphlan_dset/model_training/test.tsv"
model_config="./model_epc_nopool.yaml"
n_epochs=80
learning_rate=0.0001
batch_size=10
seed=12345

outdir="/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/trained_model/${model_name}_d${pca_dim}_epc/sgbpool_epoch${n_epochs}_kl"
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
  --memmap-tensor-dir "$embeddings_memmap" \
  --epochs "$n_epochs" \
  --learning-rate "$learning_rate" \
  --batch-size "$batch_size" \
  --print-every 5 \
  --checkpoint-every 10 \
  --workers 10 \
  --seed "$seed" \
  --prefetch-factor 2 \
  --cuda-device "cuda" \
  --model-version "EPC"

echo "Done."
