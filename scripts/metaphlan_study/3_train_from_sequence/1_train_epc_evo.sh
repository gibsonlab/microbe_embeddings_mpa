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
  echo "Error: hyena_layers is required to train on Evo embeddings."
  echo "Usage: $0 <hyena_layers>"
  exit 1
fi
model_name="evo"
hyena_layers="$1"

training_set="/data/cctm/youn/metaphlan_dset/model_training/train.tsv"
test_set="/data/cctm/youn/metaphlan_dset/model_training/test.tsv"
marker_sequence_dir="/data/cctm/youn/metaphlan_dset/phylophlan_data/processed/dna_only"
model_config="./model_epc_pool.yaml"
n_epochs=80
learning_rate=0.0001
batch_size=10
seed=12345

outdir="/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/trained_model/${model_name}_hyena${hyena_layers}/epc_sgbpool_epoch${n_epochs}_kl"
echo "Model will be saved to: ${outdir}"
mkdir -p ${outdir}

metadata="$outdir/metadata.txt"
echo "====== Params ======"
echo "epochs=${n_epochs}" | tee $metadata
echo "LR=${learning_rate}" | tee -a $metadata
echo "batch_size=${batch_size}" | tee -a $metadata
echo "seed=${seed}" | tee -a $metadata
echo "===================="

export HF_HOME="/data/cctm/youn/huggingface_cache"
python train_model.py \
  --model-version "EPC" \
  --embedding-model "${model_name}:${hyena_layers}" \
  --train "$training_set" \
  --test "$test_set" \
  --model-config "$model_config" \
  --out-dir "$outdir" \
  --loss "kl" \
  --marker-sequence-dir "$marker_sequence_dir" \
  --epochs "$n_epochs" \
  --learning-rate "$learning_rate" \
  --batch-size "$batch_size" \
  --print-every 5 \
  --seed "$seed" \
  --prefetch-factor 2 \
  --checkpoint-every 10 \
  --cuda-devices "0" \

echo "Done."
