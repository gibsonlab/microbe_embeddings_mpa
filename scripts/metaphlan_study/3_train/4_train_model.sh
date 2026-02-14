#!/bin/bash
#SBATCH --partition=bwh_comppath
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=5-00:00:00
#SBATCH --job-name=train
#SBATCH --output=train_%A_%a.out
#SBATCH --error=train_%A_%a.err

# Note: this is a Slurm script, meant to be run on ErisXDL compute nodes with GPUs.
set -e

#if [ $# -eq 0 ]; then
#  echo "Error: embedding_model is required"
#  echo "Usage: $0 <embedding_model>"
#  exit 1
#fi
#embedding_model="$1"

if ! [ $# -eq 3 ]; then
  echo "Error: analysis_name, embed_model_name, pred_model_name are required"
  echo "Usage: $0 <analysis_name> <embed_model_name> <pred_model_name>"
  exit 1
fi
analysis_name="$1"
embedding_model="$2"
pred_model="$3"

pca_dim=200

# point to the proper pretrained model embeddings
embeddings_memmap="/data/bwh-comppath-seq/youn/metaphlan_dset/memmap_samples/dna/${embedding_model}_d${pca_dim}"


training_set="/data/cctm/youn/metaphlan_dset/${analysis_name}/train.tsv"
test_set="/data/cctm/youn/metaphlan_dset/${analysis_name}/test.tsv"
model_config="./model_${pred_model}.yaml"
if ! [ -f "${model_config}" ]; then
  echo "Model configuration ${model_config} does not exist!"
  exit 1
fi

n_epochs=80
learning_rate=0.0001
batch_size=10
seed=12345

outdir="/data/bwh-comppath-seq/youn/metaphlan_dset/${analysis_name}/trained_model/dna/${embedding_model}_d${pca_dim}/${pred_model}"
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
  --workers 10 \
  --seed "$seed" \
  --prefetch-factor 2 \
  --cuda-device "cuda"

echo "Done."
