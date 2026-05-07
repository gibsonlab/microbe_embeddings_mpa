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


EMBEDDING_BASE_DIR="/data/bwh-comppath-seq/youn/metaphlan_dset/embeddings"
ANALYSIS_BASE_DIR="/data/bwh-comppath-seq/youn/metaphlan_dset/analyses"
if ! [ $# -eq 4 ]; then
  echo "Error: analysis_name, embed_model_name, pred_model_name are required"
  echo "Usage: $0 <analysis_name> <embed_family> <embed_model_name> <pred_model_name>"
  echo "Available analysis names:"
  for dir in "${ANALYSIS_BASE_DIR}"/*; do
    echo "-> $(basename "${dir}")"
  done
  echo "Available embedding families:"
  for dir in "${EMBEDDING_BASE_DIR}"/*; do
    echo "-> $(basename "${dir}")"
  done
  exit 1
fi
analysis_name="$1"
embed_family="$2"
embed_model="$3"
pred_model="$4"


# point to the proper pretrained model embeddings
if [ $embed_family == "offline" ]; then
  embedding_file="${EMBEDDING_BASE_DIR}/${embed_family}/${embed_model}.pt"
  # deep-learning embeddings are much larger; can't accommodate large batch sizes.
  batch_size=30
else
  embedding_file="${EMBEDDING_BASE_DIR}/${embed_family}/${embed_model}.ipca_d200.pt"
  batch_size=30
fi
echo "Input embedding file: ${embedding_file}"

analysis_subdir="${ANALYSIS_BASE_DIR}/${analysis_name}"
training_set="${analysis_subdir}/train.tsv"
test_set="${analysis_subdir}/test.tsv"

echo "Training set: ${training_set}"
echo "Test set: ${test_set}"


model_config="./model_${pred_model}.yaml"
if ! [ -f "${model_config}" ]; then
  echo "Model configuration ${model_config} does not exist!"
  exit 1
fi

n_epochs=80
learning_rate=0.0001
seed=12345

outdir="${analysis_subdir}/trained_model/${embed_family}/${embed_model}/${pred_model}"
echo "Target outdir: ${outdir}"
mkdir -p ${outdir}

metadata="$outdir/metadata.txt"
echo "====== Params ======"
echo "epochs=${n_epochs}" | tee $metadata
echo "LR=${learning_rate}" | tee -a $metadata
echo "batch_size=${batch_size}" | tee -a $metadata
echo "seed=${seed}" | tee -a $metadata
echo "===================="


if [[ "$my_string" == *"_epc_"* ]]; then
    echo "Using EPC model type"
    model_type="EPC"
elif [[ "$my_string" == *"_set_transformer"* ]]; then
    echo "Using Set Transformer model type"
    model_type="SetTransformer"
else
    echo "Model type unknown!!!"
    exit 1
fi


if [ "$embed_family" == "offline" ]; then
  echo "Using full-precision model for offline embeddings."
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
  --workers 8 \
  --seed "$seed" \
  --prefetch-factor 2 \
  --model-type "${model_type}" \
  --cuda-device "cuda"
else
  echo "Using bfloat16-precision model for deep-learning per-gene embeddings."
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
  --workers 8 \
  --seed "$seed" \
  --prefetch-factor 2 \
  --cuda-device "cuda" \
  --model-type "${model_type}" \
  --use-bfloat16
fi


echo "Done."
