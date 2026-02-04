# Human Microbiome Compendium Analysis -- README

Please follow the below directions for performing the analysis.

1) `bash 0_download_dataset.sh` - download and pre-process the dataset files from Zenodo.
2) Run each of the jupyter notebooks `1_*.ipynb`, which extracts the appropriate train-test split files and produces the necessary intermediate dataset files for the rest of the pipeline.

At this point, you have a choice:

- Run `PIPELINE_all.sh` to run all of the model embeddings & training in a single sequential loop (CUDA required!)
- Or, if you want to run the steps separately (e.g. on an HPC cluster), run the pipeline components individually. Namely:
  - `2_embed.sh <embed_model_name> <dataset_name>`
  - `3_train_model_epc.sh <embed_model_name>`

# Note about Evo2
Evo2 models (embed_model_name of the form `evo2_*`) are meant to be run from an Apptainer (formerly known as Singularity) image.

Namely, invoking "evo2" for the embedding step (`2_embed.sh`) will attempt to load the pre-built .sif image.
The .sif image should be built using the separately-provided pipeline located in `setup/evo2` of this project.
