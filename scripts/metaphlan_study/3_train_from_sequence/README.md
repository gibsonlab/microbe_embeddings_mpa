# About
This directory contains the pipeline scripts for training models, where embeddings are computed on-the-fly during 
training.

# Prerequisites:
- `1_preprocess` (whole pipeline)

Note that `2_embed` is completely unnecessary for this pipeline.

# Instructions
1) Run `1_train_test_split.py` from the `3_train` directory.
2) Run the desired `2_train_*.sh` scripts.
3) Move onto `4_eval`.
