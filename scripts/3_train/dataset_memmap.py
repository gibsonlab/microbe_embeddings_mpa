"""
Pre-compute the tensors in the dataset, and convert it into memory-mapped tensordicts.
This is meant to reduce the time it takes to dynamically re-alloate memory for each sample.
"""

"""
From https://docs.pytorch.org/tensordict/main/saving.html

1) Saving a memmapped tensordict:
    x = TensorDict()
    x_disk = x.memmap("/path/to/saved/dir", num_threads=30)

2) Loading a memmapped tensordict:
    x = TensorDict.load_memmap("/path/to/saved/dir")
"""
import argparse
from pathlib import Path

import pandas as pd
from gem.mpa import MetaphlanMarkerEmbedding
from gem.mpa import MetaphlanDatasetMemmapped


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset-tsv", dest="dataset_tsv", required=True, type=str)
    parser.add_argument("-e", "--embedding-dir", dest="marker_embedding_basedir", required=True, type=str)
    parser.add_argument("-m", "--memmap-dir", dest="memmap_dir", required=True, type=str)
    return parser.parse_args()


def main(
        dataset_df: pd.DataFrame,
        marker_embedding: MetaphlanMarkerEmbedding,
        memmap_dir: Path,
):
    memmap_dir.mkdir(parents=True, exist_ok=True)
    MetaphlanDatasetMemmapped(
        dataset_df=dataset_df,
        marker_embedding=marker_embedding,
        cache_dir=memmap_dir,
    )
    print("Finished memory-mapping tensors.")


if __name__ == "__main__":
    args = parse_args()
    main(
        dataset_df=pd.read_csv(args.dataset_tsv),
        marker_embedding=MetaphlanMarkerEmbedding(marker_embedding_basedir=args.marker_embedding_basedir),
        memmap_dir=Path(args.memmap_dir),
    )