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

import h5py
import pandas as pd
from gem.mpa import MetaphlanMarkerEmbedding
from gem.mpa import MetaphlanDataset
from tqdm import tqdm


def hdf5_convert_dataset(dataset: MetaphlanDataset, out_path: Path, chunk_size: int = 32):
    max_S = dataset.max_num_sgbs()
    max_M = dataset.max_num_markers()
    embed_dim = dataset.embed_feature_dim()
    n_samples = len(dataset.samples)

    with h5py.File(out_path, "w") as f:
        f.create_dataset(
            'features',
            shape=(n_samples, max_S, max_M, embed_dim),
            dtype='float32',
            chunks=(chunk_size, max_S, max_M, embed_dim),
            compression='lzf'
        )
        f.create_dataset(
            'mpadding',
            shape=(n_samples, max_S, max_M),
            dtype='bool',
            chunks=(chunk_size, max_S, max_M)
        )
        f.create_dataset(
            'spadding',
            shape=(n_samples, max_S),
            dtype='bool',
            chunks=(chunk_size, max_S)
        )
        f.create_dataset(
            'targets',
            shape=(n_samples, max_S),
            dtype='float32',
            chunks=(chunk_size, max_S),
            compression='lzf'
        )

    # Fill data
    for i, sample in enumerate(tqdm(dataset.samples, desc="Dataset HDF5 conversion")):
        _, features, marker_padding_mask, sgb_padding_mask, targets = dataset.load_sample_embeddings(sample)
        S, M = features.shape[0], features.shape[1]
        f['features'][i, :S, :M, :] = features.numpy()
        f['mpadding'][i, :S, :M] = marker_padding_mask.numpy()
        f['spadding'][i, :S] = sgb_padding_mask.numpy()
        f['targets'][i, :S] = targets.numpy()

    f.create_dataset(
        'sample_ids',
        data=[s.sample_id.encode() for s in dataset.samples]
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset-tsv", dest="dataset_tsv", required=True, type=str)
    parser.add_argument("-e", "--embedding-dir", dest="marker_embedding_basedir", required=True, type=str)
    parser.add_argument("-o", "--out-path", dest="out_path", required=True, type=str)
    parser.add_argument("--dimension-reduce", dest="dimension_reduce_pca", required=False, default=None, type=int,
                        help="If specified (an integer greater than zero), will perform incremental PCA on the entire"
                             "set of embeddings for dimensionality reduction.")
    parser.add_argument("--pca-batch-size", dest="ipca_batch_size", required=False, default=10000, type=int,
                        help="Specify the batch size for incremental PCA. Default: 10000")
    return parser.parse_args()


def main(
        dataset_df: pd.DataFrame,
        marker_embedding: MetaphlanMarkerEmbedding,
        out_path: Path,
):
    if out_path.exists():
        print(f"Target file ({out_path}) already exists. Exiting.")
        exit(0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    regular_dset = MetaphlanDataset(dataset_df, marker_embedding)
    hdf5_convert_dataset(
        dataset=regular_dset,
        out_path=out_path,
    )
    print("Finished memory-mapping tensors.")


if __name__ == "__main__":
    args = parse_args()
    dataset_df = pd.read_csv(args.dataset_tsv, sep='\t', index_col="SampleID")

    marker_embedding = MetaphlanMarkerEmbedding(
        marker_embedding_basedir=Path(args.marker_embedding_basedir),
        dimension_reduce_pca=args.dimension_reduce_pca,
        ipca_batch_size=args.ipca_batch_size,
    )
    main(
        dataset_df=dataset_df,
        marker_embedding=marker_embedding,
        out_path=Path(args.out_path),
    )