"""
Convert the HDF5-stored tensors in step 1, and convert it into memory-mapped tensordicts.

This step was separated from the previous one, for safety.
The alternative is to bypass the HDF5 storage altogether, opting for only the memmapped tensors.
However, note that these files can accidentally be changed; the extra layer introduces some room for user error.
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
import h5py
import torch
from tensordict import TensorDict
from tqdm import tqdm

import sys
import logging
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-embed-dir", dest="marker_embed_root_dir", type=str)
    parser.add_argument("-o", "--out-memmap-dir", dest="memmap_target_dir", type=str)
    parser.add_argument(
        "-t", "--threads", dest="num_memmap_threads", type=int,
        required=False, default=1
    )
    return parser.parse_args()


def populate_embeddings(
        tdict: TensorDict,
        marker_embedding_root_dir: Path,
        embed_dtype,
) -> pd.DataFrame:
    """
    Return a dictionary mapping SGB IDs to a list of SGB-specific markers.
    Also populates (in-place) the tensordict, mapping marker IDs to marker embeddings.
    :param tdict: The empty tensordict to populate.
    :param marker_embedding_root_dir: The root dir containing the "part" subdirs ("complete" subdir not implemented! TODO).
    :param embed_dtype: The dtype to use to store the tensors.
    :return: A dataframe containing 'SGB', 'Protein', 'Marker' columns.
    """
    df_sections = []
    for embed_dir in sorted(marker_embedding_root_dir.glob("part*")):
        logger.info(f"Reading subdirectory {embed_dir.name}")
        assert (embed_dir / ".embed.DONE").exists(), f"Embedding in ({embed_dir}) was not finished."
        populate_tensordict_embeddings(embed_dir, tdict, embed_dtype)
        df_subdir = fetch_embeddings_index(embed_dir)

        # Validate the dataframe -- check if each marker exists.
        for marker_name in df_subdir['Marker']:
            assert marker_name in tdict, f"Marker {marker_name} found in index, but not in tensordict!"
        df_sections.append(df_subdir)
    return pd.concat(df_sections, axis=0, ignore_index=True)


def populate_tensordict_embeddings(embed_dir: Path, tdict: TensorDict, embed_dtype):
    # Read each shard file, one by one.
    for shard_file in embed_dir.glob("shard-*.h5"):
        logger.info(f"Reading shard file {shard_file}")
        shard = h5py.File(shard_file, "r")
        shard_size = len(shard.keys())

        for marker_id in tqdm(shard.keys(), total=shard_size, desc="{embed_dir.name}/{shard_file.name}"):
            marker_embedding = torch.tensor(
                shard[marker_id],
                dtype=embed_dtype,
                device="cpu"
            )
            tdict[marker_id] = marker_embedding
        logger.info(f"Populated {shard_size} marker keys from shard {shard_file.name}")


def fetch_embeddings_index(embed_dir: Path):
    df = pd.read_csv(embed_dir / "index.tsv", sep='\t')

    # Parse the Marker ID Name, encoded in a previous stage (1_preprocess/1_degap_alignments.py)
    first_split = df['Marker'].str.split(":").str
    df['Protein'] = first_split[0]
    second_split = first_split[1].str.split("__").str
    df['SGB'] = second_split[0]

    return_col_order = ['SGB', 'Marker']
    return df[return_col_order]


def main(
        memmap_dir: Path,
        marker_embed_root_dir: Path,
        num_memmap_threads: int,
        embed_dtype,
):
    if memmap_dir.exists():
        logger.info(f"Target memmap dir {memmap_dir} doesn't yet exist. It will be created.")
        memmap_dir.mkdir(parents=True, exist_ok=True)

    embed_tdict = TensorDict()
    marker_df = populate_embeddings(
        embed_tdict,
        marker_embed_root_dir,
        embed_dtype=embed_dtype,
    )

    # Apply memory-mapping.
    logger.info(f"Populated {len(embed_tdict)} embedding tensors total.")
    logger.info(f"Performing memory-mapping on disk. Destination = {memmap_dir}")
    memmap_dir.mkdir(exist_ok=True)
    embed_tdict.memmap(
        str(memmap_dir),
        num_threads=num_memmap_threads
    )

    # Save dataframe.
    df_path = memmap_dir / "embedding_index.parquet"
    marker_df.to_parquet(df_path)
    logger.info(f"Saved dataframe to disk: {df_path}")


if __name__ == "__main__":
    args = parse_args()
    main(
        memmap_dir=Path(args.memmap_target_dir),
        marker_embed_root_dir=Path(args.marker_embed_root_dir),
        num_memmap_threads=args.num_memmap_threads,
        embed_dtype=torch.float32,
    )
