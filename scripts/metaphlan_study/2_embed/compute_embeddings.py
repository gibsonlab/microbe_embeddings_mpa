"""
Script for pre-computing evo embeddings for marker genes.
"""
import math
import logging
import sys
import argparse
from typing import *
from pathlib import Path
import json

from numpy.lib.recfunctions import assign_fields_by_name
from torch.cuda import OutOfMemoryError
from tqdm import tqdm
import h5py
from pyfaidx import Fasta
import torch
import zstandard as zstd

from gem.glms import GenomeEmbedding, EvoWrapper, DNABertSWrapper


logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model_name", type=str)
    parser.add_argument("-f", "--fasta", dest="fasta_path", type=str)
    parser.add_argument("-sl", "--sgb-list", dest="sgb_subset_file", type=str)
    parser.add_argument("-si", "--sgb-index-file", dest="sgb_marker_index_file", type=str)
    parser.add_argument("-s", "--start", dest="start", type=int, help="the first index (1-indexed, inclusive) of the marker to embed.")
    parser.add_argument("-e", "--end", dest="end", type=int, help="the final index (1-indexed, inclusive) of the marker to embed.")
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int)
    parser.add_argument("-o", "--out-dir", dest="out_dir", type=str)
    parser.add_argument("-sz", "--shard-size", dest="shard_size", type=int)
    return parser.parse_args()


def compute_embedding(
        sgb_subset: List[str],
        sgb_marker_index: Dict[str, Dict],
        fasta_path: Path,
        create_embed_model: Callable[[], GenomeEmbedding],
        h5_output_dir: Path,
        batch_size: int,
        shard_size: int,
        **embed_kwargs
):
    """
    :param create_embed_model:
    :param fai_path:
    :param start_idx: The start index (inclusive) for the marker to embed.
    :param end_idx: The end index (exclusive) for the marker to embed.
    :return:
    """

    fasta = Fasta(fasta_path)
    marker_ids_subset = []
    n_skipped_sgbs = 0
    for sgb_id in sgb_subset:
        sgb_id_numeric_str = sgb_id[3:]
        if sgb_id_numeric_str not in sgb_marker_index:
            logger.warning(f"Key {sgb_id_numeric_str} (derived from {sgb_id}) not found in sgb marker index. Skipping.")
            n_skipped_sgbs += 1
        else:
            marker_ids_subset += sgb_marker_index[sgb_id_numeric_str]

    n_seqs = len(marker_ids_subset)
    logger.info("{} SGBs [{} skipped] --> {} sequences.".format(len(sgb_subset), n_skipped_sgbs, n_seqs))
    logger.info("Desired shard size is {} sequences.".format(shard_size))

    example_marker_id = marker_ids_subset[0]
    example_seq = str(fasta[example_marker_id])
    logger.info(f"Example sequence: {example_marker_id} --> {example_seq}")

    # divide dataframe into shards.
    index_path = h5_output_dir / "index.tsv"

    logger.info("Loading embedding model.")
    embedding_model = create_embed_model()
    padding_embedding = embedding_model.embed_empty_sequence().cpu().float().numpy()
    logger.info("Got padding embedding of shape {}".format(padding_embedding.shape))

    logger.info("Target # shards = {}".format(math.ceil(n_seqs / shard_size)))
    pbar = tqdm(total=n_seqs)

    with open(index_path, "wt") as index_file:
        index_file.write("Marker\tShard\n")
        for shard_idx, shard_start in enumerate(range(0, n_seqs, shard_size)):
            logger.info(f"Opening shard #{shard_idx}")
            marker_id_shard_subset = marker_ids_subset[shard_start:shard_start + shard_size]
            compute_embedding_shard(
                marker_id_shard_subset,
                fasta,
                batch_size,
                embedding_model,
                embed_kwargs,
                h5_output_dir / f"shard-{shard_idx}.h5",
                pbar,
            )

            logger.info(f"Writing index for shard #{shard_idx}")
            for marker_id in marker_id_shard_subset:
                index_file.write(
                    "{}\t{}\n".format(marker_id, shard_idx)
                )


def compute_embedding_shard(
        marker_id_subset: List[str],
        fasta: Fasta,
        batch_size: int,
        embedding_model: GenomeEmbedding,
        embed_kwargs: dict,
        h5_output_path: Path,
        pbar: tqdm
):
    n_seqs_shard = len(marker_id_subset)

    with h5py.File(h5_output_path, 'w') as h5_file:
        # Next, embed the marker sequences in batches.
        for batch_idx, _i in enumerate(range(0, n_seqs_shard, batch_size)):
            marker_ids_batch = marker_id_subset[_i:_i + batch_size]
            marker_seqs = [str(fasta[m_id]) for m_id in marker_ids_batch]
            try:
                batch_embeddings = embedding_model.embed_batch(marker_seqs, **embed_kwargs).cpu().float().numpy()  # shape (batch_len, embed_dim)
                for marker_id, marker_embedding in zip(marker_ids_batch, batch_embeddings):  # each marker gets its own hdf5 entry.
                    h5_file.create_dataset(marker_id, data=marker_embedding, compression='lzf')
            except RuntimeError as e:
                logger.error("Encountered unexpected error while computing batched embedding. Reverting to unbatched mode for this batch. Error message: %s", e)
                for marker_id, marker_seq in zip(marker_ids_batch, marker_seqs):
                    try:
                        marker_embedding = embedding_model.embed_sequence(marker_seq).cpu().float().numpy()
                    except RuntimeError as e:
                        if "out of memory" in str(e) or "CUDA" in str(e):
                            print(f"CUDA OOM: {e}")
                            raise
                        else:
                            logger.error("For some reason, was unable to embed marker {marker_id} -- Skipping. Error message: %s",e)
                    else:
                        h5_file.create_dataset(marker_id, data=marker_embedding, compression='lzf')

            pbar.update(len(marker_ids_batch))


def do_job(
        model_name: str,
        fasta_path: Path,
        sgb_subset_file: Path,
        sgb_marker_index_file: Path,
        start_idx: int,
        end_idx: int,
        batch_size: int,
        out_dir: Path,
        shard_size: int,
):
    if model_name.startswith("evo-1"):
        tokens = model_name.split(":")
        if len(tokens) == 1:
            evo_checkpoint_name = tokens[0]
            num_hyena_layers = 32
        elif len(tokens) == 2:
            evo_checkpoint_name = tokens[0]
            num_hyena_layers = int(tokens[-1])
        else:
            raise RuntimeError("Incorrect model name syntax. Expected '<evo_checkpoint_name>:<n_layers>', but got {} instead.".format(model_name))

        model_fn = lambda: EvoWrapper(device=torch.device('cuda'), num_hyena_layers=num_hyena_layers, checkpoint_name=evo_checkpoint_name)
    elif model_name == 'dnabert-s':
        model_fn = lambda: DNABertSWrapper(device=torch.device('cuda'))
    else:
        raise ValueError("Unknown model name {}".format(model_name))

    assert out_dir.parent.exists(), f"Directory {out_dir.parent} does not exist."
    out_dir.mkdir(exist_ok=True)

    with open(sgb_subset_file, "rt") as f:
        sgb_subset = list(l.strip() for l in f)

    # Next, trim any of the necessary "_group" prefixes.
    sgb_subset = [
        s[:-len("_group")] if s.endswith("_group")
        else s
        for s in sgb_subset
    ]
    with zstd.open(sgb_marker_index_file, "rt") as f:
        sgb_marker_index = json.load(f)

    logger.info(f"Subset contains {len(sgb_subset)} SGBs.")
    logger.info(f"Handling subset index {start_idx} through {end_idx-1} (inclusive).")
    compute_embedding(
        sgb_subset=sgb_subset[start_idx:end_idx],
        sgb_marker_index=sgb_marker_index,
        fasta_path=fasta_path,
        create_embed_model=model_fn,
        h5_output_dir=out_dir,
        batch_size=batch_size,
        shard_size=shard_size
    )


if __name__ == "__main__":
    args = parse_args()

    logger.info("CUDA available devices: {}".format(
        torch.cuda.device_count()
    ))

    start_idx = args.start-1  # 0-indexed, inclusive
    end_idx = args.end  # 0-indexed, exclusive
    do_job(
        model_name=args.model_name,
        fasta_path=args.fasta_path,
        sgb_subset_file=Path(args.sgb_subset_file),
        sgb_marker_index_file=Path(args.sgb_marker_index_file),
        start_idx=start_idx,
        end_idx=end_idx,
        batch_size=args.batch_size,
        out_dir=Path(args.out_dir),
        shard_size=args.shard_size,
    )