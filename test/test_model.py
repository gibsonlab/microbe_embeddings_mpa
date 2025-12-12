from pathlib import Path
from typing import *

import torch
from torch import nn
from gem.ml import MetaphlanDataLoader, SGBAbundancePredictionModel
from gem.mpa import MetaphlanDatasetMemmapped


def create_test_config() -> Dict:
    model_rng = torch.Generator()
    model_rng.manual_seed(12345)
    config_dict = {
        'sgb_embed_dim': 32,
        'sgb_embed_latent_dim': 16,
        'n_layers': 2,
        'model_dim': 32,
        'num_heads': 8,
        'key_query_dim': 16,
        'latent_collection_dim': 16,
        'combination_latent_dim': 16,
        'init_rng': model_rng,
        'marker_embed_dim': 768,
    }
    return config_dict


def test_model_memmap_input(memmap_tensor_sample_dir: Path):
    """
    Test the model with pre-computed memmap tensor input (3_train/3_memmap_*.sh)
    """
    test_sample_ids = ["SAMEA7041133", "SAMEA7041172"]
    test_dset = MetaphlanDatasetMemmapped(test_sample_ids)
    test_dset.load_memmap_tensors(memmap_tensor_sample_dir)

    model_cfg = create_test_config()
    print("Creating test model from configuration: {}".format(model_cfg))
    test_model = SGBAbundancePredictionModel(**model_cfg).to("cuda")
    test_model = torch.compile(test_model)

    batch_sz = 2
    test_dloader = MetaphlanDataLoader(
        dataset=test_dset,
        batch_size=batch_sz,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        worker_rng_seed=314159
    )

    test_sample_ids, test_batch_features, test_marker_mask, test_sgb_mask, test_y = next(iter(test_dloader))
    test_y = test_model(test_batch_features.to("cuda"), test_marker_mask, test_sgb_mask)
    print(test_y)


if __name__ == "__main__":
    memmap_tensor_sample_dir = Path("/data/bwh-comppath-seq/youn/metaphlan_dset/model_training/memmmap_samples")
    test_model_memmap_input(memmap_tensor_sample_dir)
