import argparse
from typing import *
from pathlib import Path
import yaml
import json

import pandas as pd
import torch
from torch import optim

from gem.datasets import OrganismGeneSequenceDataset, MetaphlanTaxaDatabase, MetaphlanProfileParser
from gem.ml import *
from gem.ml.models import *

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()


def load_model_config(config_file: Path, marker_embed_dim: int, rng_seed: int) -> Dict:
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
        model_rng = torch.Generator()
        model_rng.manual_seed(rng_seed)
        config_dict['init_rng'] = model_rng
        config_dict['marker_embed_dim'] = marker_embed_dim
        print("Initializing model {} with marker embedding dimension = {}".format(
            config_file.resolve(), marker_embed_dim
        ))
        return config_dict


def train_and_save_model(
        model_version: str,
        model_cfg: Dict,
        model_save_dir: Path,
        loss_name: str,
        train_dset: OrganismGeneSequenceDataset,
        test_dset: OrganismGeneSequenceDataset,
        n_epochs: int,
        shuffle_dataset: bool,
        cuda_devices: List[torch.device],
        lr: float = 0.0001,
        print_every: int = 5,
        batch_size: int = 10,
        batch_prefetch_factor: int = 2,
        train_rng_seed: int = 314159,
        auto_mixed_precision: bool = False,
        checkpoint_every: int = 50,
        load_checkpoint_file: Optional[Path] = None,
        timer_profile: bool = False,
        # specify whether to store sample-specific SGB embeddings to disk (not RAM).
):
    """
    :param model_version:
    :param model_cfg:
    :param model_save_dir:
    :param loss_name:
    :param train_dset:
    :param test_dset:
    :param n_epochs:
    :param lr:
    :param print_every:
    :param batch_size:
    :param batch_prefetch_factor:
    :param train_rng_seed:
    :param cuda_devices:
    :param auto_mixed_precision:
    """

    """ loss function """
    if loss_name == 'kl':
        print("Using KL loss")
        loss_fn = safe_kl_div_loss
        clip_grad_norm_ub = None
    elif loss_name == 'mse_log':
        print("Using Mean-Squared (Log-probability) loss")
        loss_fn = safe_mse_log_loss
        clip_grad_norm_ub = 1.0  # apply gradient clipping for MSE-log, as this tends to have exploding gradient issues.
    elif loss_name == 'mse':
        print("Using Mean-Squared (Linear/non-log probability) loss")
        loss_fn = safe_mse_loss
        clip_grad_norm_ub = None
    elif loss_name == 'cross_entropy':
        print("Using Cross-entropy loss")
        loss_fn = safe_cross_entropy_loss
        clip_grad_norm_ub = None
    else:
        raise ValueError(f"Unsupported loss name '{loss_name}'")

    """ Divide up CUDA devices. """
    if len(cuda_devices) == 0:
        raise ValueError("No CUDA devices specified.")
    elif len(cuda_devices) == 1:
        main_cuda_device = cuda_devices[0]
        num_workers = 0
        worker_devices = []
    else:
        main_cuda_device = cuda_devices[0]
        num_workers = len(cuda_devices) - 1
        worker_devices = cuda_devices[1:]

    """ Create model. """
    ## ======== Model & Optimizer instantiation. ========
    print("Training using cuda device: {}".format(main_cuda_device))
    if model_version == "V1":
        torch_embedding_model = SGBAbundancePredictionModel(**model_cfg).to(main_cuda_device)
    elif model_version == "V2":
        torch_embedding_model = SGBAbundanceLayeredPredictionModel(**model_cfg).to(main_cuda_device)
    elif model_version == "EPC":
        torch_embedding_model = SGBEmbedPoolConcatPredictionModel(**model_cfg).to(main_cuda_device)
    else:
        raise ValueError(f"Unsupported model_version `{model_version}`")

    print("Using model class: {}".format(
        torch_embedding_model.__class__.__name__
    ))
    model_class_name = torch_embedding_model.__class__.__name__  # save name before compilation.

    torch_embedding_model = torch.compile(
        torch_embedding_model
    )  # Invoke compile() to get some optimization. Uses up-front compilation cost.
    print(
        f"Number of trainable parameters: {sum(p.numel() for p in torch_embedding_model.parameters() if p.requires_grad)}"
    )
    optimizer = optim.Adam(torch_embedding_model.parameters(), lr=lr,
                           weight_decay=0.1)  # Note: weight_decay is L2 regularization.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=1e-6  # Very small final LR
    )
    torch.set_float32_matmul_precision('high')

    """ output files -- preparation """
    loss_plot_path = model_save_dir / "loss_history.pdf"
    model_save_path = model_save_dir / "model_weights.pt"
    model_config_path = model_save_dir / "model_config.json"

    """ invoke main training loop. """
    checkpoint_dir = model_save_dir / "model_checkpoints"
    main_training_loop(
        model=torch_embedding_model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_dset=train_dset,
        test_dset=test_dset,
        loss_fn=loss_fn,
        num_workers=num_workers,
        batch_size=batch_size,
        num_epochs=n_epochs,
        shuffle_dataset=shuffle_dataset,
        clip_gradient_norm_ub=clip_grad_norm_ub,
        print_progress=True,
        print_every=print_every,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=checkpoint_dir,
        resume_from_checkpoint=load_checkpoint_file,
        loss_plot_path=loss_plot_path,
        auto_mixed_precision=auto_mixed_precision,
        rng_seed=train_rng_seed,
        cuda_device=main_cuda_device,
        prefetch_factor=batch_prefetch_factor,
        timer_profile=timer_profile,
    )

    """ save model to file. """
    # torch.save(torch_embedding_model.state_dict(), model_save_path)
    # logger.info(f"Wrote model parameters to {model_save_path}")

    with open(model_config_path, "wt") as out_f:
        rng = model_cfg['init_rng']
        model_init_seed = rng.initial_seed()
        model_cfg['init_rng_seed'] = model_init_seed
        del model_cfg['init_rng']

        # also include model class name.
        model_cfg['class'] = model_class_name

        json.dump(model_cfg, out_f, indent=4)
        logger.info(f"Wrote model config to {model_config_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--model-version", dest="model_version", required=True, type=str)
    parser.add_argument("-train", "--train", dest="train", required=True, type=str)
    parser.add_argument("-test", "--test", dest="test", required=True, type=str)
    parser.add_argument("-c", "--model-config", dest="model_cfg_path", required=True, type=str)
    parser.add_argument("-o", "--out-dir", dest="model_save_dir", required=True, type=str)
    parser.add_argument("-loss", "--loss", dest="loss_name", required=True, type=str,
                        help="Name of loss function. Either 'kl' or 'mse'")
    parser.add_argument() # todo specify json and fasta files

    parser.add_argument("-epochs", "--epochs", dest="n_epochs", type=int, required=True)
    parser.add_argument("-lr", "--learning-rate", dest="lr", type=float, required=True)
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, required=True)

    parser.add_argument("-p", "--print-every", dest="print_every", type=int, default=5)
    parser.add_argument("-s", "--seed", dest="seed", required=False, type=int, default=314159)
    parser.add_argument("-pf", "--prefetch-factor", dest="batch_prefetch_factor", required=False, type=int, default=2)
    parser.add_argument("-resume", "--resume-from", dest="resume_from_path", required=False, type=str, default=None)
    parser.add_argument("-checkpoint", "--checkpoint-every", dest="checkpoint_every", required=False, type=int, default=20)
    parser.add_argument(
        "-amp", "--use-auto-mixed-precision", dest="use_auto_mixed_precision",
        action="store_true", default=False
    )
    parser.add_argument(
        "-cd", "--cuda-devices", dest="cuda_device_names", type=str, required=True,
        help="A comma-separated list of CUDA devices to use during training. "
             "If more than one is passed, the first CUDA device will be used for gradients and backprop, and the "
             "rest will be used to compute the embeddings. "
             "Example: -cd 0,1,2,3 uses cuda:0 for model optimization, and 1,2,3 will be used for embeddings."
    )
    return parser.parse_args()


def parse_cuda_device_ids(cuda_device_ids: str) -> List[torch.device]:
    cuda_device_ids = [int(x) for x in cuda_device_ids.split(",") if len(x) > 0]
    if len(cuda_device_ids) == 0:
        print(f"At least one CUDA device ID must be specified. Got: {cuda_device_ids}")
        exit(1)

    cuda_devices = []
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Total CUDA devices available: {device_count}")

        for device_id in cuda_device_ids:
            if device_id < device_count:
                print(f"CUDA device :{device_id} exists and is available.")
                # You can now create a device object for it
                device = torch.device(f"cuda:{device_id}")
                cuda_devices.append(device)
            else:
                print(f"CUDA device :{device_id} does not exist. Only devices 0 to {device_count - 1} are available.")
                exit(1)
    else:
        print("CUDA is not available on this system.")

    assert len(cuda_devices) > 0, "Unexpected error: parsed zero CUDA devices."
    return cuda_devices


def main():
    args = parse_args()
    train_df = pd.read_csv(args.train, sep='\t', index_col="SampleID")
    test_df = pd.read_csv(args.test, sep='\t', index_col="SampleID")
    train_profile_parser = MetaphlanProfileParser(train_df)
    test_profile_parser = MetaphlanProfileParser(test_df)
    db = MetaphlanTaxaDatabase(json_index_path=, fasta_path=)

    """ Create datasets. """
    print(f"Train: {args.train} ({len(train_df)} samples)")
    print(f"Test: {args.test} ({len(test_df)} samples)")
    train_dset = OrganismGeneSequenceDataset(db, profile_parser)
    test_dset = OrganismGeneSequenceDataset(db, profile_parser)

    memmap_tensor_sample_dir = Path(args.memmap_tensor_sample_dir)
    print(f"Loading memmapped sample tensors from {memmap_tensor_sample_dir}")
    train_dset.load_memmap_tensors(memmap_tensor_sample_dir)
    test_dset.load_memmap_tensors(memmap_tensor_sample_dir)

    """ Create model configuration. """
    seed = args.seed
    model_cfg = load_model_config(
        config_file=Path(args.model_cfg_path),
        rng_seed=seed + 1,
        marker_embed_dim=train_dset.embed_feature_dim(),
    )
    model_save_dir = Path(args.model_save_dir)
    model_save_dir.mkdir(exist_ok=True, parents=True)
    print(f"Target output directory: {model_save_dir}")

    model_version = args.model_version
    model_supported_versions = {"V1", "V2", "EPC"}
    if model_version not in model_supported_versions:
        raise ValueError(
            f"Unsupported model version string {model_version}. "
            f"Must be one of: {list(model_supported_versions)}"
        )

    if args.resume_from_path is not None:
        resume_from_checkpoint_path = Path(args.resume_from_path)
    else:
        resume_from_checkpoint_path = None

    cuda_devices = parse_cuda_device_ids(args.cuda_device_ids)
    assert len(cuda_devices) > 0, "Unexpected error: parsed zero CUDA devices."
    train_and_save_model(
        model_version=model_version,
        model_cfg=model_cfg,
        model_save_dir=model_save_dir,
        load_checkpoint_file=resume_from_checkpoint_path,
        checkpoint_every=args.checkpoint_every,
        loss_name=args.loss_name,
        train_dset=train_dset,
        test_dset=test_dset,
        n_epochs=args.n_epochs,
        shuffle_dataset=True,
        lr=args.lr,
        print_every=args.print_every,
        batch_size=args.batch_size,
        batch_prefetch_factor=args.batch_prefetch_factor,
        train_rng_seed=seed + 2,
        cuda_devices=cuda_devices,
        auto_mixed_precision=args.use_auto_mixed_precision,
        timer_profile=False,
    )


if __name__ == "__main__":
    main()
