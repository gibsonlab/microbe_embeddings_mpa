import argparse
from typing import *
from pathlib import Path
import yaml
import json

import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

from gem.ml import *
from gem.ml.models import *
from common.ml import create_dataloader

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
        train_dloader: DataLoader,
        test_dloader: DataLoader,
        n_epochs: int,
        lr: float = 0.0001,
        print_every: int = 5,
        train_rng_seed: int = 314159,
        cuda_device_name: str = "cuda",
        checkpoint_every: int = 50,
        load_checkpoint_file: Optional[Path] = None,
        timer_profile: bool = False,
):
    """
    :param model_version:
    :param model_cfg:
    :param model_save_dir:
    :param loss_name:
    :param train_dloader:
    :param test_dloader:
    :param n_epochs:
    :param lr:
    :param print_every:
    :param train_rng_seed:
    :param cuda_device_name:
    :param checkpoint_every:
    :param load_checkpoint_file:
    :param timer_profile:
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

    """ Create model. """
    ## ======== Model & Optimizer instantiation. ========
    print("Using target cuda device: {}".format(cuda_device_name))
    if model_version == "V1":
        torch_embedding_model = SGBAbundancePredictionModel(**model_cfg).to(cuda_device_name)
    elif model_version == "V2":
        torch_embedding_model = SGBAbundanceLayeredPredictionModel(**model_cfg).to(cuda_device_name)
    elif model_version == "EPC":
        torch_embedding_model = SGBEmbedPoolConcatPredictionModel(**model_cfg).to(cuda_device_name)
    elif model_version == "SetTransformer":
        torch_embedding_model = HierarchicalSetTransformer(**model_cfg).to(cuda_device_name)
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
                           weight_decay=1e-3)  # Note: weight_decay is L2 regularization.

    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    warmup_epochs = 5

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,  # Start at 1% of lr
        total_iters=warmup_epochs
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=n_epochs - warmup_epochs,
        eta_min=1e-6
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    torch.set_float32_matmul_precision('high')

    """ output files -- preparation """
    loss_plot_path = model_save_dir / "loss_history.pdf"
    model_config_path = model_save_dir / "model_config.json"

    """ invoke main training loop. """
    checkpoint_dir = model_save_dir / "model_checkpoints"
    main_training_loop(
        model=torch_embedding_model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_dloader=train_dloader,
        test_dloader=test_dloader,
        loss_fn=loss_fn,
        num_epochs=n_epochs,
        clip_gradient_norm_ub=clip_grad_norm_ub,
        print_progress=True,
        print_every=print_every,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=checkpoint_dir,
        resume_from_checkpoint=load_checkpoint_file,
        loss_plot_path=loss_plot_path,
        rng_seed=train_rng_seed,
        timer_profile=timer_profile,
    )

    """ save model to file. """
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
    parser.add_argument("--abundance-tables", dest="abundance_table_dir", required=True, type=str)
    parser.add_argument("--embedding-h5", dest="embedding_h5_path", required=True, type=str)

    parser.add_argument("-epochs", "--epochs", dest="n_epochs", type=int, required=True)
    parser.add_argument("-lr", "--learning-rate", dest="lr", type=float, required=True)
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, required=True)
    parser.add_argument("-p", "--print-every", dest="print_every", type=int, default=5)
    parser.add_argument("-w", "--workers", dest="num_workers", type=int, default=1)
    parser.add_argument("-s", "--seed", dest="seed", required=False, type=int, default=314159)
    parser.add_argument("-pf", "--prefetch-factor", dest="batch_prefetch_factor", required=False, type=int, default=2)
    parser.add_argument("-resume", "--resume-from", dest="resume_from_path", required=False, type=str, default=None)
    parser.add_argument("-checkpoint", "--checkpoint-every", dest="checkpoint_every", required=False, type=int, default=20)
    parser.add_argument(
        "-cd", "--cuda-device", dest="cuda_device_name", type=str, default="cuda",
        help="Specify which CUDA device name to use. (Example: cuda, cuda:0, cuda:1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    seed = args.seed
    abundance_table_dir = Path(args.abundance_table_dir)
    embedding_h5_path = Path(args.embedding_h5_path)
    train_df = pd.read_csv(args.train, sep='\t')
    test_df = pd.read_csv(args.test, sep='\t')

    """ Create datasets. """
    train_rng = torch.Generator()
    train_rng.manual_seed(seed + 2)
    print(f"Train: {args.train} ({len(train_df)} samples)")
    print(f"Test: {args.test} ({len(test_df)} samples)")
    train_dset, train_dloader = create_dataloader(
        train_df, abundance_table_dir, embedding_h5_path,
        batch_size=args.batch_size, num_workers=args.num_workers,
        rng=train_rng, drop_last=False, shuffle=True,
        prefetch_factor=args.batch_prefetch_factor, dtype=torch.float32
    )
    test_dset, test_dloader = create_dataloader(
        test_df, abundance_table_dir, embedding_h5_path,
        batch_size=args.batch_size, num_workers=args.num_workers,
        rng=None, drop_last=False, shuffle=False,
        prefetch_factor=args.batch_prefetch_factor, dtype=torch.float32
    )

    """ Create model configuration. """
    model_cfg = load_model_config(
        config_file=Path(args.model_cfg_path),
        rng_seed=seed + 1,
        marker_embed_dim=train_dset.embed_feature_dim(),
    )
    model_save_dir = Path(args.model_save_dir)
    model_save_dir.mkdir(exist_ok=True, parents=True)
    print(f"Target output directory: {model_save_dir}")

    model_version = args.model_version

    if args.resume_from_path is not None:
        resume_from_checkpoint_path = Path(args.resume_from_path)
    else:
        resume_from_checkpoint_path = None

    train_and_save_model(
        model_version=model_version,
        model_cfg=model_cfg,
        model_save_dir=model_save_dir,
        load_checkpoint_file=resume_from_checkpoint_path,
        checkpoint_every=args.checkpoint_every,
        loss_name=args.loss_name,
        train_dloader=train_dloader,
        test_dloader=test_dloader,
        n_epochs=args.n_epochs,
        lr=args.lr,
        print_every=args.print_every,
        train_rng_seed=seed + 2,
        cuda_device_name=args.cuda_device_name,
        timer_profile=False,
    )


if __name__ == "__main__":
    main()
