import argparse
from typing import *
from pathlib import Path
import yaml
import json

import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

from gem.datasets.mpa import TorchStackedMetaphlanPreembeddedDataset
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
        model_type: str,
        model_cfg: Dict,
        model_save_dir: Path,
        loss_name: str,
        train_dloader: DataLoader,
        test_dloader: DataLoader,
        n_epochs: int,
        train_in_bfloat16: bool,
        lr: float = 0.0001,
        print_every: int = 5,
        train_rng_seed: int = 314159,
        cuda_device_name: str = "cuda",
        checkpoint_every: int = 50,
        load_checkpoint_file: Optional[Path] = None,
        timer_profile: bool = False,
):
    """
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
    if model_type == "EPC":
        torch_embedding_model = SGBEmbedPoolConcatPredictionModel(**model_cfg).to(device=cuda_device_name)
    elif model_type == "SetTransformer":
        torch_embedding_model = HierarchicalSetTransformer(**model_cfg).to(device=cuda_device_name)

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
        train_in_bfloat16=train_in_bfloat16,
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
    parser.add_argument("-train", "--train", dest="train", required=True, type=str)
    parser.add_argument("-test", "--test", dest="test", required=True, type=str)
    parser.add_argument("-model", "--model-type", dest="model_type", required=True, type=str)
    parser.add_argument("-c", "--model-config", dest="model_cfg_path", required=True, type=str)
    parser.add_argument("-o", "--out-dir", dest="model_save_dir", required=True, type=str)
    parser.add_argument("-loss", "--loss", dest="loss_name", required=True, type=str,
                        help="Name of loss function. Either 'kl' or 'mse'")
    parser.add_argument("-e", "--embed-memmap-file", dest="embed_memmap_file", required=True, type=str)

    parser.add_argument("-epochs", "--epochs", dest="n_epochs", type=int, required=True)
    parser.add_argument("-lr", "--learning-rate", dest="lr", type=float, required=True)
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, required=True)

    parser.add_argument("-p", "--print-every", dest="print_every", type=int, default=5)
    parser.add_argument("-w", "--workers", dest="num_workers", type=int, default=1)
    parser.add_argument("-s", "--seed", dest="seed", required=False, type=int, default=314159)
    parser.add_argument("-pf", "--prefetch-factor", dest="batch_prefetch_factor", required=False, type=int, default=2)
    parser.add_argument("-resume", "--resume-from", dest="resume_from_path", required=False, type=str, default=None)
    parser.add_argument("-checkpoint", "--checkpoint-every", dest="checkpoint_every", required=False, type=int,
                        default=20)
    parser.add_argument(
        "-cd", "--cuda-device", dest="cuda_device_name", type=str, default="cuda",
        help="Specify which CUDA device name to use. (Example: cuda, cuda:0, cuda:1)",
    )
    parser.add_argument(
        "--use-bfloat16", dest="use_bfloat16", action="store_true", default=False,
        help="Use bfloat16 inputs (may help with many-gene representations)."
    )
    return parser.parse_args()


def main(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        embed_memmap_file: Path,
        seed: int,
        model_type: str,
        model_cfg_path: Path,
        model_save_dir: Path,
        batch_size: int,
        num_workers: int,
        batch_prefetch_factor: int,
        loss_name: str,
        lr: float,
        n_epochs: int,
        resume_from_checkpoint_path: Union[Path, None],
        checkpoint_every: int,
        print_every: int,
        use_bfloat16: bool,
        cuda_device_name: str = "cuda"
):
    """ Create datasets. """
    if use_bfloat16:
        # half-precision
        train_dset = TorchStackedMetaphlanPreembeddedDataset(train_df, embed_memmap_file, dtype=torch.bfloat16)
        test_dset = TorchStackedMetaphlanPreembeddedDataset(test_df, embed_memmap_file, dtype=torch.bfloat16)
    else:
        # full precision
        train_dset = TorchStackedMetaphlanPreembeddedDataset(train_df, embed_memmap_file, dtype=torch.float32)
        test_dset = TorchStackedMetaphlanPreembeddedDataset(test_df, embed_memmap_file, dtype=torch.float32)
        torch.set_float32_matmul_precision('high')

    """ Create dataloaders. """
    train_rng = torch.Generator()
    train_rng.manual_seed(seed + 2)
    train_dloader = train_dset.create_dataloader(
        batch_size=batch_size, shuffle=True, generator=train_rng,
        pin_memory=True,
        drop_last=False, num_workers=num_workers, prefetch_factor=batch_prefetch_factor,
    )
    test_dloader = test_dset.create_dataloader(
        batch_size=batch_size, shuffle=False, generator=None,
        pin_memory=True,
        drop_last=False, num_workers=num_workers, prefetch_factor=batch_prefetch_factor,
    )

    """ Create model configuration. """
    model_cfg = load_model_config(
        config_file=model_cfg_path,
        rng_seed=seed + 1,
        marker_embed_dim=train_dset.embed_feature_dim(),
    )
    print(f"Target output directory: {model_save_dir}")

    train_and_save_model(
        model_type=model_type,
        model_cfg=model_cfg,
        model_save_dir=model_save_dir,
        load_checkpoint_file=resume_from_checkpoint_path,
        checkpoint_every=checkpoint_every,
        loss_name=loss_name,
        train_dloader=train_dloader,
        test_dloader=test_dloader,
        n_epochs=n_epochs,
        lr=lr,
        train_in_bfloat16=use_bfloat16,
        print_every=print_every,
        train_rng_seed=seed + 2,
        cuda_device_name=cuda_device_name,
        timer_profile=False,
    )


if __name__ == "__main__":
    _args = parse_args()
    _train_df = pd.read_csv(_args.train, sep='\t', index_col="SampleID")
    _test_df = pd.read_csv(_args.test, sep='\t', index_col="SampleID")
    print(f"Train: {_args.train} ({_train_df.shape[0]} samples)")
    print(f"Test: {_args.test} ({_test_df.shape[0]} samples)")

    _model_save_dir = Path(_args.model_save_dir)
    _model_save_dir.mkdir(exist_ok=True, parents=True)

    if _args.resume_from_path is not None:
        resume_from_path = Path(_args.resume_from_path)
    else:
        resume_from_path = None

    main(
        train_df=_train_df, test_df=_test_df,
        embed_memmap_file=Path(_args.embed_memmap_file),
        seed=_args.seed,
        model_type=_args.model_type,
        model_cfg_path=Path(_args.model_cfg_path),
        model_save_dir=_model_save_dir,
        batch_size=_args.batch_size,
        num_workers=_args.num_workers,
        batch_prefetch_factor=_args.batch_prefetch_factor,
        loss_name=_args.loss_name,
        lr=_args.lr,
        n_epochs=_args.n_epochs,
        resume_from_checkpoint_path=resume_from_path,
        checkpoint_every=_args.checkpoint_every,
        print_every=_args.print_every,
        cuda_device_name=_args.cuda_device_name,
        use_bfloat16=_args.use_bfloat16,
    )
