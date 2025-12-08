import argparse
from typing import *
from pathlib import Path
import yaml
import json

import pandas as pd
import torch
from torch import nn, optim

from gem.mpa import MetaphlanMarkerEmbedding, MetaphlanDataset
from gem.ml import safe_kl_div_loss, main_training_loop, SGBAbundancePredictionModel

import sys
import logging
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()


def load_model_config(config_file: Path, rng_seed: int) -> Dict:
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
        model_rng = torch.Generator()
        model_rng.manual_seed(rng_seed)
        config_dict['init_rng'] = model_rng
        return config_dict


def train_and_save_model(
        model_cfg: Dict,
        model_save_dir: Path,
        marker_embedding_memmap_dir: Path,
        loss_fn: nn.Module,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        n_epochs: int,
        lr: float = 0.0001,
        print_every: int = 5,
        batch_size: int = 10,
        train_rng_seed: int = 314159,
        num_workers: int = 4,
        auto_mixed_precision: bool = False,
        # specify whether to store sample-specific SGB embeddings to disk (not RAM).
):
    """
    :param model_cfg:
    :param model_save_dir:
    :param marker_embedding_memmap_dir:
    :param loss_fn:
    :param train_df:
    :param test_df:
    :param n_epochs:
    :param lr:
    :param print_every:
    :param batch_size:
    :param train_rng_seed:
    :param num_workers:
    :param auto_mixed_precision:
    """

    """ Create datasets. """
    marker_embedding = MetaphlanMarkerEmbedding(
        marker_embedding_memmap_dir=marker_embedding_memmap_dir,
    )

    train_dset = MetaphlanDataset(train_df, marker_embedding)
    test_dset = MetaphlanDataset(test_df, marker_embedding)

    """ Create model. """
    ## ======== Model & Optimizer instantiation. ========
    torch_embedding_model = SGBAbundancePredictionModel(**model_cfg).to("cuda")
    torch_embedding_model = torch.compile(
        torch_embedding_model)  # Invoke compile() to get some optimization. Uses up-front compilation cost.
    print(
        f"Number of trainable parameters: {sum(p.numel() for p in torch_embedding_model.parameters() if p.requires_grad)}")
    optimizer = optim.Adam(torch_embedding_model.parameters(), lr=lr,
                           weight_decay=0.1)  # Note: weight_decay is L2 regularization.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=1e-6  # Very small final LR
    )

    """ output files -- preparation """
    loss_plot_path = model_save_dir / "loss_history.pdf"
    model_save_path = model_save_dir / "model_weights.pt"
    model_config_path = model_save_dir / "model_config.json"

    """ invoke main training loop. """
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
        print_progress=True,
        print_every=print_every,
        loss_plot_path=loss_plot_path,
        auto_mixed_precision=auto_mixed_precision,
        rng_seed=train_rng_seed,
    )

    """ save model to file. """
    torch.save(torch_embedding_model.state_dict(), model_save_path)
    logger.info(f"Wrote model parameters to {model_save_path}")

    with open(model_config_path, "wt") as out_f:
        json.dump(model_cfg, out_f, indent=4)
        logger.info(f"Wrote model config to {model_config_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", "--train", dest="train", required=True, type=str)
    parser.add_argument("-test", "--test", dest="test", required=True, type=str)
    parser.add_argument("-c", "--model-config", dest="model_cfg_path", required=True, type=str)
    parser.add_argument("-o", "--out-dir", dest="model_save_dir", required=True, type=str)
    parser.add_argument("-loss", "--loss", dest="loss_name", required=True, type=str,
                        help="Name of loss function. Either 'kl' or 'mse'")
    parser.add_argument("-e", "--embed-dir", dest="marker_embedding_memmap_dir", required=True, type=str,
                        help="The output of the previous step, where the memmapped tensordict is stored.")

    parser.add_argument("-epochs", "--epochs", dest="n_epochs", type=int, required=True)
    parser.add_argument("-lr", "--learning-rate", dest="lr", type=float, required=True)
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, required=True)

    parser.add_argument("-p", "--print-every", dest="print_every", type=int, default=5)
    parser.add_argument("-w", "--workers", dest="num_workers", type=int, default=1)
    parser.add_argument("-s", "--seed", dest="seed", required=False, type=int, default=314159)
    parser.add_argument("-amp", "--use-auto-mixed-precision", dest="use_auto_mixed_precision",
                        action="store_true", default=False, type=bool)
    return parser.parse_args()


def main():
    args = parse_args()
    train_df = pd.read_csv(args.train, sep='\t', index_col="SampleID")
    test_df = pd.read_csv(args.test, sep='\t', index_col="SampleID")

    seed = args.seed
    model_cfg = load_model_config(args.model_cfg_path, rng_seed=seed + 1)
    model_save_dir = Path(args.model_save_dir)
    model_save_dir.mkdir(exist_ok=True, parents=True)
    marker_embedding_memmap_dir = Path(args.marker_embedding_memmap_dir)

    """ loss function """
    if args.loss_name == 'kl':
        loss_fn = safe_kl_div_loss
    elif args.loss_name == 'mse':
        loss_fn = nn.MSELoss(reduction='mean')
    else:
        raise ValueError(f"Unsupported loss name '{args.loss_name}'")

    train_and_save_model(
        model_cfg=model_cfg,
        model_save_dir=model_save_dir,
        marker_embedding_memmap_dir=marker_embedding_memmap_dir,
        loss_fn=loss_fn,
        train_df=train_df,
        test_df=test_df,
        n_epochs=args.n_epochs,
        lr=args.lr,
        print_every=args.print_every,
        batch_size=args.batch_size,
        train_rng_seed=seed + 2,
        num_workers=args.num_workers,
        auto_mixed_precision=args.use_auto_mixed_precision,
    )

if __name__ == "__main__":
    main()
