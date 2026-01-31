import argparse
from typing import *
from pathlib import Path
import yaml
import json

import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

from gem.datasets import OrganismGeneSequenceDataset, MetaphlanTaxaDatabase, MetaphlanProfileParser
from gem.ml import *
from gem.ml.dataloader_dynamic.dataloader import GenomeEmbedding_Subclass
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


def generate_embedding_initializers(model_name: str) -> Tuple[Type[GenomeEmbedding_Subclass], Dict]:
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

        from gem.glms import EvoWrapper
        return EvoWrapper, dict(num_hyena_layers=num_hyena_layers, checkpoint_name=evo_checkpoint_name)
    elif model_name == "evo2":
        raise NotImplementedError("Evo2 is not yet implemented for this training script.")
    elif model_name == "dnabert-s":
        from gem.glms import DNABertSWrapper
        return DNABertSWrapper, dict()
    else:
        raise ValueError(f"Unsupported model name '{model_name}'")


def train_and_save_model(
        model_version: str,
        model_cfg: Dict,
        model_device: torch.device,
        model_save_dir: Path,
        loss_name: str,
        train_dloader: DataLoader,
        test_dloader: DataLoader,
        n_epochs: int,
        lr: float = 0.0001,
        print_every: int = 5,
        train_rng_seed: int = 314159,
        auto_mixed_precision: bool = False,
        checkpoint_every: int = 50,
        load_checkpoint_file: Optional[Path] = None,
):
    """
    :param model_version:
    :param model_cfg:
    :param model_device:
    :param model_save_dir:
    :param loss_name:
    :param train_dloader:
    :param test_dloader:
    :param n_epochs:
    :param lr:
    :param print_every:
    :param train_rng_seed:
    :param checkpoint_every:
    :param load_checkpoint_file:
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

    """ Create model. """
    ## ======== Model & Optimizer instantiation. ========
    print("Training using cuda device: {}".format(model_device))
    if model_version == "V1":
        torch_embedding_model = SGBAbundancePredictionModel(**model_cfg).to(model_device)
    elif model_version == "V2":
        torch_embedding_model = SGBAbundanceLayeredPredictionModel(**model_cfg).to(model_device)
    elif model_version == "EPC":
        torch_embedding_model = SGBEmbedPoolConcatPredictionModel(**model_cfg).to(model_device)
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
    model_config_path = model_save_dir / "model_config.json"

    """ invoke main training loop. """
    checkpoint_dir = model_save_dir / "model_checkpoints"
    main_training_loop(
        model=torch_embedding_model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_dloader=train_dloader, test_dloader=test_dloader,
        loss_fn=loss_fn, num_epochs=n_epochs, clip_gradient_norm_ub=clip_grad_norm_ub,
        print_progress=True, print_every=print_every,
        checkpoint_every=checkpoint_every, checkpoint_dir=checkpoint_dir,
        resume_from_checkpoint=load_checkpoint_file, loss_plot_path=loss_plot_path,
        auto_mixed_precision=auto_mixed_precision, rng_seed=train_rng_seed, timer_profile=False,
    )

    """ save model config file. """
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
    parser.add_argument("-e", "--embedding-model", dest="embedding_model_name", required=True, type=str,
                        help="Name of the embedding model to use. Currently supported: 'dnabert-s', 'evo'. "
                             "For evo, specify the number of hyena layers to use (layers 1 thru k, k <= 32) using the format 'evo:<n_layers>'."
                             "For example, 'evo:5' uses the first 5 layers to produce an embedding.")
    parser.add_argument("-train", "--train", dest="train", required=True, type=str)
    parser.add_argument("-test", "--test", dest="test", required=True, type=str)
    parser.add_argument("-c", "--model-config", dest="model_cfg_path", required=True, type=str)
    parser.add_argument("-o", "--out-dir", dest="model_save_dir", required=True, type=str)
    parser.add_argument("-loss", "--loss", dest="loss_name", required=True, type=str,
                        help="Name of loss function. Either 'kl' or 'mse'")
    parser.add_argument("-markers", "--marker-sequence-dir", dest="marker_sequence_dir", required=True, type=str,
                        help="Path to the preprocessed FASTA index files (from 1_preprocess pipeline)")

    parser.add_argument("-epochs", "--epochs", dest="n_epochs", type=int, required=True)
    parser.add_argument("-lr", "--learning-rate", dest="lr", type=float, required=True)
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, required=True)

    parser.add_argument("-p", "--print-every", dest="print_every", type=int, default=5)
    parser.add_argument("-s", "--seed", dest="seed", required=False, type=int, default=314159)
    parser.add_argument("-pf", "--prefetch-factor", dest="batch_prefetch_factor", required=False, type=int, default=2)
    parser.add_argument("-resume", "--resume-from", dest="resume_from_path", required=False, type=str, default=None)
    parser.add_argument("-checkpoint", "--checkpoint-every", dest="checkpoint_every", required=False, type=int, default=20)
    parser.add_argument("-mb", "--minibatch-embed-size", dest="embed_minibatch_size", required=False, type=int, default=32)
    parser.add_argument(
        "-amp", "--use-auto-mixed-precision", dest="use_auto_mixed_precision",
        action="store_true", default=False
    )
    parser.add_argument(
        "-cd", "--cuda-devices", dest="cuda_device_ids", type=str, required=True,
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

    """ Load dataset. """
    print("Loading dataset files.")
    train_df = pd.read_csv(args.train, sep='\t', index_col="SampleID")
    test_df = pd.read_csv(args.test, sep='\t', index_col="SampleID")
    print(f"Train: {args.train} ({len(train_df)} samples)")
    print(f"Test: {args.test} ({len(test_df)} samples)")

    train_profile_parser = MetaphlanProfileParser(train_df)
    test_profile_parser = MetaphlanProfileParser(test_df)
    seed = args.seed

    """ Load Taxa Marker Gene database. """
    print("Loading Taxa Marker Gene database.")
    marker_sequence_dir = Path(args.marker_sequence_dir)
    json_index_path = marker_sequence_dir / "sgb_marker_index.json.zst"
    marker_fasta_path = marker_sequence_dir / "markers.fna"
    db = MetaphlanTaxaDatabase(json_index_path=json_index_path, fasta_path=marker_fasta_path)

    """ Create datasets. """
    print("Creating Dataset objects.")
    train_dset = OrganismGeneSequenceDataset(db, train_profile_parser)
    test_dset = OrganismGeneSequenceDataset(db, test_profile_parser)

    """ Divide up CUDA devices. """
    cuda_devices = parse_cuda_device_ids(args.cuda_device_ids)
    assert len(cuda_devices) > 0, "Unexpected error: parsed zero CUDA devices."
    if len(cuda_devices) == 0:
        raise ValueError("No CUDA devices specified.")
    elif len(cuda_devices) == 1:
        model_cuda_device = cuda_devices[0]
        num_workers = 0
        worker_devices = []
    else:
        model_cuda_device = cuda_devices[0]
        num_workers = len(cuda_devices) - 1
        worker_devices = cuda_devices[1:]
    print("Using CUDA devices: {}".format(
        ",".join(str(dev) for dev in cuda_devices)
    ))

    """ Initialize DataLoaders. """
    print("Initializing DataLoader objects.")
    train_rng = torch.Generator()
    train_rng_seed = seed + 2
    train_rng.manual_seed(train_rng_seed)

    # Create one instance of Evo (per worker) to share amongst train/test dataloaders.
    embedding_class, embedding_kwargs = generate_embedding_initializers(args.embedding_model_name)
    print("Embedding: {}  --> {}".format(embedding_class.__name__, embedding_kwargs))
    embedding_collate_fn = MultiGPUEmbeddingCollateFn(embedding_class, embedding_kwargs, worker_devices, minibatch_size=args.embed_minibatch_size)

    data_batch_size = args.batch_size
    shuffle_dataset = True
    if len(worker_devices) > 0:
        batch_prefetch_factor = args.batch_prefetch_factor
        persistent_workers = True
    else:
        batch_prefetch_factor = None
        persistent_workers = False
    train_dloader = DataLoader(
        dataset=train_dset, collate_fn=embedding_collate_fn, multiprocessing_context='spawn',
        batch_size=data_batch_size, shuffle=shuffle_dataset, generator=train_rng, drop_last=False,
        num_workers=num_workers, prefetch_factor=batch_prefetch_factor, persistent_workers=persistent_workers,
    )
    test_dloader = DataLoader(
        dataset=test_dset, collate_fn=embedding_collate_fn, multiprocessing_context='spawn',
        batch_size=data_batch_size, shuffle=False, generator=None, drop_last=False,
        num_workers=num_workers, prefetch_factor=batch_prefetch_factor, persistent_workers=persistent_workers,
    )

    """ Create model configuration. """
    print("Loading model once to infer the target embedding dimension...")
    # embedding_example = embedding_class(**embedding_kwargs, device=torch.device("cpu"))
    # embed_dim = embedding_example.embed_dim()
    # del embedding_example
    # DEBUG:
    print("DEBUG!")
    embed_dim = 4096
    print(f"Got embedding dimension = {embed_dim}")

    model_cfg = load_model_config(
        config_file=Path(args.model_cfg_path),
        rng_seed=seed + 1,
        marker_embed_dim=embed_dim,
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

    train_rng_seed = 314159
    train_and_save_model(
        model_version=model_version,
        model_cfg=model_cfg,
        model_device=model_cuda_device,
        model_save_dir=model_save_dir,
        loss_name=args.loss_name,
        train_dloader=train_dloader,
        test_dloader=test_dloader,
        n_epochs=args.n_epochs,
        lr=args.lr,
        print_every=args.print_every,
        train_rng_seed=train_rng_seed,
        auto_mixed_precision=args.use_auto_mixed_precision,
        checkpoint_every=args.checkpoint_every,
        load_checkpoint_file=resume_from_checkpoint_path,
    )


if __name__ == "__main__":
    main()
