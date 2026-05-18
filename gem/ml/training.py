from typing import *
from pathlib import Path

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from gem.datasets import AbstractMetaphlanPreembeddedDataset, MetaphlanHDF5PreembeddedDataset, OrganismGeneSequenceDataset
from gem.util import timer


def create_metaphlan_preembedded_hdf5_dloader(
        dataset: MetaphlanHDF5PreembeddedDataset,
        batch_size: int,
        shuffle: bool,
        rng: Optional[torch.Generator],
        drop_last: bool,
        num_workers: Optional[int] = 0,
        prefetch_factor: int = 2,
):
    from gem.datasets.mpa import HDF5BatchShuffledSampler
    from gem.ml.dataloader_preembedded.data_loader import MetaphlanDataLoader
    sampler = HDF5BatchShuffledSampler(
        data_source=dataset, batch_size=batch_size, shuffle=shuffle, rng_seed=rng.initial_seed() + 1,
    )
    dloader = MetaphlanDataLoader(
        dataset=dataset,
        batch_size=batch_size, num_workers=num_workers,
        generator=rng, drop_last=drop_last, prefetch_factor=prefetch_factor,
        persistent_workers=True, pin_memory=True,
        shuffle=shuffle, sampler=sampler,
    )


def create_metaphlan_preembedded_generic_dloader(
        dataset: AbstractMetaphlanPreembeddedDataset,
        batch_size: int,
        shuffle: bool,
        rng: Optional[torch.Generator],
        drop_last: bool,
        num_workers: Optional[int] = 0,
        prefetch_factor: int = 2,
):
    from gem.ml.dataloader_preembedded.data_loader import MetaphlanDataLoader
    return MetaphlanDataLoader(
        dataset=dataset,
        batch_size=batch_size, num_workers=num_workers,
        generator=rng, drop_last=drop_last, prefetch_factor=prefetch_factor,
        persistent_workers=True, pin_memory=True,
        shuffle=shuffle
    )


from .dataloader_dynamic.dataloader import GenomeEmbedding_Subclass
def create_dynamic_embedding_dloader(
        dataset: OrganismGeneSequenceDataset,
        embedding_class: Type[GenomeEmbedding_Subclass],
        embedding_kwargs: Dict[str, Any],
        batch_size: int,
        shuffle: bool,
        rng: Optional[torch.Generator],
        drop_last: bool,
        worker_devices: List[torch.device],
        num_workers: Optional[int] = 0,
        prefetch_factor: int = 2,
):
    from gem.ml.dataloader_dynamic import create_dataloader_dynamic_embedding
    create_dataloader_dynamic_embedding(
        dataset,
        embedding_class=embedding_class,
        embedding_kwargs=embedding_kwargs,
        worker_devices=worker_devices, num_workers=num_workers,
        data_batch_size=batch_size,
        shuffle=shuffle, generator=rng, drop_last=drop_last, prefetch_factor=prefetch_factor,
    )


def get_model_device(model: nn.Module) -> torch.device:
    """
    Checks the device of a PyTorch model.
    """
    params = model.parameters()
    first_param = next(params)
    return first_param.device


def main_training_loop(
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        train_dloader: DataLoader,
        test_dloader: DataLoader,
        loss_fn: Union[nn.Module, Callable],
        checkpoint_dir: Path,
        clip_gradient_norm_ub: Optional[float] = None,
        num_epochs: int = 5000,
        print_progress: bool = True,
        print_every: int = 5,
        resume_from_checkpoint: Optional[Path] = None,
        checkpoint_every: int = 25,
        loss_plot_path: Optional[Path] = None,
        rng_seed: int = 314159,
        train_in_bfloat16: bool = False,
        timer_profile: bool = False,
):
    """
    Train an input model on the given dataset. Uses torch.optim.Adam by default.
    This function's training loop mutates the parameters of the `model` object.

    :param model: The model to train. Should be designed to take a tensor of shape (*, MAX_NUM_SGBS, MAX_NUM_MARKERS, MARKER_EMBED_DIM) as input, and output (*, MAX_NUM_SGBS) which represents a batched vector of logits.
    :param optimizer: The torch.optim.Optimizer instance used to optimize.
    :param lr_scheduler: the torch.optim.lr_scheduler instance used to tune learning rates.
    :param train_dloader: The DataLoader that batches the training dataset. Must have a torch.Generator specified during creation.
    :param test_dloader: The DataLoader that batches the test dataset.
    :param loss_fn: The loss function to use for optimization.
    :param checkpoint_dir: The directory to save the model checkpoints.
    :param clip_gradient_norm_ub: If specified and greater than zero, applies gradient clipping.
    :param num_epochs: The total number of epochs to train. (1 epoch is a full pass on the entire training dataset, after batching.)
    :param print_progress: Indicate whether to print loss values periodically to stdout. Specify `print_every` to change how often this occurs.
    :param print_every: Indicate how often to print progress as a debug message to stdout. Only relevant if `print_progress` is set to true.
    :param loss_plot_path: If provided, plots the training loss/test loss history. Test loss is only plotted if test_df is provided.
    :param rng_seed: The random generator seed to use for training. Specify for reproducibility. (default: 314159)
    """

    """ Initialization. """
    model_device = get_model_device(model)
    # scaler = GradScaler(str(model_device), enabled=auto_mixed_precision)

    print(f"Checkpoints saved to {checkpoint_dir} --> every {checkpoint_every} epochs")
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    """ Initialize dataset objects. """
    training_data_rng: torch.Generator = train_dloader.generator
    assert training_data_rng is not None, "Training dataset DataLoader must have a pre-seeded Generator instance provided."
    assert isinstance(training_data_rng, torch.Generator), "Training dataset DataLoader must have a torch.Generator instance for generator. Got: {}".format(training_data_rng.__class__.__name__)
    training_data_rng.manual_seed(rng_seed)

    n_training_examples = len(train_dloader.dataset)
    n_test_examples = len(test_dloader.dataset)

    print(f"Training dataset size: {n_training_examples}")
    print(f"Number of training batches: {len(train_dloader)}")
    print(f"Test dataset size: {n_test_examples}")
    print(f"Number of test batches: {len(test_dloader)}")

    # also set global RNG seed for dropout reproducibility (maybe figure out how dropout RNG is controlled during training.)
    torch.manual_seed(rng_seed + 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng_seed + 1)

    # specify whether to train in bfloat16
    amp_enabled = train_in_bfloat16

    """ Training loop -- Optimize using batches. """
    print("NOTE: the first iteration of model evaluation may take considerably longer, due to compilation overhead.")
    def _compute_test_loss(show_pbar: bool = False) -> float:
        total_test_loss = 0.0
        model.eval()
        with torch.no_grad():
            if show_pbar:
                collection = tqdm(test_dloader, total=len(test_dloader), desc="Test Loss Eval", unit="batch")
            else:
                collection = test_dloader
            for batch_idx, (test_sample_ids, test_batch_features, test_marker_mask, test_taxa_mask, test_y) in enumerate(collection):
                with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp_enabled):
                    test_y_hat = model(
                        test_batch_features.to(model_device, non_blocking=True),
                        test_marker_mask.to(model_device, non_blocking=True),
                        test_taxa_mask.to(model_device, non_blocking=True)
                    )

                # assert test_y_hat.shape == test_y.shape, f"Neural Network output and ground truth have different shapes: {test_y_hat.shape} (NN) vs {test_y.shape} (truth)"
                batch_loss = loss_fn(
                    nn.functional.log_softmax(test_y_hat, dim=-1),  # log pred probabilities
                    torch.log(test_y.to(model_device, dtype=test_y_hat.dtype, non_blocking=True))  # log target probabilities
                )
                if torch.isnan(batch_loss).item():
                    # ========== Found NaN batch loss. Try to report current status and terminate training loop.
                    for i in range(0, len(test_sample_ids)):
                        print(f"Batch {batch_idx}, sample {i}")
                        sample_id = test_sample_ids[i]
                        feat_i = test_batch_features[i]
                        taxa_mask_i = test_taxa_mask[i]
                        marker_mask_i = test_marker_mask[i]
                        y_hat_i = nn.functional.log_softmax(test_y_hat[i], dim=-1)
                        yi = torch.log(test_y[i].to(model_device, dtype=test_y_hat.dtype, non_blocking=True),)
                        loss_i = loss_fn(
                            torch.unsqueeze(y_hat_i, dim=0),
                            torch.unsqueeze(yi, dim=0)
                        )
                        if torch.any(torch.isnan(loss_i)):
                            print("Found NaN loss (i = {}, sample = {})".format(i, sample_id))
                            print("feat:", feat_i)
                            print("taxa mask:", taxa_mask_i)
                            print("marker mask:", marker_mask_i)
                            print("y_hat_i (before log_softmax):", test_y_hat[i])
                            print("y_hat_i:", y_hat_i)
                            print("yi:", yi)
                            print("Features -- any nan?", torch.any(torch.isnan(feat_i)))

                            # Save input tensor to file.
                            from datetime import datetime
                            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                            torch.save({
                                'sample': sample_id,
                                'feat': feat_i,
                                'taxa_mask': taxa_mask_i,
                                'marker_mask': marker_mask_i,
                                'target': yi,
                            }, checkpoint_dir / f"crash_input_dump_{current_time}.pt")

                            raise Exception("NaN error!")

                # divide by total dataset size, to contribute to the overall average estimate.
                total_test_loss += batch_loss.item() * test_y.shape[0] / n_test_examples
        return total_test_loss

    """ Try to resume from checkpoint file, if specified. """
    if resume_from_checkpoint is not None:
        print(f"Resuming from: {resume_from_checkpoint}")
        last_epoch, epoch_history, training_loss_history, test_loss_history = load_checkpoint(
            resume_from_checkpoint,
            model, optimizer, lr_scheduler, training_data_rng
        )
        print(f"Last completed epoch = {last_epoch}")
        start_epoch = last_epoch + 1
        current_lr = lr_scheduler.get_last_lr()[0]
    else:
        print(f"Initial Test Loss: {_compute_test_loss(show_pbar=True)}")
        epoch_history = []
        training_loss_history = []
        test_loss_history = []
        start_epoch = 1
        current_lr = torch.nan

    pbar = tqdm(
        range(start_epoch, num_epochs + 1),
        desc="Training", unit="epoch",
        initial=start_epoch-1,
        total=num_epochs
    )
    epoch = None  # make sure variable exists at least once, for safety
    for epoch in pbar:
        epoch_training_loss = 0.0
        model.train()
        for batch_idx, (_, training_batch_features, training_marker_mask, training_taxa_mask, training_y) in enumerate(train_dloader):
            optimizer.zero_grad()

            with timer("Model-With-Grad ({}/{})".format(batch_idx+1, len(train_dloader)), enabled=timer_profile):
                with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp_enabled):
                    training_y_hat = model(
                        training_batch_features.to(model_device, non_blocking=True),
                        training_marker_mask.to(model_device, non_blocking=True),
                        training_taxa_mask.to(model_device, non_blocking=True),
                    )

            # assert training_y_hat.shape == training_y.shape, f"Neural Network output and ground truth have different shapes: {training_y_hat.shape} (NN) vs {training_y.shape} (truth)"
            with timer("Loss-With-Grad ({}/{})".format(batch_idx+1, len(train_dloader)), enabled=timer_profile):
                training_loss = loss_fn(
                    nn.functional.log_softmax(training_y_hat, dim=-1),  # log pred probabilities
                    torch.log(training_y.to(model_device, dtype=training_y_hat.dtype, non_blocking=True))  # log target probabilities
                )

            with timer("Backward-Update", enabled=timer_profile):
                # scaler.scale(training_loss).backward()
                training_loss.backward()
                if clip_gradient_norm_ub is not None and clip_gradient_norm_ub > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_gradient_norm_ub)
                optimizer.step()
                # scaler.step(optimizer)
                # scaler.update()

            # print("cleaning up.")
            epoch_training_loss += training_loss.item() * training_y.shape[0] / n_training_examples

        with timer("LR-Step", enabled=timer_profile):
            lr_scheduler.step()
            current_lr = lr_scheduler.get_last_lr()[0]

        # option implementation
        if epoch % print_every == 0:
            """ Evaluate test loss. """
            epoch_test_loss = _compute_test_loss()

            if print_progress:
                print(
                    f"Epoch {epoch} | "
                    f"Training Loss: {epoch_training_loss} | "
                    f"Test Loss: {epoch_test_loss} | "
                    f"LR = {current_lr}"
                )

            epoch_history.append(epoch)
            training_loss_history.append(epoch_training_loss)
            test_loss_history.append(epoch_test_loss)

        if (epoch % checkpoint_every == 0) and (epoch > 0):
            """ Save the model and optimizer states. """
            filepath = checkpoint_dir / f"checkpoint_{epoch}.pt"
            save_checkpoint(
                epoch, model, optimizer, lr_scheduler, training_data_rng,
                epoch_history, training_loss_history, test_loss_history,
                filepath,
            )
    pbar.close()

    if loss_plot_path is not None:
        fig, ax = plt.subplots(1, 1)
        ax.plot(epoch_history, training_loss_history, color='red', label='train')
        ax.plot(epoch_history, test_loss_history, color='green', label='test')
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epoch")
        ax.legend()
        plt.savefig(loss_plot_path, bbox_inches="tight")

    # finally, save the final checkpoint file.
    if epoch is not None:
        model.eval()
        filepath = checkpoint_dir / f"checkpoint_{epoch}.pt"
        save_checkpoint(
            epoch, model, optimizer, lr_scheduler, training_data_rng,
            epoch_history, training_loss_history, test_loss_history,
            filepath,
        )


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    data_rng: torch.Generator,
    epoch_history: List[int],
    training_history: List[float],
    test_history: List[float],
    filepath: Path,
) -> None:
    """Save training checkpoint including RNG state"""
    checkpoint: Dict[str, Any] = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'data_rng_state': data_rng.get_state(),
        'epoch_history': epoch_history,
        'training_history': training_history,
        'test_history': test_history,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at epoch {epoch}")


def load_checkpoint(
        filepath: Path,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        data_rng: torch.Generator
) -> Tuple[int, List[int], List[float], List[float]]:
    """
    Load training checkpoint and restore RNG state.
    The input objects rae modified in-place from the checkpoint file.
    :return: the last completed epoch.
    """
    checkpoint: Dict[str, Any] = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    data_rng.set_state(checkpoint['data_rng_state'])

    last_epoch: int = checkpoint['epoch']
    epoch_history: List[int] = checkpoint['epoch_history']
    training_history: List[float] = checkpoint['training_history']
    test_history: List[float] = checkpoint['test_history']
    print(f"Checkpoint loaded, last epoch completed = {last_epoch}")
    return last_epoch, epoch_history, training_history, test_history