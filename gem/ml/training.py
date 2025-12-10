from typing import *
from pathlib import Path

from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, GradScaler, autocast

from gem.mpa import AbstractMetaphlanDataset
from gem.util.timer import timer
from gem.ml.dataloader.data_loader import MetaphlanDataLoader


def main_training_loop(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        train_dset: AbstractMetaphlanDataset,
        test_dset: AbstractMetaphlanDataset,
        loss_fn: nn.Module,
        num_workers: Optional[int] = 0,
        batch_size: int = 5,
        num_epochs: int = 5000,
        print_progress: bool = True,
        print_every: int = 50,
        loss_plot_path: Path = Optional[None],
        auto_mixed_precision: bool = False,
        rng_seed: int = 314159,
):
    """
    Train an input model on the given dataset. Uses torch.optim.Adam by default.
    This function's training loop mutates the parameters of the `model` object.

    :param model: The model to train. Should be designed to take a tensor of shape (*, MAX_NUM_SGBS, MAX_NUM_MARKERS, MARKER_EMBED_DIM) as input, and output (*, MAX_NUM_SGBS) which represents a batched vector of logits.
    :param optimizer: The torch.optim.Optimizer instance used to optimize.
    :param lr_scheduler: the torch.optim.lr_scheduler instance used to tune learning rates.
    :param train_dset: The object that specifies the training dataset.
    :param test_dset: The object that specifies the test dataset.
    :param loss_fn: The loss function to use for optimization.
    :param num_workers: Specify the number of workers to load data in parallel. Note that a separate dataloader is created for train/test; so if both are provided the actual # of workers is 2 times the specified value.
    :param batch_size: The size of each batch.
    :param num_epochs: The total number of epochs to train. (1 epoch is a full pass on the entire training dataset, after batching.)
    :param print_progress: Indicate whether to print loss values periodically to stdout. Specify `print_every` to change how often this occurs.
    :param print_every: Indicate how often to print progress as a debug message to stdout. Only relevant if `print_progress` is set to true.
    :param loss_plot_path: If provided, plots the training loss/test loss history. Test loss is only plotted if test_df is provided.
    :param rng_seed: The random generator seed to use for training. Specify for reproducibility. (default: 314159)
    """

    """ Initialization. """
    plot_history = (loss_plot_path is not None)
    epoch_history = []
    training_loss_history = []
    test_loss_history = []
    scaler = GradScaler("cuda", enabled=auto_mixed_precision)

    """ Initialize dataset objects. """

    def _seed_worker(worker_id):
        worker_seed = worker_id + rng_seed
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_rng = torch.Generator()
    train_rng.manual_seed(rng_seed)
    train_dloader = MetaphlanDataLoader(
        dataset=train_dset,
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
        generator=train_rng, worker_init_fn=_seed_worker, drop_last=True,
    )
    # also set RNG seed for dropout reproducibility.
    torch.manual_seed(rng_seed + 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng_seed + 1)

    print(f"Training dataset size: {len(train_dset)}")
    print(f"Number of training batches: {len(train_dloader)}")

    test_dloader = MetaphlanDataLoader(
        dataset=test_dset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
        generator=train_rng, worker_init_fn=_seed_worker, drop_last=True,
    )
    print(f"Test dataset size: {len(test_dset)}")
    print(f"Number of test batches: {len(test_dloader)}")

    """ Training loop -- Optimize using batches. """

    def _compute_test_loss() -> float:
        total_test_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch_idx, (test_batch_features, test_marker_mask, test_sgb_mask, test_y) in enumerate(test_dloader):
                with autocast(device_type='cuda', enabled=auto_mixed_precision):
                    with timer("[Model]"):
                        test_y_hat = model(
                            test_batch_features.cuda(non_blocking=True),
                            test_marker_mask.cuda(non_blocking=True),
                            test_sgb_mask.cuda(non_blocking=True),
                        )

                    with timer("[Loss Eval]"):
                        # assert test_y_hat.shape == test_y.shape, f"Neural Network output and ground truth have different shapes: {test_y_hat.shape} (NN) vs {test_y.shape} (truth)"
                        batch_loss = loss_fn(
                            nn.functional.log_softmax(test_y_hat, dim=-1),  # log pred probabilities
                            torch.log(test_y.cuda(non_blocking=True))  # log target probabilities
                        )

                total_test_loss += scaler.scale(batch_loss).item() * test_y.shape[0]

                # clear some space for next batch.
                del test_y_hat
        return total_test_loss / len(test_dset)  # divide by total dataset size.

    test_dset.track_runtime = True
    print(f"Initial Test Loss: {_compute_test_loss()}")
    test_dset.track_runtime = False

    current_lr = "n/a"
    for epoch in tqdm(range(num_epochs)):
        epoch_training_loss = 0.0
        for batch_idx, (training_batch_features, training_marker_mask, training_sgb_mask, training_y) in enumerate(
                train_dloader):
            model.train()
            optimizer.zero_grad()

            with autocast(device_type='cuda', enabled=auto_mixed_precision):
                training_y_hat = model(
                    training_batch_features.cuda(non_blocking=True),
                    training_marker_mask.cuda(non_blocking=True),
                    training_sgb_mask.cuda(non_blocking=True),
                )
                # assert training_y_hat.shape == training_y.shape, f"Neural Network output and ground truth have different shapes: {training_y_hat.shape} (NN) vs {training_y.shape} (truth)"
                training_loss = loss_fn(
                    nn.functional.log_softmax(training_y_hat, dim=-1),  # log pred probabilities
                    torch.log(training_y.cuda(non_blocking=True))  # log target probabilities
                )

            scaler.scale(training_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr_scheduler.step()
            current_lr = lr_scheduler.get_last_lr()[0]

            # print("cleaning up.")
            epoch_training_loss += training_loss.item() * training_y.shape[0]

            # clear some space for next batch.
            del training_y_hat
            del training_loss
        epoch_training_loss /= len(train_dloader.dataset)

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

            if plot_history:
                epoch_history.append(epoch)
                training_loss_history.append(epoch_training_loss)
                test_loss_history.append(epoch_test_loss)

    if loss_plot_path is not None:
        fig, ax = plt.subplots(1, 1)
        ax.plot(epoch_history, training_loss_history, color='red', label='train')
        ax.plot(epoch_history, test_loss_history, color='green', label='test')
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epoch")
        ax.legend()
        plt.savefig(loss_plot_path, bbox_inches="tight")