import torch


def safe_kl_div_loss(log_pred, log_target, reduction='batchmean'):
    """
    Safe KL divergence loss that handles -inf in log_target.
    Matches PyTorch's KLDivLoss interface.
    Assumes that both 'pred' and 'target' are both LOG-probabilities.
    """
    # Compute KL elementwise
    kl_elementwise = torch.exp(log_target) * (log_target - log_pred)

    # Mask out -inf contributions
    finite_mask = torch.isfinite(log_target)
    kl_elementwise = torch.where(finite_mask, kl_elementwise, torch.tensor(0.0))

    if reduction == 'batchmean':
        return kl_elementwise.sum() / log_pred.shape[0]
    elif reduction == 'mean':
        return kl_elementwise.mean()
    elif reduction == 'sum':
        return kl_elementwise.sum()
    else:
        return kl_elementwise
