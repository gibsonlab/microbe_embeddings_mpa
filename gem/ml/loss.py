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


def safe_cross_entropy_loss(log_pred, log_target, reduction='batchmean'):
    """
    Safe cross-entropy loss.
    """
    # Compute KL elementwise
    entropy_elementwise = torch.exp(log_target) * (-log_pred)

    # Mask out -inf contributions
    finite_mask = torch.isfinite(log_target)
    entropy_elementwise = torch.where(finite_mask, entropy_elementwise, torch.tensor(0.0))

    if reduction == 'batchmean':
        return entropy_elementwise.sum() / log_pred.shape[0]
    elif reduction == 'mean':
        return entropy_elementwise.mean()
    elif reduction == 'sum':
        return entropy_elementwise.sum()
    else:
        return entropy_elementwise


def safe_mse_log_loss(log_pred, log_target, reduction='batchmean'):
    """
    Safe mean-squared-error loss, but gracefully excluding -inf in log_pred.
    The Mean-squared-error is evaluated in LOG-probability space.
    It is assumed that wherever log_target is -inf, log_pred is -inf also. (and the inverse is also true about non-inf)
    """
    finite_mask = torch.isfinite(log_target)

    # Compute squared-error elementwise
    n_sgbs = torch.sum(finite_mask, dim=-1)
    diff_elementwise = n_sgbs.reciprocal().unsqueeze(-1) * torch.square(log_target - log_pred)

    # Mask out -inf contributions
    diff_elementwise = torch.where(finite_mask, diff_elementwise, torch.tensor(0.0))

    if reduction == 'batchmean':
        return diff_elementwise.sum() / log_pred.shape[0]
    elif reduction == 'mean':
        return diff_elementwise.mean()
    elif reduction == 'sum':
        return diff_elementwise.sum()
    else:
        return diff_elementwise


def safe_mse_loss(log_pred, log_target, reduction='batchmean'):
    """
    Safe mean-squared-error loss, but gracefully excluding -inf in log_pred.
    The Mean-squared-error is evaluated in probability space.
    It is assumed that wherever log_target is -inf, log_pred is -inf also. (and the inverse is also true about non-inf)
    """
    finite_mask = torch.isfinite(log_target)
    pred = torch.exp(log_pred)
    target = torch.exp(log_target)

    # Compute squared-error elementwise
    n_sgbs = torch.sum(finite_mask, dim=-1)
    diff_elementwise = n_sgbs.reciprocal().unsqueeze(-1) * torch.square(target - pred)

    # Mask out -inf contributions
    diff_elementwise = torch.where(finite_mask, diff_elementwise, torch.tensor(0.0))

    if reduction == 'batchmean':
        return diff_elementwise.sum() / log_pred.shape[0]
    elif reduction == 'mean':
        return diff_elementwise.mean()
    elif reduction == 'sum':
        return diff_elementwise.sum()
    else:
        return diff_elementwise
